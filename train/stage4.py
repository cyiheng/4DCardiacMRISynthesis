"""Stage 4 training script for flow-aware latent diffusion training.

This script trains a conditional diffusion model over latent motion fields,
using a pretrained VAE and latent motion predictor to supervise the model
with both spatial and temporal consistency.
"""

import os
import sys
import glob
import re
import shutil
import random
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# --- MONAI Imports ---
from monai.transforms import (
    Compose,
    LoadImaged,
    ScaleIntensityRangePercentilesd,
    EnsureTyped,
    Spacingd,
    SpatialPadd,
    CenterSpatialCropd
)

# --- Accelerate, Diffusers & Transformers Imports ---
from accelerate import Accelerator
from diffusers import DDIMScheduler, UNet3DConditionModel
from diffusers.optimization import get_scheduler
import bitsandbytes as bnb

# --- CUSTOM IMPORTS ---
from models.custom_AEKL import DualDecoderVAE
from models.latent_motion import LatentFeatureMotionPredictor
from models.conditioning import SinusoidalPositionalEmbedding, ClinicalConditioningModel
from datasets.latent_datasets import ConditionedLatentFlowDataset
from utils.common import normalize_scalars
from utils.visualization import plot_and_save_learning_curve

# =========================================================================================
# --- 1. CONFIGURATION ---
# =========================================================================================
class Config:
    # --- Paths ---
    latent_cache_dir = "data/latents/" 
    original_image_data_dir = "data/images"
    label_data_dir = 'data/labels/'
    
    # Unified Metadata CSVs
    train_csv = "data/train_split_final.csv" 
    test_csv = "data/test_split_final.csv" 

    # Pretrained Models
    vae_model_path = "./results/001_VAE3D_dualdecoder/checkpoints/best_autoencoder_model.pth"
    lfm_flow_predictor_path = "./results/002_LatentMotion/checkpoints/flow_3.pth"
    flow_scale = 1.65115030
    
    output_dir = "results/004_FlowDiffusion/"
    os.makedirs(output_dir, exist_ok=True)
    script_path = os.path.abspath(__file__)
    script_name = os.path.basename(script_path)
    shutil.copyfile(script_path, os.path.join(output_dir, script_name))
    resume_from_checkpoint = "latest"

    vae_scale = 0.643433 # From Pre-encoding

    # --- Model Params ---
    latent_channels = 8
    flow_channels = 8
    unet_in_channels = latent_channels + flow_channels + 1  # +1 mask
    spatial_size = (192, 192, 16) # Matches VAE Training
    target_pixdim = (1.0, 1.0, 10.0) 
    num_classes = 4
    
    # Conditioning Params
    cross_attention_dim = 512 
    num_diagnosis_classes = 7 # 6 classes + 1 for CFG mask
    
    # --- Diffusion Params ---
    num_train_timesteps = 1000
    num_inference_steps = 50
    beta_schedule = "scaled_linear"
    
    # --- Training Params ---
    max_train_steps = 300000
    batch_size = 2
    learning_rate = 2e-5
    lr_scheduler_type = "constant_with_warmup"
    lr_warmup_steps = 500
    adam_beta1 = 0.9; adam_beta2 = 0.999; adam_weight_decay = 1e-2
    
    validation_steps = 10000
    checkpointing_steps = 20000
    
    # --- Normalization Constants ---
    MAX_EDV = 600.0  
    MAX_ESV = 600.0  
    MAX_SLICES = 20.0 
    SKIP_FRAMES = 1
    
    # Probability to drop clinical condition for Classifier-Free Guidance
    cfg_dropout_prob = 0.1 

cfg = Config()

# Update Diagnosis Map to match your static generator
DIAGNOSIS_MAP = { "Unknown": 0, "NOR": 1, "MINF": 2, "DCM": 3, "HCM": 4, "RV": 5 }


# =========================================================================================
# --- 2. DATA PREPARATION ---
# =========================================================================================

def parse_combined_csv(csv_path):
    """Parse a combined metadata CSV and return patient metadata for motion pairing."""
    if not os.path.exists(csv_path):
        print(f"Warning: CSV {csv_path} not found.")
        return {}, set()

    df = pd.read_csv(csv_path)
    valid_pids = set()
    metadata_map = {}
    
    for _, row in df.iterrows():
        dataset = row['dataset']
        pid_raw = str(row['pid'])
        prefix = f"{dataset}_{pid_raw}" if dataset == "ACDC" else f"DSB_{pid_raw}"

        if dataset == "DSB" and int(row.get("n_slices", 10)) <= 4:
            continue

        if dataset == "ACDC":
            ed_frame = int(row['ed_frame']) - 1 if pd.notna(row.get('ed_frame')) else 0
            es_frame = int(row['es_frame']) - 1 if pd.notna(row.get('es_frame')) else None
        else:
            ed_frame = 0 
            es_raw = row.get('es_frame')
            es_frame = int(float(es_raw)) - 1 if pd.notna(es_raw) else None

        if es_frame is None: continue 

        ef = float(row.get('ef', -1.0))
        n_slices = float(row.get('n_slices', 10.0))
        edv = float(row.get('diastole_volume', row.get('edv', -1.0)))
        esv = float(row.get('systole_volume', row.get('esv', -1.0)))
        pathology = row.get("pathology", "Unknown") if dataset == "ACDC" else "Unknown"

        metadata_map[prefix] = {
            "pathology": pathology,
            "ef": ef, "edv": edv, "esv": esv, "n_slices": n_slices,
            "ed_frame": ed_frame, "es_frame": es_frame
        }
        valid_pids.add(prefix)
            
    return metadata_map, valid_pids

def prepare_datalist_flow_diffusion_3d(cfg, frame_skip_interval=3):
    """Build training and validation pairs for flow diffusion using latent pairs."""
    print(f"--- Preparing Motion Datalist (Pairs with Time + Structured Condition) ---")
    
    # 1. Parse Consolidated Metadata
    meta_tr, pids_tr = parse_combined_csv(cfg.train_csv)
    meta_val, pids_val = parse_combined_csv(cfg.test_csv)
    all_metadata = {**meta_tr, **meta_val}
    
    print(f"Total Train Subjects: {len(pids_tr)}")
    print(f"Total Val Subjects: {len(pids_val)}")

    # 2. Scan Latent Files & Group by Patient
    all_latent_files = glob.glob(os.path.join(cfg.latent_cache_dir, "*.pt"))
    pattern = re.compile(r"^(ACDC_patient\d+|DSB_\d+)_frame_(\d+)\.pt$")
    
    patient_volumes = defaultdict(list)
    for f_path in all_latent_files:
        filename = os.path.basename(f_path)
        match = pattern.match(filename)
        if match:
            pid = match.group(1)
            frame_idx = int(match.group(2))
            patient_volumes[pid].append({"path": f_path, "frame_idx": frame_idx})

    # 3. Build Datalist
    def build_datalist(target_pids):
        dataset_list = []
        for pid in target_pids:
            if pid not in patient_volumes: continue
            
            volumes = patient_volumes[pid]
            meta = all_metadata.get(pid, {})
            
            ed_idx = meta.get('ed_frame')
            es_idx = meta.get('es_frame')
            if ed_idx is None or es_idx is None: continue
            
            ref_vol = next((v for v in volumes if v['frame_idx'] == ed_idx), None)
            if not ref_vol: continue
            
            max_frame_idx = max([v['frame_idx'] for v in volumes])
            num_frames_total = max_frame_idx + 1 
            
            contraction_duration = (es_idx - ed_idx + num_frames_total) % num_frames_total
            relaxation_duration = num_frames_total - contraction_duration
            if contraction_duration <= 0 or relaxation_duration <= 0: continue

            scalars = normalize_scalars(
                meta.get('ef', -1), meta.get('edv', -1), 
                meta.get('esv', -1), meta.get('n_slices', 10)
            )
            diag_id = DIAGNOSIS_MAP.get(meta.get('pathology'), 0)
            
            volumes.sort(key=lambda x: x['frame_idx'])

            for i, dri_vol in enumerate(volumes):
                curr_idx = dri_vol['frame_idx']
                dist_from_ed = (curr_idx - ed_idx + num_frames_total) % num_frames_total
                
                if curr_idx == ed_idx: time_val = 0.0
                elif dist_from_ed <= contraction_duration: time_val = 100.0 * (dist_from_ed / contraction_duration)
                else: time_val = 100.0 + 100.0 * ((dist_from_ed - contraction_duration) / relaxation_duration)
                    
                orig_dri_name = os.path.basename(dri_vol["path"]).replace(".pt", ".nii.gz")
                orig_ref_name = os.path.basename(ref_vol["path"]).replace(".pt", ".nii.gz")
                
                dataset_list.append({
                    "ref_latent_path": ref_vol["path"],
                    "dri_latent_path": dri_vol["path"],
                    "ref_image_path": os.path.join(cfg.original_image_data_dir, orig_ref_name),
                    "dri_image_path": os.path.join(cfg.original_image_data_dir, orig_dri_name),
                    "ref_label_path": os.path.join(cfg.label_data_dir, orig_ref_name),
                    "dri_label_path": os.path.join(cfg.label_data_dir, orig_dri_name),
                    "time_value": time_val,
                    "diag_id": diag_id,
                    "scalars": scalars
                })
        return dataset_list

    train_data = build_datalist(pids_tr)
    val_data = build_datalist(pids_val)
    
    print(f"Training Pairs: {len(train_data)}")
    print(f"Validation Pairs: {len(val_data)}")
    return train_data, val_data


# =========================================================================================
# --- 3. VISUALIZATION ---
# =========================================================================================

@torch.no_grad()
def generate_and_visualize_comparison(
    accelerator, unet, vae, flow_predictor, 
    cond_model, time_signal_embedder, 
    scheduler, scaling_factor, flow_scale_factor,
    image_transforms,
    ref_image_path, dri_image_path,
    ref_label_path, dri_label_path, 
    global_step, output_dir,
    guidance_scale=5.0
):
    """Generate a comparison visualization for a fixed validation pair."""
    print(f"\n--- Step {global_step}: Generating and visualizing fixed pair comparison ---")
    unet.eval(); vae.eval(); flow_predictor.eval()
    device = accelerator.device
    save_dir = os.path.join(output_dir, "validation_samples"); os.makedirs(save_dir, exist_ok=True)
    
    test_conditions = {"name": "DCM_14slice", "diag": 3, "raw": [29.0, 223.0, 156.0, 14.0]}
    target_diag_id = test_conditions['diag']
    raw = test_conditions["raw"]
    target_scalars = normalize_scalars(raw[0], raw[1], raw[2], raw[3])
    target_time_val = 100

    try:
        raw_img = nib.load(ref_image_path)
        true_slices = raw_img.shape[-1]
    except:
        true_slices = 10
    
    input_dict = {
        "ref_image": ref_image_path, "dri_image": dri_image_path,
        "ref_label": ref_label_path if os.path.exists(ref_label_path) else None, 
        "dri_label": dri_label_path if os.path.exists(dri_label_path) else None
    }
    input_dict = {k:v for k,v in input_dict.items() if v is not None}
    gt_data = image_transforms(input_dict)
    
    ref_image_gt = gt_data["ref_image"].to(device).unsqueeze(0)
    dri_image_gt = gt_data["dri_image"].to(device).unsqueeze(0)

    if "dri_label" in gt_data:
        dri_label_gt = gt_data["dri_label"].to(device).unsqueeze(0)
    else:
        dri_label_gt = torch.zeros_like(dri_image_gt)

    _, _, ref_latent_gt, _ = vae(ref_image_gt)
    _, _, dri_latent_gt, _ = vae(dri_image_gt)
    ref_latent_gt = ref_latent_gt * scaling_factor
    dri_latent_gt = dri_latent_gt * scaling_factor
    
    _, _, h, w, d = ref_latent_gt.shape
    mask = torch.zeros((1, 1, h, w, d), device=accelerator.device)
    
    valid_z = min(true_slices, d)
    diff = d - valid_z
    start = diff // 2
    end = start + valid_z
    mask[..., start:end] = 1.0

    affine = gt_data["dri_image_meta_dict"]["affine"]
    feat_motion_gt = flow_predictor(ref_latent_gt, dri_latent_gt)

    diag_cond = torch.tensor([target_diag_id], device=device)
    scalars_cond = torch.tensor([target_scalars], dtype=torch.float32, device=device)
    diag_uncond = torch.tensor([6], device=device) 
    scalars_uncond = torch.zeros_like(scalars_cond)
    
    time_val_tensor = torch.tensor([[target_time_val]], dtype=torch.float32, device=device)
    time_emb = time_signal_embedder(time_val_tensor) 
    if time_emb.ndim == 2:
        time_emb = time_emb.unsqueeze(1) # Ensure shape is (1, 1, 512)

    emb_cond_clinical = cond_model(diag_cond, scalars_cond)     
    emb_uncond_clinical = cond_model(diag_uncond, scalars_uncond) 
    
    context_cond = torch.cat([emb_cond_clinical, time_emb], dim=1)     
    context_uncond = torch.cat([emb_uncond_clinical, time_emb], dim=1) 
    context_input = torch.cat([context_uncond, context_cond]) 

    latent_shape = (1, cfg.latent_channels, *ref_latent_gt.shape[2:])
    transformation_pred_scaled = torch.randn(latent_shape, device=device)
    scheduler.set_timesteps(cfg.num_inference_steps)
    
    for t in tqdm(scheduler.timesteps, desc="Sampling with CFG", leave=False):
        latents_input = torch.cat([transformation_pred_scaled] * 2) 
        ref_input = torch.cat([ref_latent_gt] * 2)                  
        mask_input = torch.cat([mask] * 2)                          
        
        model_input = torch.cat([latents_input, ref_input, mask_input], dim=1)
        t_batch = t.expand(2).to(device)
        noise_pred = unet(model_input, t_batch, encoder_hidden_states=context_input).sample
        
        noise_uncond, noise_cond = noise_pred.chunk(2)
        noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
        transformation_pred_scaled = scheduler.step(noise_pred, t, transformation_pred_scaled).prev_sample

    feat_motion_pred = transformation_pred_scaled / flow_scale_factor

    z_warped = ref_latent_gt + feat_motion_pred
    img_pred = vae.decoder_img(z_warped / scaling_factor)
    seg_logits = vae.decoder_seg(z_warped / scaling_factor)
    seg_pred = torch.argmax(seg_logits, dim=1, keepdim=True).float()

    def crop_z(tensor_4d, n_original):
        current_z = tensor_4d.shape[-1]
        if n_original >= current_z: return tensor_4d
        diff = current_z - n_original
        start = diff // 2
        return tensor_4d[..., start : start + n_original]
    
    def to_np_cropped(t): 
        t_cropped = crop_z(t.squeeze(0), true_slices) 
        arr = t_cropped.cpu().numpy()
        return np.squeeze(arr)

    ref_img_np = to_np_cropped(ref_image_gt)
    dri_img_np = to_np_cropped(dri_image_gt)
    dri_lbl_np = to_np_cropped(dri_label_gt)
    pred_img_np = to_np_cropped(img_pred)
    pred_seg_np = to_np_cropped(seg_pred)

    d, h, w = ref_img_np.shape
    slice_idx = w // 2 

    fig, axes = plt.subplots(4, 2, figsize=(10, 16))
    plt.suptitle(f"Conditioned Gen (Step {global_step}) - {true_slices} slices", fontsize=16)
    
    def plot_row(row, title, gt, pred, cmap="gray", vmin=None, vmax=None):
        axes[row, 0].imshow(np.rot90(gt[:, :, slice_idx]), cmap=cmap, vmin=vmin, vmax=vmax)
        axes[row, 1].imshow(np.rot90(pred[:, :, slice_idx]), cmap=cmap, vmin=vmin, vmax=vmax)
        axes[row, 0].set_ylabel(title)

    plot_row(0, "Image", dri_img_np, pred_img_np)
    plot_row(1, "Segmentation", dri_lbl_np, pred_seg_np, cmap="tab10", vmin=0, vmax=3)
    
    mag_gt = torch.mean(torch.abs(feat_motion_gt), dim=1).squeeze().cpu().numpy()
    mag_pred = torch.mean(torch.abs(feat_motion_pred), dim=1).squeeze().cpu().numpy()
    mag_gt = crop_z(torch.tensor(mag_gt), true_slices).numpy()
    mag_pred = crop_z(torch.tensor(mag_pred), true_slices).numpy()
    plot_row(2, "Motion Mag", mag_gt, mag_pred, cmap="magma")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"vis_step_{global_step:08d}.png"))
    plt.close()
    
    nib.save(nib.Nifti1Image(dri_img_np, affine), os.path.join(save_dir, f"step_{global_step:08d}_dri_img_np.nii.gz"))
    nib.save(nib.Nifti1Image(pred_img_np, affine), os.path.join(save_dir, f"step_{global_step:08d}_pred_img.nii.gz"))
    nib.save(nib.Nifti1Image(pred_seg_np, affine), os.path.join(save_dir, f"step_{global_step:08d}_pred_seg.nii.gz"))
    
    unet.train()

# =========================================================================================
# --- 4. MAIN ---
# =========================================================================================
def main():
    """Set up and execute the stage 4 training loop for flow-guided diffusion."""
    accelerator = Accelerator(mixed_precision="fp16")
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(os.path.join(cfg.output_dir, "checkpoints"), exist_ok=True)

    # --- Load Models ---
    vae = DualDecoderVAE(latent_dim=cfg.latent_channels, num_classes=cfg.num_classes)
    vae.load_state_dict(torch.load(cfg.vae_model_path, map_location="cpu"))
    vae.to(accelerator.device);vae.eval(); vae.requires_grad_(False)
    
    lfm_flow_predictor = LatentFeatureMotionPredictor(in_channels=2 * cfg.latent_channels, latent_channels=cfg.latent_channels)
    lfm_flow_predictor.load_state_dict(torch.load(cfg.lfm_flow_predictor_path, map_location="cpu"))
    lfm_flow_predictor.to(accelerator.device); lfm_flow_predictor.eval(); lfm_flow_predictor.requires_grad_(False)

    cond_model = ClinicalConditioningModel(
        embedding_dim=cfg.cross_attention_dim, 
        num_diagnosis=cfg.num_diagnosis_classes
    )
    time_signal_embedder = SinusoidalPositionalEmbedding(cfg.cross_attention_dim).to(accelerator.device)

    unet = UNet3DConditionModel(
        in_channels=cfg.unet_in_channels,
        out_channels=cfg.flow_channels,
        block_out_channels=(256, 512, 768, 768),
        layers_per_block=2,
        cross_attention_dim=cfg.cross_attention_dim, 
        attention_head_dim=64,
        down_block_types=("CrossAttnDownBlock3D", "CrossAttnDownBlock3D", "CrossAttnDownBlock3D", "DownBlock3D"),
        up_block_types=("UpBlock3D", "CrossAttnUpBlock3D", "CrossAttnUpBlock3D", "CrossAttnUpBlock3D"),
    )
    noise_scheduler = DDIMScheduler(num_train_timesteps=cfg.num_train_timesteps, beta_schedule=cfg.beta_schedule, clip_sample=False)

    train_datalist, val_datalist = prepare_datalist_flow_diffusion_3d(cfg, frame_skip_interval=cfg.SKIP_FRAMES)
    train_dataset = ConditionedLatentFlowDataset(train_datalist, cfg.vae_scale)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    optimizer = bnb.optim.AdamW8bit(
        list(unet.parameters()) + list(cond_model.parameters()),
        lr=cfg.learning_rate, 
        weight_decay=cfg.adam_weight_decay, 
    )
    lr_scheduler = get_scheduler(cfg.lr_scheduler_type, optimizer=optimizer, num_warmup_steps=cfg.lr_warmup_steps, num_training_steps=cfg.max_train_steps)
    
    unet, cond_model, optimizer, train_loader, lr_scheduler = accelerator.prepare(unet, cond_model, optimizer, train_loader, lr_scheduler)

    val_transforms = Compose([
        LoadImaged(keys=["dri_image", "ref_image", "dri_label", "ref_label"], image_only=False, ensure_channel_first=True, allow_missing_keys=True),
        Spacingd(keys=["dri_image", "ref_image", "dri_label", "ref_label"], pixdim=cfg.target_pixdim, mode=("bilinear", "bilinear", "nearest", "nearest"), allow_missing_keys=True),
        SpatialPadd(keys=["dri_image", "ref_image", "dri_label", "ref_label"], spatial_size=cfg.spatial_size, method="symmetric", mode="edge", allow_missing_keys=True),
        CenterSpatialCropd(keys=["dri_image", "ref_image", "dri_label", "ref_label"], roi_size=cfg.spatial_size, allow_missing_keys=True),
        ScaleIntensityRangePercentilesd(keys=["dri_image", "ref_image"], lower=0.5, upper=99.5, b_min=-1.0, b_max=1.0, clip=True, allow_missing_keys=True),
        EnsureTyped(keys=["dri_image", "ref_image", "dri_label", "ref_label"], dtype=torch.float32, allow_missing_keys=True),
    ])

    # --- DYNAMIC VALIDATION SAMPLE SELECTION ---
    if len(val_datalist) > 0:
        # Prioritize picking a validation pair that has existing labels on disk
        labeled_val_pairs = [
            p for p in val_datalist 
            if p["ref_label_path"] and os.path.exists(p["ref_label_path"])
            and p["dri_label_path"] and os.path.exists(p["dri_label_path"])
        ]
        
        if labeled_val_pairs:
            fixed_pair = labeled_val_pairs[0]  # Pick the first fully-labeled pair
        else:
            fixed_pair = val_datalist[0]        # Fallback to the first overall pair on disk
            
        fixed_ref_image = fixed_pair["ref_image_path"]
        fixed_dri_image = fixed_pair["dri_image_path"]
        fixed_ref_label = fixed_pair["ref_label_path"] if fixed_pair["ref_label_path"] else ""
        fixed_dri_label = fixed_pair["dri_label_path"] if fixed_pair["dri_label_path"] else ""
        
        print(f"\n[INFO] Selected dynamic validation pair for visual logging:")
        print(f"       Ref: {os.path.basename(fixed_ref_image)}")
        print(f"       Dri: {os.path.basename(fixed_dri_image)}\n")
    else:
        raise RuntimeError("Validation dataset `val_datalist` is empty. Check split CSV.")

    if not cfg.flow_scale:
        print("Calculating flow statistics on training set...")
        sampled_flow_values = []
        samples_count = 0
        target_samples = len(train_loader)
        lfm_flow_predictor.eval()

        with torch.no_grad():
            for batch in train_loader:
                ref_latents = batch["ref_latent"] 
                dri_latents = batch["dri_latent"]
                mask = batch["mask"].to(accelerator.device)
                
                flow_batch = lfm_flow_predictor(ref_latents, dri_latents) 
                mask_expanded = mask.expand_as(flow_batch) > 0.5
                valid_flow_values = flow_batch[mask_expanded]
                
                sampled_flow_values.append(valid_flow_values.cpu())
                samples_count += ref_latents.shape[0]
                if samples_count >= target_samples: break
        
        all_valid_flows = torch.cat(sampled_flow_values, dim=0)
        std_dev = all_valid_flows.std()
        
        if std_dev == 0: cfg.flow_scale = 1.0
        else: cfg.flow_scale = 1.0 / std_dev.item()
            
        print(f"Calculated Flow Scaling Factor: {cfg.flow_scale:.8f}")
        del all_valid_flows, sampled_flow_values, batch, ref_latents, dri_latents, flow_batch, valid_flow_values, mask_expanded
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    global_step = 0 
    if cfg.resume_from_checkpoint == "latest":
        dirs = [d for d in os.listdir(os.path.join(cfg.output_dir, "checkpoints")) if "checkpoint_step_" in d]
        if dirs:
            dirs.sort(key=lambda x: int(x.split("_")[-1]))
            accelerator.load_state(os.path.join(os.path.join(cfg.output_dir, "checkpoints"), dirs[-1]))
            global_step = int(dirs[-1].split("_")[-1])
            print(global_step, dirs[-1])
            input("continue?")

    step_loss_history = []
    progress_bar = tqdm(
        range(cfg.max_train_steps), 
        initial=global_step, 
        total=cfg.max_train_steps, 
        disable=not accelerator.is_local_main_process
    )

    while global_step < cfg.max_train_steps:
        unet.train()
        cond_model.train() 
        
        for batch in train_loader:
            with accelerator.accumulate(unet):
                ref_latents = batch["ref_latent"]
                dri_latents = batch["dri_latent"]
                mask = batch["mask"]
                diag_ids = batch["diag_id"]
                scalars = batch["scalars"]
                time_vals = batch["time_val"].to(torch.float32).unsqueeze(1)

                with torch.no_grad():
                    gt_flow = lfm_flow_predictor(ref_latents, dri_latents) * cfg.flow_scale
                    is_ed_frame = (time_vals.squeeze() < 1e-4) 
                    if is_ed_frame.any():
                        gt_flow[is_ed_frame] = 0.0
                
                if random.random() < cfg.cfg_dropout_prob:
                    diag_ids = torch.full_like(diag_ids, 6) 
                    scalars = torch.zeros_like(scalars)

                clinical_emb = cond_model(diag_ids, scalars)
                time_emb = time_signal_embedder(time_vals)
                if time_emb.ndim == 2:
                    time_emb = time_emb.unsqueeze(1) # Shape: (B, 1, 512)
                context = torch.cat([clinical_emb, time_emb], dim=1)

                noise = torch.randn_like(gt_flow)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (gt_flow.shape[0],), device=gt_flow.device).long()
                noisy_flow = noise_scheduler.add_noise(gt_flow, noise, timesteps)
                
                model_input = torch.cat([noisy_flow, ref_latents, mask], dim=1)
                
                noise_pred = unet(model_input, timesteps, encoder_hidden_states=context).sample
                loss = F.mse_loss(noise_pred.float(), noise.float())

                accelerator.backward(loss)
                if accelerator.sync_gradients: accelerator.clip_grad_norm_(unet.parameters(), 1.0)
                optimizer.step(); lr_scheduler.step(); optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1); global_step += 1
                if accelerator.is_main_process:
                    step_loss_history.append(loss.item())
                    
                    if global_step % cfg.validation_steps == 0:
                        plot_and_save_learning_curve(step_loss_history, os.path.join(cfg.output_dir, "loss.png"), 100)
                        generate_and_visualize_comparison(
                            accelerator, unet, vae, lfm_flow_predictor, 
                            cond_model, time_signal_embedder,
                            noise_scheduler, cfg.vae_scale, cfg.flow_scale, val_transforms,
                            fixed_ref_image, fixed_dri_image, fixed_ref_label, fixed_dri_label,
                            global_step=global_step, 
                            output_dir = cfg.output_dir
                        )
                        torch.cuda.empty_cache()
                    
                    if global_step % cfg.checkpointing_steps == 0:
                        accelerator.save_state(os.path.join(cfg.output_dir, f"checkpoints/checkpoint_step_{global_step}"))

            if global_step >= cfg.max_train_steps: break

    accelerator.end_training()

if __name__ == "__main__":
    main()