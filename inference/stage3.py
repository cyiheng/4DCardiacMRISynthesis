import os
import torch
import numpy as np
import nibabel as nib
from tqdm import tqdm
from diffusers import DDIMScheduler
from accelerate import Accelerator
from torch.utils.data import DataLoader

# --- MONAI ---
from monai.networks.nets import DiffusionModelUNet

# --- CUSTOM VAE ---
from models.custom_AEKL import DualDecoderVAE

# =========================================================================================
# --- 1. CONFIGURATION & MODEL DEFINITIONS ---
# =========================================================================================

class Config:
    # --- Paths ---
    output_dir = "results/003_FirstFrame3D/generated_samples/"
    
    # Path to the trained weights
    vae_ckpt = "results/001_VAE3D_dualdecoder/checkpoints/best_autoencoder_model.pth"
    unet_ckpt = "results/003_FirstFrame3D/unet_final.pth"
    cond_ckpt = "results/003_FirstFrame3D/cond_final.pth"
    # checkpoint_path = "results/003_FirstFrame3D/checkpoints/ckpt_20" 


    # --- Model Params ---
    latent_channels = 8
    model_in_channels = 9  # <--- 8 Latent + 1 Mask
    use_mask = True
    num_classes = 4
    cross_attention_dim = 512
    vae_scale = 0.643433  # From training
    
    # Latent Dimensions (192 / 16 = 12)
    latent_h = 24
    latent_w = 24
    latent_d = 16
    num_diagnosis = 7 # 6 classes + 1 null

    # --- Normalization Limits (Must match Training!) ---
    MAX_EDV = 600.0
    MAX_ESV = 600.0
    MAX_SLICES = 20.0

    device = "cuda" if torch.cuda.is_available() else "cpu"

cfg = Config()

DIAGNOSIS_MAP = {
    "Unknown": 0, "NOR": 1, "MINF": 2, "DCM": 3, "HCM": 4, "RV": 5
}

# --- Conditioning Model Classes (Must Match Training) ---
ROOT_DIR = None
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
import sys
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Conditioning classes and positional embedding imported from shared module
from models.conditioning import ClinicalConditioningModel
from utils.common import normalize_scalars

# =========================================================================================
# --- 2. SETUP FUNCTIONS ---
# =========================================================================================
def extract_ckpt():
    print(f"--- Extracting weights from: {cfg.checkpoint_path} ---")
    
    # 1. Initialize Accelerator
    # We don't need mixed precision for extraction
    accelerator = Accelerator(mixed_precision="no") 
    
    # 2. Re-Initialize Models
    # UNet
    unet = DiffusionModelUNet(
        spatial_dims=3, 
        in_channels=cfg.model_in_channels if cfg.use_mask else cfg.model_in_channels - 1,
        out_channels=cfg.latent_channels,  # 8
        channels=(256, 512, 1024),
        attention_levels=(False, True, True),
        num_res_blocks=2, num_head_channels=32, with_conditioning=True,
        cross_attention_dim=cfg.cross_attention_dim 
    )
    
    # Cond Model
    cond_model = ClinicalConditioningModel(
        embedding_dim=cfg.cross_attention_dim, 
        num_diagnosis=cfg.num_diagnosis
    )
        
    # We need dummy objects to satisfy the loading mapping
    optimizer = torch.optim.AdamW(list(unet.parameters()) + list(cond_model.parameters()))
    
    # Dummy Scheduler
    from diffusers.optimization import get_scheduler
    lr_scheduler = get_scheduler("constant_with_warmup", optimizer=optimizer, num_warmup_steps=500)
    
    # Dummy Loader (Just an empty list is often enough, or a minimal dataloader)
    # Accelerate creates a dummy dataloader wrapper.
    dummy_loader = DataLoader([0], batch_size=1)

    # 4. Prepare
    unet, cond_model, optimizer, dummy_loader, lr_scheduler = accelerator.prepare(
        unet, cond_model, optimizer, dummy_loader, lr_scheduler
    )

    # 5. Load State
    try:
        accelerator.load_state(cfg.checkpoint_path)
        print(" -> Checkpoint state loaded successfully into Accelerator.")
    except Exception as e:
        print(f"ERROR Loading Checkpoint: {e}")
        print("Ensure 'checkpoint_path' points to the folder containing pytorch_model.bin or similar.")
        return

    # 6. Unwrap and Save
    print(" -> Saving unwrapped models...")
    
    # Unwrap
    unet_unwrapped = accelerator.unwrap_model(unet)
    cond_unwrapped = accelerator.unwrap_model(cond_model)
    return unet_unwrapped, cond_unwrapped

def load_models():
    print("--- Loading Models ---")
    
    # 1. VAE
    vae = DualDecoderVAE(latent_dim=cfg.latent_channels, num_classes=cfg.num_classes)
    vae.load_state_dict(torch.load(cfg.vae_ckpt, map_location="cpu"))
    vae.to(cfg.device)
    vae.eval()
    
    if cfg.checkpoint_path:
        unet, cond_model = extract_ckpt()
    else:
        # 2. Conditioning
        cond_model = ClinicalConditioningModel(embedding_dim=cfg.cross_attention_dim, num_diagnosis=len(DIAGNOSIS_MAP)+1)
        cond_model.load_state_dict(torch.load(cfg.cond_ckpt, map_location="cpu"))
        
        # 3. UNet
        unet = DiffusionModelUNet(
            spatial_dims=3, 
            in_channels=cfg.model_in_channels if cfg.use_mask else cfg.model_in_channels - 1,
            out_channels=cfg.latent_channels,
            channels=(256, 512, 1024),
            attention_levels=(False, True, True),
            num_res_blocks=2, num_head_channels=32, with_conditioning=True,
            cross_attention_dim=cfg.cross_attention_dim
        )
        unet.load_state_dict(torch.load(cfg.unet_ckpt, map_location="cpu"))
    
    cond_model.to(cfg.device)
    cond_model.eval()
    unet.to(cfg.device)
    unet.eval()
    # 4. Scheduler
    scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule="scaled_linear", clip_sample=False)
    
    print("--- Models Loaded Successfully ---")
    return vae, cond_model, unet, scheduler

# =========================================================================================
# --- 3. METRIC CALCULATION UTILS ---
# =========================================================================================

def check_metrics(seg_np, target_edv, n_slices):
    """
    Calculates the volume of the LV from the generated segmentation.
    Assumptions: 
      - Label 3 = Left Ventricle (Standard ACDC)
      - Label 1 = Right Ventricle
      - Voxel Size = 1.0 x 1.0 x 10.0 mm (from script config)
    """
    # 1. Define Voxel Volume in ml
    # 1 voxel = 1mm * 1mm * 10mm = 10mm^3 = 0.01 ml
    voxel_vol_ml = 1.0 * 1.0 * 10.0 / 1000.0 
    
    # 2. Count Voxels (Standard ACDC: 0=Bg, 1=RV, 2=Myo, 3=LV)
    lv_voxels = np.sum(seg_np == 3)
    rv_voxels = np.sum(seg_np == 1)
    
    # 3. Calculate Volumes
    measured_lv_vol = lv_voxels * voxel_vol_ml
    measured_rv_vol = rv_voxels * voxel_vol_ml
    
    # 4. Error Calculation
    error_edv = measured_lv_vol - target_edv
    perc_error = (error_edv / target_edv) * 100.0 if target_edv > 0 else 0.0
    
    print("\n--- Clinical Verification ---")
    print(f"Target EDV:    {target_edv:.1f} ml")
    print(f"Measured LV:   {measured_lv_vol:.1f} ml")
    print(f"Measured RV:   {measured_rv_vol:.1f} ml")
    print(f"Difference:    {error_edv:.1f} ml ({perc_error:.1f}%)")
    
    # Note on EF
    print(f"Note: EF cannot be calculated from a single frame. This generation represents the End-Diastole (ED).")
    print("-----------------------------\n")

# =========================================================================================
# --- 4. INFERENCE LOOP ---
# =========================================================================================

def generate_heart(
    models, 
    diagnosis: str, 
    ef: float, 
    edv: float, 
    esv: float, 
    n_slices: int, 
    filename: str,
    guidance_scale=5.0,
    num_inference_steps=50,
    seed=None
):
    vae, cond_model, unet, scheduler = models
    
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # 1. Prepare Conditioning (Same)
    diag_id = DIAGNOSIS_MAP.get(diagnosis, 0)
    scalars_norm = normalize_scalars(
        ef, edv, esv, n_slices, 
        max_edv=cfg.MAX_EDV, 
        max_esv=cfg.MAX_ESV, 
        max_slices=cfg.MAX_SLICES
    )
    
    print(f"Generating: {diagnosis} | EF: {ef}% | EDV: {edv}ml | ESV: {esv}ml | Slices: {n_slices}")
    
    diag_t = torch.tensor([diag_id], device=cfg.device)
    scal_t = torch.tensor([scalars_norm], dtype=torch.float32, device=cfg.device)
    
    with torch.no_grad():
        cond_emb = cond_model(diag_t, scal_t)
        uncond_diag = torch.tensor([6], device=cfg.device)
        uncond_scal = torch.zeros_like(scal_t)
        uncond_emb = cond_model(uncond_diag, uncond_scal)
        context = torch.cat([uncond_emb, cond_emb])

    # --- GENERATE CENTERED MASK ---
    # Create mask shape (1, 1, H, W, D)
    mask = torch.zeros((1, 1, cfg.latent_h, cfg.latent_w, cfg.latent_d), device=cfg.device)
    
    # Calculate Center Indices
    # Note: Training logic ensures slices are centered in the volume
    valid_z = min(n_slices, cfg.latent_d)
    diff = cfg.latent_d - valid_z
    start_idx = diff // 2
    end_idx = start_idx + valid_z
    
    mask[..., start_idx:end_idx] = 1.0

    # 2. Initialize Latents
    latents = torch.randn(
        (1, cfg.latent_channels, cfg.latent_h, cfg.latent_w, cfg.latent_d), 
        device=cfg.device
    )
    
    scheduler.set_timesteps(num_inference_steps)
    
    # 3. Diffusion Process
    for t in tqdm(scheduler.timesteps, desc="Denoising"):
        # Double latent for CFG (Uncond, Cond)
        latent_input = torch.cat([latents] * 2)
        
        if cfg.use_mask:
            # Double mask for CFG
            mask_input = torch.cat([mask] * 2)
            
            # Concatenate: (B, 8, ...) + (B, 1, ...) = (B, 9, ...)
            model_input = torch.cat([latent_input, mask_input], dim=1)
        else :
            model_input = latent_input
        
        t_batch = t.expand(latent_input.shape[0]).to(cfg.device)
        
        with torch.no_grad():
            # Feed model_input (9 channels)
            noise_pred = unet(x=model_input, context=context, timesteps=t_batch)
            
        u, c = noise_pred.chunk(2)
        noise_pred = u + guidance_scale * (c - u)
        
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    # 4. VAE Decoding
    print("Decoding...")
    with torch.no_grad():
        latents_scaled = latents / cfg.vae_scale
        img_recon = vae.decoder_img(latents_scaled)
        seg_logits = vae.decoder_seg(latents_scaled)
        seg_recon = torch.argmax(seg_logits, dim=1, keepdim=True).float()

    # 5. Post-Processing (Cropping)
    # This logic matches the centered mask generation perfectly
    img_np = img_recon.squeeze().cpu().numpy()
    seg_np = seg_recon.squeeze().cpu().numpy()
    
    full_z = img_np.shape[-1]
    
    if n_slices < full_z:
        diff = full_z - n_slices
        start = diff // 2
        end = start + n_slices
        img_np = img_np[..., start:end]
        seg_np = seg_np[..., start:end]
        print(f" -> Cropped from {full_z} to {n_slices} slices.")

    # 7. Verification
    # We compare the generated volume against the input EDV
    check_metrics(seg_np, target_edv=edv, n_slices=n_slices)

    # 8. Save
    os.makedirs(cfg.output_dir, exist_ok=True)
    affine = np.eye(4); affine[0,0]=1.0; affine[1,1]=1.0; affine[2,2]=10.0
    
    nifti_img = nib.Nifti1Image(img_np, affine)
    nifti_seg = nib.Nifti1Image(seg_np, affine)
    
    img_path = os.path.join(cfg.output_dir, f"seed{seed}_{filename}_img.nii.gz")
    seg_path = os.path.join(cfg.output_dir, f"seed{seed}_{filename}_seg.nii.gz")
    
    nib.save(nifti_img, img_path)
    nib.save(nifti_seg, seg_path)
    print(f"Saved to {img_path}")

# =========================================================================================
# --- 4. RUN ---
# =========================================================================================

if __name__ == "__main__":
    # Load all models once
    models = load_models()
    
    # --- Define Cases to Generate ---
    cases = [
        # Case 1: Perfectly Healthy
        {
            "diagnosis": "NOR",
            "ef": 70.0,
            "edv": 79.0,
            "esv": 23.0, # (140 * 0.4)
            "n_slices": 8,
            "filename": "sample_01_healthy"
        },
        # Case 2: Severe Dilated Cardiomyopathy (Big heart, thin walls, low EF)
        {
            "diagnosis": "DCM",
            "ef": 29.0,
            "edv": 223.0,
            "esv": 156.0,
            "n_slices": 10,
            "filename": "sample_02_dcm_severe"
        },
        # Case 3: Hypertrophic Cardiomyopathy (Thick walls)
        {
            "diagnosis": "HCM",
            "ef": 84.0,
            "edv": 141.0,
            "esv": 22.0,
            "n_slices": 9,
            "filename": "sample_03_hcm"
        },
        # Case 4: Right Ventricle Abnormality
        {
            "diagnosis": "RV",
            "ef": 52.0,
            "edv": 158.0,
            "esv": 75.0,
            "n_slices": 8,
            "filename": "sample_04_rv"
        }
    ]
    
    # Run Generation
    for c in cases:
        generate_heart(
            models, 
            diagnosis=c["diagnosis"],
            ef=c["ef"],
            edv=c["edv"],
            esv=c["esv"],
            n_slices=c["n_slices"],
            filename=c["filename"],
            seed=52 
        )