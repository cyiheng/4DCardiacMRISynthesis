"""Stage 3 training script for conditional latent diffusion of first-frame content.

This script trains a diffusion model that generates latent image features
from clinical and structural conditions. It includes dataset preparation,
sample generation, and loss tracking for model convergence.
"""

import os
import re
import random
import shutil
import sys
from pathlib import Path
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import nibabel as nib
import bitsandbytes as bnb

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from monai.networks.nets import DiffusionModelUNet
from accelerate import Accelerator
from diffusers import DDIMScheduler
from diffusers.optimization import get_scheduler

from models.custom_AEKL import DualDecoderVAE
from models.conditioning import ClinicalConditioningModel
from datasets.latent_datasets import LatentDataset
from utils.common import normalize_scalars
from utils.visualization import plot_and_save_learning_curve

# =========================================================================================
# --- 1. CONFIGURATION ---
# =========================================================================================
class Config:
    # --- Paths ---
    dataset_dir = "data/latents_aug/"     
    train_csv = os.path.join(dataset_dir, "train.csv")
    test_csv = os.path.join(dataset_dir, "test.csv")
    original_image_data_dir = 'data/images'

    vae_model_path = "results/001_VAE3D_dualdecoder/checkpoints/best_autoencoder_model.pth"
    output_dir = "results/003_FirstFrame3D/" 
    os.makedirs(output_dir, exist_ok=True)
    script_path = os.path.abspath(__file__)
    script_name = os.path.basename(script_path)
    shutil.copyfile(script_path, os.path.join(output_dir, script_name))
    resume_from_checkpoint = "latest" 

    # --- Model Parameters ---
    spatial_size = (192, 192, 16)
    latent_channels = 8
    model_in_channels = 9  # 8 Latent + 1 Mask
    num_classes = 4
    cross_attention_dim = 512 
    vae_scale = 0.645331

    # --- Training Params ---
    num_train_timesteps = 1000
    beta_schedule = "scaled_linear"
    max_train_steps = 200000 
    batch_size = 10
    learning_rate = 2e-5
    lr_scheduler_type = "constant_with_warmup"
    lr_warmup_steps = 500
    cfg_dropout_prob = 0.1 

    validation_steps = 10000
    checkpointing_steps = 10000
    plot_smoothing_window = 100

cfg = Config()

DIAGNOSIS_MAP = {
    "Unknown": 0, "NOR": 1, "MINF": 2, "DCM": 3, "HCM": 4, "RV": 5
}
NUM_DIAGNOSIS_CLASSES = len(DIAGNOSIS_MAP) + 1 

# =========================================================================================
# --- 2. DATA PREPARATION ---
# =========================================================================================
def prepare_data(cfg):
    """Load latent dataset metadata and convert it into training/validation lists."""
    print(f"--- Loading Dataset from Augmented Latent CSVs in {cfg.dataset_dir} ---")
    
    # 1. Load DataFrames
    df_train = pd.read_csv(cfg.train_csv)
    df_test = pd.read_csv(cfg.test_csv)
    
    # 2. Helper to process rows
    def process_df(df, subfolder):
        data_list = []
        missing_count = 0
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Loading {subfolder}"):
            filename = row['filename']
            full_path = os.path.join(cfg.dataset_dir, subfolder, filename)
            
            if not os.path.exists(full_path):
                missing_count += 1
                continue
            
            diag_str = row.get('diagnosis', 'Unknown')
            if pd.isna(diag_str): diag_str = 'Unknown'
            diag_id = DIAGNOSIS_MAP.get(diag_str, 0)
            
            # The CSV already contains the augmented EDV/ESV values
            scalars = normalize_scalars(
                float(row['ef']), 
                float(row['edv']), 
                float(row['esv']), 
                float(row['n_slices'])
            )
            
            data_list.append({
                "latent_path": full_path,
                "diag_id": diag_id,
                "scalars": scalars
            })
            
        if missing_count > 0:
            print(f"WARNING: {missing_count} files listed in CSV were not found on disk.")
        return data_list

    # 3. Process
    train_list = process_df(df_train, "train")
    val_list = process_df(df_test, "test")

    print(f"Final Dataset Size -> Train: {len(train_list)} | Val: {len(val_list)}")
    return train_list, val_list


# =========================================================================================
# --- 3. VALIDATION & CROPPING ---
# =========================================================================================
def generate_sample(accelerator, unet, vae, cond_model, noise_scheduler, global_step, ref_affine, latent_shape):
    """Generate validation samples from the diffusion model and save outputs."""
    print(f"\n--- Step {global_step}: Generating & Cropping Samples ---")
    unet.eval(); cond_model.eval()

    test_conditions = [
        {"name": "Healthy_10slice", "diag": 1, "raw": [70.0, 79.0, 23.0, 10.0]}, 
        {"name": "DCM_14slice",     "diag": 3, "raw": [29.0, 223.0, 156.0, 14.0]}
    ]
    
    save_dir = os.path.join(cfg.output_dir, "validation_samples")
    os.makedirs(save_dir, exist_ok=True)
    guidance_scale = 5.0 
    _, h, w, d = latent_shape 

    for case in test_conditions:
        raw = case["raw"]
        sc_norm = normalize_scalars(raw[0], raw[1], raw[2], raw[3])
        requested_slices = int(raw[3])
        
        mask = torch.zeros((1, 1, h, w, d), device=accelerator.device)
        valid_z = min(requested_slices, d)
        diff = d - valid_z
        start_idx = diff // 2
        end_idx = start_idx + valid_z
        mask[..., start_idx:end_idx] = 1.0

        diag_idx = torch.tensor([case["diag"]], device=accelerator.device)
        scalars = torch.tensor([sc_norm], dtype=torch.float32, device=accelerator.device)
        cond_emb = cond_model(diag_idx, scalars)
        
        uncond_diag = torch.tensor([6], device=accelerator.device)
        uncond_sc = torch.zeros_like(scalars)
        uncond_emb = cond_model(uncond_diag, uncond_sc)
        
        context = torch.cat([uncond_emb, cond_emb]) 
        
        latents = torch.randn((1, *latent_shape), device=accelerator.device)
        noise_scheduler.set_timesteps(50)

        with torch.no_grad():
            for t in tqdm(noise_scheduler.timesteps, desc=f"Gen {case['name']}", leave=False):
                latent_input = torch.cat([latents] * 2) 
                mask_input = torch.cat([mask] * 2)
                model_input = torch.cat([latent_input, mask_input], dim=1)
                t_batch = t.expand(latent_input.shape[0]).to(accelerator.device)
                
                noise_pred = unet(x=model_input, context=context, timesteps=t_batch)
                u, c = noise_pred.chunk(2)
                noise_pred = u + guidance_scale * (c - u)
                latents = noise_scheduler.step(noise_pred, t, latents).prev_sample

            img = vae.decoder_img(latents / cfg.vae_scale)
            seg_logits = vae.decoder_seg(latents / cfg.vae_scale)
            seg = torch.argmax(seg_logits, dim=1, keepdim=True).float()
            
            img_np = img.squeeze().cpu().numpy()
            seg_np = seg.squeeze().cpu().numpy()
            
            full_z = img_np.shape[-1]
            if requested_slices < full_z:
                diff = full_z - requested_slices
                start = diff // 2
                end = start + requested_slices
                img_np = img_np[..., start:end]
                seg_np = seg_np[..., start:end]
            
            n_img = nib.Nifti1Image(img_np, ref_affine)
            n_seg = nib.Nifti1Image(seg_np, ref_affine)
            nib.save(n_img, os.path.join(save_dir, f"step_{global_step}_{case['name']}_img.nii.gz"))
            nib.save(n_seg, os.path.join(save_dir, f"step_{global_step}_{case['name']}_seg.nii.gz"))

    unet.train(); cond_model.train()

# =========================================================================================
# --- 4. MAIN ---
# =========================================================================================
def main():
    """Set up model, data, and training loop for stage 3 diffusion training."""
    accelerator = Accelerator(mixed_precision="fp16")
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(os.path.join(cfg.output_dir,"checkpoints"), exist_ok=True)

    vae = DualDecoderVAE(latent_dim=cfg.latent_channels, num_classes=cfg.num_classes)
    vae.load_state_dict(torch.load(cfg.vae_model_path, map_location="cpu"))
    vae.to(accelerator.device)
    vae.eval().requires_grad_(False)

    cond_model = ClinicalConditioningModel(embedding_dim=cfg.cross_attention_dim, num_diagnosis=NUM_DIAGNOSIS_CLASSES)
    
    unet = DiffusionModelUNet(
        spatial_dims=3, in_channels=cfg.model_in_channels, out_channels=cfg.latent_channels,
        channels=(256, 512, 1024),
        attention_levels=(False, True, True),
        num_res_blocks=2, num_head_channels=32, with_conditioning=True,
        cross_attention_dim=cfg.cross_attention_dim 
    )

    noise_scheduler = DDIMScheduler(num_train_timesteps=cfg.num_train_timesteps, beta_schedule=cfg.beta_schedule, clip_sample=False)

    train_lst, val_lst = prepare_data(cfg)
    train_ds = LatentDataset(train_lst)
    val_ds = LatentDataset(val_lst)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=16, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=16)
    
    ref_affine = np.eye(4)
    ref_affine[0, 0] = 1.0   
    ref_affine[1, 1] = 1.0   
    ref_affine[2, 2] = 10.0  

    optimizer = bnb.optim.AdamW8bit(
        list(unet.parameters()) + list(cond_model.parameters()), 
        lr=cfg.learning_rate
    )
    lr_scheduler = get_scheduler(cfg.lr_scheduler_type, optimizer=optimizer, num_warmup_steps=cfg.lr_warmup_steps)

    unet, cond_model, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        unet, cond_model, optimizer, train_loader, lr_scheduler
    )

    detected_shape = None
    for batch in train_loader:
        detected_shape = batch["latent"].shape[1:] 
        print(f"Detected Latent Shape from DataLoader: {detected_shape}")
        break
    if detected_shape is None: raise ValueError("DataLoader is empty!")

    global_step = 0 
    if cfg.resume_from_checkpoint == "latest":
        checkpoint_dirs = []
        if os.path.exists(cfg.output_dir):
            for d in os.listdir(cfg.output_dir):
                # Matches either "ckpt_XXXX" or "checkpoint_step_XXXX" formats
                match = re.match(r"^(ckpt|checkpoint_step)_(\d+)$", d)
                if match:
                    step = int(match.group(2))
                    checkpoint_dirs.append((step, d))
        
        if checkpoint_dirs:
            # Sort numerically by the step integer (first item in the tuple)
            checkpoint_dirs.sort(key=lambda x: x[0])
            latest_step, latest_dir_name = checkpoint_dirs[-1]
            
            latest_checkpoint_path = os.path.join(cfg.output_dir, latest_dir_name)
            print(f"\n[INFO] Automatically resuming from latest checkpoint:")
            print(f"       Path: {latest_checkpoint_path}")
            print(f"       Step: {latest_step}\n")
            
            accelerator.load_state(latest_checkpoint_path)
            global_step = latest_step
        else:
            print("\n[INFO] No previous checkpoints found in output directory. Starting training from scratch (Step 0).\n")
            
    progress_bar = tqdm(
        range(cfg.max_train_steps), 
        initial=global_step, 
        total=cfg.max_train_steps, 
        disable=not accelerator.is_local_main_process
    )
    step_loss_list = []
    
    while global_step < cfg.max_train_steps:
        unet.train(); cond_model.train()
        
        for batch in train_loader:
            with accelerator.accumulate(unet):
                latents = batch["latent"] * cfg.vae_scale
                latents = latents + torch.randn_like(latents) * 0.05
                mask = batch["mask"] 
                diag_ids = batch["diag_id"]
                scalars = batch["scalars"] 

                if unet.training:
                    jitter = 1.0 + (torch.rand_like(scalars) * 0.1 - 0.05) 
                    scalars = scalars * jitter
                    noise_aug = torch.randn_like(latents) * random.uniform(0.01, 0.05)
                    latents = latents + noise_aug

                if random.random() < cfg.cfg_dropout_prob:
                    diag_ids = torch.full_like(diag_ids, 6)
                    scalars = torch.zeros_like(scalars)
                
                context = cond_model(diag_ids, scalars)

                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                model_input = torch.cat([noisy_latents, mask], dim=1)

                noise_pred = unet(x=model_input, context=context, timesteps=timesteps)
                loss = F.mse_loss(noise_pred.float(), noise.float())

                accelerator.backward(loss)
                optimizer.step(); lr_scheduler.step(); optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1); global_step += 1
                avg_loss = accelerator.gather(loss.detach()).mean()
                if accelerator.is_main_process:
                    step_loss_list.append(avg_loss.item())
                progress_bar.set_postfix({"loss": loss.item()})

                if global_step % cfg.validation_steps == 0 or global_step == 1000:
                    if accelerator.is_main_process:
                        plot_and_save_learning_curve(step_loss_list, os.path.join(cfg.output_dir, "loss.png"), 100)
                        
                        generate_sample(accelerator, accelerator.unwrap_model(unet), vae, accelerator.unwrap_model(cond_model), noise_scheduler, global_step, ref_affine, detected_shape)
                        

                if global_step % cfg.checkpointing_steps == 0:
                    accelerator.save_state(os.path.join(cfg.output_dir, f"checkpoints/ckpt_{global_step}"))

            if global_step >= cfg.max_train_steps: break

    print("Training Complete.")
    if accelerator.is_main_process:
        torch.save(accelerator.unwrap_model(unet).state_dict(), os.path.join(cfg.output_dir, "unet_final.pth"))
        torch.save(accelerator.unwrap_model(cond_model).state_dict(), os.path.join(cfg.output_dir, "cond_final.pth"))

if __name__ == "__main__":
    main()