"""Stage 2 training script for latent motion prediction.

This script trains a latent motion model using precomputed VAE latents,
adversarial supervision, and clinical conditioning. It is designed to
learn temporal transitions in the latent space between image pairs.
"""

import os
import sys
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# --- MONAI Imports ---
from monai.transforms import (
    Compose, LoadImaged, ScaleIntensityRangePercentilesd, EnsureTyped,
    Spacingd, SpatialPadd, CenterSpatialCropd,
)
from monai.networks.nets import PatchDiscriminator
from monai.losses import PatchAdversarialLoss, PerceptualLoss, DiceCELoss

# --- Accelerate Imports ---
from accelerate import Accelerator

# --- CUSTOM IMPORTS ---
from models.custom_AEKL import DualDecoderVAE
from models.latent_motion import LatentFeatureMotionPredictor
from models.flow_losses import AntiCheatMonitor
from datasets.latent_datasets import LatentFlowDataset
from datasets.paired_latent import prepare_paired_datalist
from utils.visualization import plot_and_save_learning_curve, visualize_fixed_pair

class Config:
    # --- Paths ---
    data_dir = "data"
    latent_data_dir = os.path.join(data_dir, "latents")
    label_data_dir = os.path.join(data_dir, 'labels')
    original_image_data_dir = os.path.join(data_dir, "images")

    train_csv = os.path.join(data_dir, "train_split_final.csv")
    test_csv = os.path.join(data_dir, "test_split_final.csv")

    vae_model_path = os.path.join("results", "001_VAE3D_dualdecoder", "checkpoints", "best_autoencoder_model.pth")
    output_dir = os.path.join("results", "002_LatentMotion")
    
    # --- Model & Image Parameters ---
    spatial_size = (192, 192, 16)
    target_pixdim = (1.0, 1.0, 10.0) 
    latent_channels = 8 
    num_classes = 4

    # --- Loss Weights ---
    l1_recon_weight = 2.0
    perceptual_weight = 1.0
    gan_weight = 0.01 
    latent_l1_weight = 1.0 
    seg_weight = 10.0 
    self_recon_weight = 0.5   
    
    vae_scaling = 0.643433 

    # --- Noise Injection ---
    flow_noise_std = 0.01 

    # --- Training Params ---
    max_train_steps = 100000
    phase1_flow_steps = 10000
    batch_size = 2
    
    learning_rate_g = 2e-4    
    learning_rate_d = 5e-5    
    learning_rate_dec = 1e-5  
    
    adam_beta1 = 0.9
    adam_beta2 = 0.999
    adam_weight_decay = 1e-2
    
    # --- Validation & Checkpointing ---
    validation_steps = 5000
    checkpointing_steps = 10000

cfg = Config()
os.makedirs(cfg.output_dir, exist_ok=True)


def main():
    """Initialize training components and run stage 2 optimization."""
    accelerator = Accelerator(mixed_precision="fp16")
    os.makedirs(os.path.join(cfg.output_dir, "validation_samples"), exist_ok=True)
    os.makedirs(os.path.join(cfg.output_dir, "checkpoints"), exist_ok=True)

    # --- 1. Load DualDecoderVAE ---
    vae = DualDecoderVAE(latent_dim=cfg.latent_channels, num_classes=cfg.num_classes)
    vae.load_state_dict(torch.load(cfg.vae_model_path, map_location="cpu"))
    vae.eval()

    # --- 2. Initialize Models ---
    flow_predictor = LatentFeatureMotionPredictor(in_channels=2 * cfg.latent_channels, latent_channels=cfg.latent_channels)
    anti_cheat = AntiCheatMonitor()
    discriminator = PatchDiscriminator(
        spatial_dims=3, num_layers_d=2, channels=32, in_channels=1, out_channels=1, norm="INSTANCE",
    )

    # --- 3. Transforms (MATCHING VAE TRAINING) ---
    image_transforms = Compose([
        LoadImaged(keys=["dri_image", "ref_image", "dri_label", "ref_label"], image_only=False, ensure_channel_first=True, allow_missing_keys=True),
        Spacingd(keys=["dri_image", "ref_image", "dri_label", "ref_label"], pixdim=(1.0, 1.0, -1.0), mode=("bilinear", "bilinear", "nearest", "nearest"), allow_missing_keys=True),
        SpatialPadd(keys=["dri_image", "ref_image", "dri_label", "ref_label"], spatial_size=cfg.spatial_size, method="symmetric", mode="edge", allow_missing_keys=True),
        CenterSpatialCropd(keys=["dri_image", "ref_image", "dri_label", "ref_label"], roi_size=cfg.spatial_size, allow_missing_keys=True),
        ScaleIntensityRangePercentilesd(keys=["dri_image", "ref_image"], lower=0.5, upper=99.5, b_min=-1.0, b_max=1.0, clip=True, allow_missing_keys=True),
        EnsureTyped(keys=["dri_image", "ref_image", "dri_label", "ref_label"], dtype=torch.float32, allow_missing_keys=True),
    ])

    train_data, val_data = prepare_paired_datalist(cfg)

    # --- DYNAMIC VALIDATION SAMPLE SELECTION ---
    if len(val_data) > 0:
        # Prioritize picking a validation pair that HAS both source & target labels
        labeled_val_pairs = [
            p for p in val_data 
            if p["ref_label"] is not None and p["dri_label"] is not None
        ]
        
        if labeled_val_pairs:
            fixed_pair = labeled_val_pairs[0]  # Pick the first fully-labeled pair
        else:
            fixed_pair = val_data[0]           # Fallback to the first overall pair on disk
            
        fixed_ref_latent_path = fixed_pair["ref_latent"]
        fixed_dri_latent_path = fixed_pair["dri_latent"]
        fixed_ref_image_path = fixed_pair["ref_image"]
        fixed_dri_image_path = fixed_pair["dri_image"]
        fixed_ref_label_path = fixed_pair["ref_label"] if fixed_pair["ref_label"] else ""
        fixed_dri_label_path = fixed_pair["dri_label"] if fixed_pair["dri_label"] else ""
        
        print(f"\n[INFO] Selected dynamic validation pair for visual logging:")
        print(f"       Ref: {os.path.basename(fixed_ref_image_path)}")
        print(f"       Dri: {os.path.basename(fixed_dri_image_path)}\n")
    else:
        raise RuntimeError("Validation dataset `val_data` is empty. Ensure test paths and CSV splits are correct.")
    
    train_loader = DataLoader(LatentFlowDataset(train_data, image_transforms, cfg.vae_scaling), batch_size=cfg.batch_size, shuffle=True, num_workers=4)
    
    # --- 4. Optimizers ---
    optimizer_flow = torch.optim.AdamW(flow_predictor.parameters(), lr=cfg.learning_rate_g, betas=(cfg.adam_beta1, cfg.adam_beta2))
    optimizer_d = torch.optim.AdamW(discriminator.parameters(), lr=cfg.learning_rate_d, betas=(cfg.adam_beta1, cfg.adam_beta2))

    # --- 5. Prepare ---
    flow_predictor, discriminator, vae, optimizer_flow, optimizer_d, train_loader = accelerator.prepare(
        flow_predictor, discriminator, vae, optimizer_flow, optimizer_d, train_loader
    )
    
    l1_loss = torch.nn.L1Loss()
    perceptual_loss = PerceptualLoss(spatial_dims=3, network_type="vgg", fake_3d_ratio=0.25).to(accelerator.device)
    adv_loss = PatchAdversarialLoss(criterion="least_squares")
    dice_ce_loss = DiceCELoss(softmax=True, to_onehot_y=True, squared_pred=True, lambda_dice=1.0, lambda_ce=1.0)

    # --- 6. Training Loop ---
    global_step = 0
    g_loss_history, d_loss_history = [], []
    progress_bar = tqdm(range(cfg.max_train_steps), disable=not accelerator.is_local_main_process)
    train_iter = iter(train_loader)
    
    while global_step < cfg.max_train_steps:
        try: batch = next(train_iter)
        except StopIteration: train_iter = iter(train_loader); batch = next(train_iter)

        ref_latents = batch["ref_latent"]
        dri_latents = batch["dri_latent"]
        dri_images = batch["dri_image"]
        dri_labels = batch["dri_label"]
        has_dri_label = batch["has_dri_label"]

        # --- GENERATOR STEP ---
        flow_predictor.train()
        discriminator.train()

        with accelerator.accumulate(flow_predictor):
            optimizer_flow.zero_grad()
            noise_decay = max(0.0, 1.0 - (global_step / (cfg.max_train_steps * 0.8)))
            current_noise_std = cfg.flow_noise_std * noise_decay

            feature_motion = flow_predictor(ref_latents, dri_latents)
            _,_ = anti_cheat.check(ref_latents, dri_latents, feature_motion)
            
            if current_noise_std > 0:
                noise = torch.randn_like(feature_motion) * current_noise_std
                z_warped = ref_latents + feature_motion + noise
            else:
                z_warped = ref_latents + feature_motion
            
            warped_img = vae.decoder_img(z_warped / cfg.vae_scaling)
            warped_logits = vae.decoder_seg(z_warped / cfg.vae_scaling)
            
            loss_warp_l1 = l1_loss(warped_img, dri_images)
            loss_warp_p = perceptual_loss(warped_img.float(), dri_images.float())
            loss_sparsity = torch.mean(feature_motion ** 2)
            
            loss_seg = 0.0
            mask_warp = has_dri_label > 0.5
            if mask_warp.any():
                loss_seg = dice_ce_loss(warped_logits[mask_warp], dri_labels[mask_warp])
            
            loss_g_adv = 0.0
            if global_step > cfg.phase1_flow_steps:
                logits_fake = discriminator(warped_img)
                loss_g_adv = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)

            zero_motion_pred = flow_predictor(ref_latents, ref_latents)
            loss_identity = F.l1_loss(zero_motion_pred, torch.zeros_like(zero_motion_pred))

            total_loss = (
                cfg.l1_recon_weight * loss_warp_l1 +
                cfg.perceptual_weight * loss_warp_p +
                0.01 * loss_sparsity +
                1.0 * loss_identity +
                cfg.gan_weight * loss_g_adv +
                cfg.seg_weight * loss_seg 
            )

            accelerator.backward(total_loss)
            optimizer_flow.step()

        # --- DISCRIMINATOR STEP ---
        loss_d = torch.tensor(0.0).to(accelerator.device)
        if global_step > cfg.phase1_flow_steps:
            with accelerator.accumulate(discriminator):
                optimizer_d.zero_grad()
                
                logits_real = discriminator(dri_images)
                loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                
                logits_fake = discriminator(warped_img.detach())
                loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                
                loss_d = 0.5 * (loss_d_real + loss_d_fake)
                
                accelerator.backward(loss_d)
                optimizer_d.step()

        # --- Logging & Validation ---
        g_loss_history.append(total_loss.item())
        d_loss_history.append(loss_d.item())
        if global_step % 100 == 0:
             monitor_stats = anti_cheat.log_and_reset()
             
        progress_bar.set_postfix({"L_Tot": f"{total_loss.item():.3f}", "L_Seg": f"{loss_seg:.3f}", 
                                  "Sim": f"{monitor_stats.get('sim_RB', 0):.2f}"})
        progress_bar.update(1)
        global_step += 1

        if global_step % cfg.validation_steps == 0 or global_step == 1000:
            if accelerator.is_main_process:
                plot_and_save_learning_curve(g_loss_history, d_loss_history, os.path.join(cfg.output_dir, "learning_curve.png"))
                visualize_fixed_pair(
                    accelerator.unwrap_model(flow_predictor),
                    accelerator.unwrap_model(vae),
                    image_transforms, accelerator.device,
                    fixed_ref_latent_path, fixed_dri_latent_path,
                    fixed_ref_image_path, fixed_dri_image_path,
                    fixed_ref_label_path, fixed_dri_label_path,
                    global_step, os.path.join(cfg.output_dir, "validation_samples"), cfg.vae_scaling
                )
                
        if global_step % cfg.checkpointing_steps == 0 and accelerator.is_main_process:
            accelerator.save_state(os.path.join(cfg.output_dir, "checkpoints", f"step_{global_step}"))
            torch.save(accelerator.unwrap_model(flow_predictor).state_dict(), os.path.join(cfg.output_dir, "checkpoints", f"flow_{global_step}.pth"))

    print("Training Complete.")

if __name__ == "__main__":
    try:
        shutil.copy(sys.argv[0], os.path.join(cfg.output_dir, os.path.basename(sys.argv[0])))
        print(f"Copied training script to '{cfg.output_dir}'")
    except Exception as e:
        pass
    main()