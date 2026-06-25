"""Stage 1 training script for dual-decoder VAE pretraining.

This script trains a 3D VAE with a segmentation decoder and image
reconstruction decoder. It uses joint unlabeled and labeled data, a
reconstruction loss, segmentation loss, perceptual loss, adversarial
loss, and KL regularization with annealing.
"""

import os
import shutil
import torch
from monai.networks.nets import PatchDiscriminator
from monai.losses import PatchAdversarialLoss, PerceptualLoss, DiceCELoss, DiceLoss
from tqdm import tqdm
import numpy as np
import nibabel as nib
import random
import gc

from datasets.dataset_utils import build_dataloaders
from models.custom_AEKL import DualDecoderVAE
from utils.visualization import plot_and_save_learning_curve

# --- 1. SET UP CONFIGURATION ---
print("--- 1. Setting up configuration ---")
data_dir = 'data'  # All inputs and CSVs are subfolders under `data/`
img_dir = os.path.join(data_dir, 'images')
label_dir = os.path.join(data_dir, 'labels/')
LATENT_DIM=8
NUM_CLASSES=4 # RV LV MYO + background

max_steps = 100000 
val_interval = 10000
batch_size = 1
learning_rate = 1e-4
perceptual_weight = 0.3
adv_weight = 0.1
seg_weight = 20.0 

# KL Annealing
kl_annealing_start_step = 20000
kl_annealing_duration = 5000
max_kl_weight = 1e-7

# Dimensions
patch_size = (192, 192, 16) 
spatial_dims_divisible = (8, 8, 1) 

# Output
output_dir = os.path.join('results', '001_VAE3D_dualdecoder')
sample_output_dir = os.path.join(output_dir, "validation_samples")
os.makedirs(sample_output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)

script_path = os.path.abspath(__file__)
script_name = os.path.basename(script_path)
shutil.copyfile(script_path, os.path.join(output_dir, script_name))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 2. PREPARE DATASET FILES ---
print("\n--- 2. Preparing dataset files ---")
train_csv = os.path.join(data_dir, 'train_split_final.csv')
test_csv = os.path.join(data_dir, 'test_split_final.csv')

# Parse Metadata to get Rules (Whitelist)
print("Parsing metadata and building dataloaders...")
train_loader_all, train_loader_labeled, val_loader = build_dataloaders(
    data_dir=img_dir,
    label_dir=label_dir,
    patch_size=patch_size,
    batch_size=batch_size,
    num_workers=4,
    train_csv=train_csv,
    test_csv=test_csv
)

# --- 3. DATASET SUMMARY ---
random.seed(42)
train_all_count = len(train_loader_all.dataset)
train_labeled_count = len(train_loader_labeled.dataset)
val_count = len(val_loader.dataset)
print(f"Total training frames (All - Unlabeled Pool): {train_all_count}")
print(f"Total labeled training frames (Target Pool): {train_labeled_count}")
print(f"Total validation frames: {val_count}")

if train_labeled_count == 0:
    raise RuntimeError("No labeled training files found! Check paths and metadata logic.")

# --- 4. DEFINE NETWORK, LOSSES, and OPTIMIZERS ---
print("\n--- 4. Defining networks, losses, and optimizers ---")
autoencoder = DualDecoderVAE(
        latent_dim=LATENT_DIM, 
        num_classes=NUM_CLASSES
    ).to(device)

discriminator = PatchDiscriminator(
    spatial_dims=3, num_layers_d=2, channels=32, in_channels=1, out_channels=1, norm="INSTANCE",
).to(device)

perceptual_loss = PerceptualLoss(spatial_dims=3, network_type="vgg", fake_3d_ratio=0.25).to(device)
adv_loss = PatchAdversarialLoss(criterion="least_squares")
dice_ce_loss = DiceCELoss(
    softmax=True, 
    to_onehot_y=True, 
    squared_pred=True,
    lambda_dice=1.0,
    lambda_ce=1.0
)
val_dice_metric = DiceLoss(
    softmax=True, 
    to_onehot_y=True,
    include_background=False 
)
l1_loss = torch.nn.L1Loss() 

optimizer_g = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=5e-4)

# --- 5. TRAINING LOOP ---
print("\n--- 5. Starting Training ---")
train_recon_losses = []

best_val_loss = float('inf')
global_step = 0

def cycle(iterable):
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)
            yield next(iterator)

train_iterator_all = iter(train_loader_all)
train_iterator_labeled = cycle(train_loader_labeled) 

progress_bar = tqdm(range(max_steps), initial=global_step, total=max_steps, ncols=140)

g_loss_history = []
d_loss_history = []
while global_step < max_steps:
    autoencoder.train()
    discriminator.train()
    
    try:
        batch_u = next(train_iterator_all)
    except StopIteration:
        train_iterator_all = iter(train_loader_all)
        batch_u = next(train_iterator_all)

    if batch_u is None: continue 
    batch_l = next(train_iterator_labeled)
    if batch_l is None: continue 
    
    img_u = batch_u["image"].to(device)
    img_l = batch_l["image"].to(device)
    lbl_l = batch_l["label"].to(device) 

    images_combined = torch.cat([img_u, img_l], dim=0)

    if global_step < kl_annealing_start_step:
        current_kl_weight = 0.0
    elif global_step < (kl_annealing_start_step + kl_annealing_duration):
        progress = (global_step - kl_annealing_start_step) / kl_annealing_duration
        current_kl_weight = max_kl_weight * progress
    else:
        current_kl_weight = max_kl_weight
    
    optimizer_g.zero_grad()
    
    recon_img, recon_seg, z_mu, z_sigma = autoencoder(images_combined)

    rec_loss_l1 = l1_loss(recon_img, images_combined)
    p_loss = perceptual_loss(recon_img.float(), images_combined.float())

    b_size = img_u.shape[0] 
    recon_seg_labeled = recon_seg[b_size:]
    seg_loss_val = dice_ce_loss(recon_seg_labeled, lbl_l)

    eps = 1e-10
    kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2) + eps) - 1, dim=list(range(1, len(z_sigma.shape))))
    kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

    logits_fake = discriminator(recon_img)[-1] 
    g_adv_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)

    loss_g = rec_loss_l1 + (current_kl_weight * kl_loss) + (perceptual_weight * p_loss) + \
             (adv_weight * g_adv_loss) + (seg_weight * seg_loss_val)
    
    loss_g.backward()
    optimizer_g.step()

    optimizer_d.zero_grad()
    logits_real = discriminator(images_combined)[-1]
    logits_fake = discriminator(recon_img.detach())[-1] 
    loss_d = (adv_loss(logits_real, True, True) + adv_loss(logits_fake, False, True)) * 0.5
    loss_d.backward()
    optimizer_d.step()
    g_loss_history.append(loss_g.item())
    d_loss_history.append(loss_d.item())
    train_recon_losses.append(rec_loss_l1.item())
    
    progress_bar.set_description(f"Step {global_step}")
    progress_bar.set_postfix({
        "L1": f"{rec_loss_l1.item():.3f}", 
        "Seg": f"{seg_loss_val.item():.3f}", 
        "G": f"{g_adv_loss.item():.2f}", 
        "KL": f"{kl_loss.item():.1e}"
    })
    progress_bar.update(1)

    # --- Validation ---
    if (global_step + 1) % val_interval == 0 or global_step+1 == 1000  or global_step+1 == 5000:
        torch.cuda.empty_cache()
        gc.collect() 
        autoencoder.eval()
        val_recon_agg = 0
        val_seg_agg = 0
        seg_count = 0
        all_latents = []

        with torch.no_grad():
            for val_batch in val_loader:
                if val_batch is None: continue
                
                v_imgs = val_batch["image"].to(device)
                v_lbls = val_batch["label"].to(device)
                v_has = val_batch["has_label"].to(device)
                
                v_rec_img, v_rec_seg, v_z_mu, _ = autoencoder(v_imgs)
                val_recon_agg += l1_loss(v_rec_img, v_imgs).item()

                if torch.sum(v_has) > 0:
                    s_loss = val_dice_metric(v_rec_seg, v_lbls)
                    val_seg_agg += (s_loss * v_has.mean()).item()
                    seg_count += 1

                all_latents.append(v_z_mu.reshape(-1).cpu())
        
        if len(val_loader) > 0: avg_val_loss = val_recon_agg / len(val_loader)
        else: avg_val_loss = 0.0
            
        avg_seg_loss = val_seg_agg / seg_count if seg_count > 0 else 0.0
        
        current_len = len(train_recon_losses)
        start_idx = max(0, current_len - val_interval)

        print(f"\nStep {global_step+1}: Val L1: {avg_val_loss:.4f} | Val Seg Dice: {avg_seg_loss:.4f}")

        sample_data = None
        for item in getattr(val_loader, 'dataset', []):
            if item is not None and item.get('has_label', 0) > 0:
                sample_data = item
                break
        
        if sample_data is not None:
            s_img = sample_data['image'].unsqueeze(0).to(device) 
            s_rec_img, s_rec_seg, _, _ = autoencoder(s_img)
            
            orig_np = s_img[0, 0].cpu().numpy()
            rec_np = s_rec_img[0, 0].cpu().detach().numpy()
            rec_seg_logits = s_rec_seg[0] 
            rec_seg_int = torch.argmax(rec_seg_logits, dim=0).cpu().detach().numpy().astype(np.float32)

            fname = sample_data['image_meta_dict']['filename_or_obj']
            try:
                original_proxy = nib.load(fname)
                disk_shape = original_proxy.shape 
                current_z = rec_np.shape[-1] 
                original_z = disk_shape[-1]  

                if current_z > original_z:
                    diff = current_z - original_z
                    start_idx = diff // 2
                    end_idx = start_idx + original_z
                    
                    orig_np = orig_np[..., start_idx:end_idx]
                    rec_np = rec_np[..., start_idx:end_idx]
                    rec_seg_int = rec_seg_int[..., start_idx:end_idx]
            except Exception as e:
                print(f"Warning: Could not crop to original size ({e}). Saving full padded volume.")

            affine = sample_data['image_meta_dict']['affine']
            nib.save(nib.Nifti1Image(orig_np, affine), os.path.join(sample_output_dir, f"step_{global_step+1}_img_orig.nii.gz"))
            nib.save(nib.Nifti1Image(rec_np, affine), os.path.join(sample_output_dir, f"step_{global_step+1}_img_rec.nii.gz"))
            nib.save(nib.Nifti1Image(rec_seg_int, affine), os.path.join(sample_output_dir, f"step_{global_step+1}_seg_rec.nii.gz"))
        
        torch.save(autoencoder.state_dict(), os.path.join(output_dir, f"checkpoints/autoencoder_epoch_{global_step+1:06d}.pth"))
        
        total_val_metric = avg_val_loss + avg_seg_loss
        if total_val_metric < best_val_loss:
            best_val_loss = total_val_metric
            torch.save(autoencoder.state_dict(), os.path.join(output_dir, "checkpoints/best_autoencoder_model.pth"))
            print(f"  -> New best model (L1+Seg) saved.")

        plot_and_save_learning_curve(
            g_loss_history, 
            d_loss_history, 
            os.path.join(output_dir, "learning_curve_final.png")
        )
        
        try:
            del v_imgs, v_lbls, v_has, v_rec_img, v_rec_seg
            del s_img, s_rec_img, s_rec_seg
            del val_batch, sample_data
        except NameError:
            pass
            
        gc.collect()
        torch.cuda.empty_cache()
    global_step += 1

progress_bar.close()
print("\n--- Training Complete ---")
print(f"Final model saved to '{os.path.join(output_dir, 'checkpoints/best_autoencoder_model.pth')}'")