import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from monai.transforms import (
    Compose, LoadImaged, Spacingd, 
    ScaleIntensityRangePercentilesd, EnsureTyped, SpatialPadd,
    CenterSpatialCropd
)
import nibabel as nib
from tqdm import tqdm

from models.custom_AEKL import DualDecoderVAE 

# --- 1. CONFIGURATION ---
MODEL_PATH = "results/001_VAE3D_dualdecoder/checkpoints/best_autoencoder_model.pth"
data_dir = 'data/'
LATENT_DIR = os.path.join(data_dir, 'latents')
ORIGINAL_DATA_DIR = os.path.join(data_dir, 'images')
LABEL_DATA_DIR = os.path.join(data_dir, 'labels')
RECON_DIR = "results/001_VAE3D_dualdecoder/reconstructions"

LATENT_DIM = 8
NUM_CLASSES = 4
TARGET_SPATIAL_SIZE = (192, 192, 16) 

NUM_SAMPLES_TO_DECODE = 20

torch.manual_seed(42)

def plot_comparison(orig_img, recon_img, orig_seg, recon_seg, filepath):
    """
    Plots a 2x2 comparison. 
    Expects 2D numpy arrays (sliced).
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    # Row 1: Images
    axes[0, 0].imshow(np.rot90(orig_img), cmap="gray")
    axes[0, 0].set_title("Original Image (Cropped)")
    axes[0, 0].axis("off")
    
    axes[0, 1].imshow(np.rot90(recon_img), cmap="gray")
    axes[0, 1].set_title("Reconstructed Image (Unpadded)")
    axes[0, 1].axis("off")
    
    # Row 2: Segmentations
    axes[1, 0].imshow(np.rot90(orig_seg), cmap="tab10", vmin=0, vmax=3, interpolation="nearest")
    axes[1, 0].set_title("Original GT")
    axes[1, 0].axis("off")
    
    axes[1, 1].imshow(np.rot90(recon_seg), cmap="tab10", vmin=0, vmax=3, interpolation="nearest")
    axes[1, 1].set_title("Reconstructed Seg")
    axes[1, 1].axis("off")
    
    fig.suptitle(f"Reconstruction Check: {os.path.basename(filepath)}")
    plt.tight_layout()
    plt.show()

def main():
    print("--- Starting VAE Decoding (DualDecoder) ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(RECON_DIR, exist_ok=True)

    # --- 1. Load Model ---
    autoencoder = DualDecoderVAE(
        latent_dim=LATENT_DIM, 
        num_classes=NUM_CLASSES
    ).to(device)
    
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    autoencoder.load_state_dict(checkpoint)
    autoencoder.eval()
    print("Model loaded.")

    # --- 2. Define Transforms to Load GT ---
    # We need to replicate the EXACT state of the image before it entered the VAE
    # so we can compare apples to apples.
    preprocess_transforms = Compose([
        LoadImaged(keys=["image", "label"], image_only=False, ensure_channel_first=True, allow_missing_keys=True),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, -1.0), mode=("bilinear", "nearest"), allow_missing_keys=True),
        # Note: We do NOT apply SpatialPad here yet, because we want to measure the "real" size
        ScaleIntensityRangePercentilesd(keys=["image"], lower=0.5, upper=99.5, b_min=-1.0, b_max=1.0, clip=True),
        EnsureTyped(keys=["image", "label"], dtype=torch.float32, allow_missing_keys=True),
    ])
    
    # Separate Padder to apply manually
    padder = SpatialPadd(
        keys=["image", "label"], 
        spatial_size=TARGET_SPATIAL_SIZE, 
        method="symmetric", 
        mode="edge", allow_missing_keys=True
    )
    cropper = CenterSpatialCropd(keys=["image", "label"], roi_size=TARGET_SPATIAL_SIZE, allow_missing_keys=True)

    # --- 3. Process Files ---
    # Get latents
    latent_files = sorted(glob.glob(os.path.join(LATENT_DIR, "DSB_9*4_*.pt")))
    
    # Filter for ACDC to ensure we have labels to look at
    latent_files = [f for f in latent_files if "DSB" in f]
    
    if NUM_SAMPLES_TO_DECODE > 0: 
        latent_files = latent_files[:NUM_SAMPLES_TO_DECODE]

    with torch.no_grad():
        for latent_filepath in tqdm(latent_files, desc="Decoding"):
            
            # A. Load Latent
            # Shape: (C, H_lat, W_lat, D_lat) -> Need (1, C, ...)
            z = torch.load(latent_filepath, weights_only=False).to(device).unsqueeze(0)
            
            # B. Decode using Dual Decoders
            # 1. Decode Image
            recon_img_padded = autoencoder.decoder_img(z) # (1, 1, 192, 192, 16)
            
            # 2. Decode Segmentation
            recon_seg_logits_padded = autoencoder.decoder_seg(z) # (1, 4, 192, 192, 16)
            recon_seg_padded = torch.argmax(recon_seg_logits_padded, dim=1, keepdim=True).float()
            
            # C. Load Original GT to compare
            base_filename = os.path.basename(latent_filepath).replace(".pt", ".nii.gz")
            img_path = os.path.join(ORIGINAL_DATA_DIR, base_filename)
            lbl_path = os.path.join(LABEL_DATA_DIR, base_filename)
            
            input_dict = {"image": img_path}
            if os.path.exists(lbl_path): input_dict["label"] = lbl_path
            
            # Load and Preprocess (up to padding)
            gt_data = preprocess_transforms(input_dict)
            if gt_data is None: continue # Skip if filter kicked it out

            # Get the "True" depth before padding
            # This is the Z-dimension after Spacing/Filtering but BEFORE Padding
            true_depth = gt_data["image"].shape[-1] 
            
            # Apply padding to GT so it matches the VAE output 1:1 for slicing
            if gt_data["image"].shape[-1] < 16:
                gt_data = padder(gt_data)
            else:
                gt_data = cropper(gt_data)

            gt_img = gt_data["image"].to(device).unsqueeze(0) # (1, 1, 192, 192, 16)
            
            if "label" in gt_data:
                gt_seg = gt_data["label"].to(device).unsqueeze(0)
            else:
                gt_seg = torch.zeros_like(gt_img)

            # --- D. THE UN-PADDING LOGIC ---
            # We know the output is 16 slices. We know the true data is 'true_depth'.
            # Padding was symmetric.
            
            current_depth = TARGET_SPATIAL_SIZE[2] # 16
            
            if true_depth < current_depth:
                diff = current_depth - true_depth
                start = diff // 2
                end = start + true_depth
            else:
                # If true depth was >= 16, we just take the whole thing
                start = 0
                end = current_depth
                
            # Crop tensors
            recon_img_clean = recon_img_padded[..., start:end]
            recon_seg_clean = recon_seg_padded[..., start:end]
            gt_img_clean = gt_img[..., start:end]
            gt_seg_clean = gt_seg[..., start:end]

            # E. Save NIfTI
            # Use original affine from GT
            meta = gt_data["image_meta_dict"]
            affine = meta["affine"].numpy()
            
            save_name = base_filename.replace(".nii.gz", "")
            
            nib.save(
                nib.Nifti1Image(recon_img_clean.squeeze().cpu().numpy(), affine), 
                os.path.join(RECON_DIR, f"{save_name}_recon_img.nii.gz")
            )
            nib.save(
                nib.Nifti1Image(recon_seg_clean.squeeze().cpu().numpy(), affine), 
                os.path.join(RECON_DIR, f"{save_name}_recon_seg.nii.gz")
            )
            
            # F. Visualization
            # Pick the middle slice of the CLEAN (unpadded) volume
            mid_slice = recon_img_clean.shape[-1] // 2
            
            orig_slice_np = gt_img_clean[0, 0, :, :, mid_slice].cpu().numpy()
            recon_slice_np = recon_img_clean[0, 0, :, :, mid_slice].cpu().numpy()
            
            orig_seg_np = gt_seg_clean[0, 0, :, :, mid_slice].cpu().numpy()
            recon_seg_np = recon_seg_clean[0, 0, :, :, mid_slice].cpu().numpy()

            plot_comparison(orig_slice_np, recon_slice_np, orig_seg_np, recon_seg_np, img_path)

    print("\n--- Decoding Complete ---")

if __name__ == "__main__":
    main()