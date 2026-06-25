import os
import sys
import torch
import numpy as np
import nibabel as nib
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from datasets.dataset_utils import build_dataloaders
from models.custom_AEKL import DualDecoderVAE

# ==============================================================================
# --- 1. CONFIGURATION ---
# ==============================================================================
class InferenceConfig:
    data_dir = 'data'
    img_dir = os.path.join(data_dir, 'images')
    label_dir = os.path.join(data_dir, 'labels/')
    
    # Checkpoint to load
    model_path = "results/001_VAE3D_dualdecoder/checkpoints/best_autoencoder_model.pth"
    
    # CSV splits
    train_csv = os.path.join(data_dir, 'train_split_final.csv')
    test_csv = os.path.join(data_dir, 'test_split_final.csv')
    
    # Directory to save reconstructed outputs
    output_dir = "results/001_VAE3D_dualdecoder/inference_results/"
    
    # Generation parameters
    patch_size = (192, 192, 16)
    num_samples_to_test = 5 # Number of validation samples to reconstruct
    latent_dim = 8
    num_classes = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = InferenceConfig()
os.makedirs(cfg.output_dir, exist_ok=True)

# ==============================================================================
# --- 2. MAIN EXECUTION ---
# ==============================================================================
def main():
    print("--- 1. Loading Trained DualDecoderVAE ---")
    vae = DualDecoderVAE(latent_dim=cfg.latent_dim, num_classes=cfg.num_classes).to(cfg.device)
    
    if not os.path.exists(cfg.model_path):
        raise FileNotFoundError(f"Trained VAE checkpoint not found at: {cfg.model_path}")
        
    vae.load_state_dict(torch.load(cfg.model_path, map_location=cfg.device))
    vae.eval()
    print(f" -> Successfully loaded model weights from: {cfg.model_path}")

    print("\n--- 2. Loading Validation Dataset ---")
    # We only need the validation loader to test the model
    _, _, val_loader = build_dataloaders(
        data_dir=cfg.img_dir,
        label_dir=cfg.label_dir,
        patch_size=cfg.patch_size,
        batch_size=1,
        num_workers=2,
        train_csv=cfg.train_csv,
        test_csv=cfg.test_csv
    )
    print(f" -> Validation loader initialized with {len(val_loader)} files.")

    print(f"\n--- 3. Running Reconstruction on first {cfg.num_samples_to_test} samples ---")
    samples_processed = 0

    with torch.no_grad():
        for i, val_batch in enumerate(val_loader):
            if samples_processed >= cfg.num_samples_to_test:
                break
                
            if val_batch is None:
                continue

            v_imgs = val_batch["image"].to(cfg.device)
            v_has_lbl = val_batch.get("has_label", torch.tensor([0]))[0].item()
            
            # Extract NIfTI file path from MONAI metadata
            fname = val_batch['image_meta_dict']['filename_or_obj']
            if isinstance(fname, list):
                fname = fname[0]
            
            basename = os.path.basename(fname).replace(".nii.gz", "")
            print(f"[{samples_processed + 1}/{cfg.num_samples_to_test}] Processing: {basename}")

            # Model Forward Pass
            v_rec_img, v_rec_seg, _, _ = vae(v_imgs)
            
            # Squeeze batch dimension for NumPy conversion
            orig_np = v_imgs[0, 0].cpu().numpy()
            rec_np = v_rec_img[0, 0].cpu().numpy()
            
            # Get segmentation class integers from probability logits
            rec_seg_int = torch.argmax(v_rec_seg[0], dim=0).cpu().numpy().astype(np.float32)

            # Crop padded Z-axis back to original shape using NIfTI header dimensions
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
                    print(f"    -> Cropped Z-axis from {current_z} back to original {original_z} slices.")
            except Exception as e:
                print(f"    [Warning] Failed to crop Z-axis: {e}")

            # Get spatial affine matrix from batch metadata
            affine = val_batch['image_meta_dict']['affine'][0].numpy()

            # Save Output NIfTIs
            nib.save(nib.Nifti1Image(orig_np, affine), os.path.join(cfg.output_dir, f"{basename}_orig.nii.gz"))
            nib.save(nib.Nifti1Image(rec_np, affine), os.path.join(cfg.output_dir, f"{basename}_rec.nii.gz"))
            nib.save(nib.Nifti1Image(rec_seg_int, affine), os.path.join(cfg.output_dir, f"{basename}_seg_rec.nii.gz"))
            
            # Also save ground truth label if it exists for comparison
            if v_has_lbl > 0:
                v_lbls = val_batch["label"][0, 0].cpu().numpy()
                if current_z > original_z:
                    v_lbls = v_lbls[..., start_idx:end_idx]
                nib.save(nib.Nifti1Image(v_lbls, affine), os.path.join(cfg.output_dir, f"{basename}_lbl_gt.nii.gz"))

            samples_processed += 1

    print(f"\n--- Inference Complete ---")
    print(f"Reconstructed NIfTI samples successfully saved to: {os.path.abspath(cfg.output_dir)}")


if __name__ == "__main__":
    main()