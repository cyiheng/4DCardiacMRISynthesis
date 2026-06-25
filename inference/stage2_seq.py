import os
import sys
from pathlib import Path
import torch
from monai.transforms import (
    Compose,
    LoadImaged,
    Spacingd,
    SpatialPadd,
    CenterSpatialCropd,
    ScaleIntensityRangePercentilesd,
    EnsureTyped,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# --- CUSTOM IMPORT ---
try:
    from models.custom_AEKL import DualDecoderVAE
except ImportError:
    print("Error: Could not import 'DualDecoderVAE' from 'custom_AEKL'.")
    sys.exit(1)

from models.latent_motion import LatentFeatureMotionPredictor
from utils.common import save_nifti_4d, encode_image

# =========================================================================================
# --- 1. CONFIGURATION ---
# =========================================================================================
class Config:
    # --- Data Definition ---
    # Defines the file pattern. We assume the folder contains frames named sequentially.
    # Example: ACDC_patient101_frame_000.nii.gz, ACDC_patient101_frame_001.nii.gz, etc.
    
    data_folder = "data/images"
    patient_prefix = "ACDC_patient101" # Common prefix for the files
    
    # We will iterate from frame_start to frame_end (inclusive)
    frame_start = 0 
    frame_end = 30
    
    # This is the "Fixed" image (usually ED, Frame 0)
    ref_frame_idx = 0 

    # --- Paths to Trained Models ---
    train_output_dir = "results/002_LatentMotion/" 
    flow_checkpoint = os.path.join(train_output_dir, "checkpoints/flow_3.pth")
    base_vae_path = "results/001_VAE3D_dualdecoder/checkpoints/best_autoencoder_model.pth"

    # --- Output ---
    output_dir = os.path.join(train_output_dir, "inference_real_sequence")

    # --- Model Parameters ---
    spatial_size = (192, 192, 16) 
    latent_channels = 8
    num_classes = 4
    vae_scaling = 0.643433 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================================================
# --- 3. HELPER FUNCTIONS ---
# =========================================================================================
def get_filename(cfg, frame_idx):
    """Constructs filename based on patient prefix and frame index."""
    fname = f"{cfg.patient_prefix}_frame_{frame_idx:03d}.nii.gz"
    return os.path.join(cfg.data_folder, fname)

# =========================================================================================
# --- 4. MAIN INFERENCE SCRIPT ---
# =========================================================================================
def main():
    print("--- Starting Pairwise Real-Image Inference ---")
    cfg = Config()
    os.makedirs(cfg.output_dir, exist_ok=True)

    # --- 1. MODEL INITIALIZATION ---
    print("Initializing models...")
    vae = DualDecoderVAE(latent_dim=cfg.latent_channels, num_classes=cfg.num_classes).to(cfg.device)
    
    # Load VAE weights
    base_sd = torch.load(cfg.base_vae_path, map_location=cfg.device)
    vae.load_state_dict(base_sd, strict=False)
    
    # Load Flow Predictor
    flow_predictor = LatentFeatureMotionPredictor(
        in_channels=2 * cfg.latent_channels, 
        latent_channels=cfg.latent_channels
    ).to(cfg.device)
    flow_predictor.load_state_dict(torch.load(cfg.flow_checkpoint, map_location=cfg.device))

    vae.eval()
    flow_predictor.eval()

    # --- 2. PREPROCESSING PIPELINE ---
    inference_transforms = Compose([
        LoadImaged(keys=["ref", "dri"], image_only=False, ensure_channel_first=True),
        Spacingd(keys=["ref", "dri"], pixdim=(1.0, 1.0, -1.0), mode="bilinear"),
        SpatialPadd(keys=["ref", "dri"], spatial_size=cfg.spatial_size, method="symmetric", mode="edge"),
        CenterSpatialCropd(keys=["ref", "dri"], roi_size=cfg.spatial_size),
        ScaleIntensityRangePercentilesd(keys=["ref", "dri"], lower=0.5, upper=99.5, b_min=-1.0, b_max=1.0, clip=True),
        EnsureTyped(keys=["ref", "dri"], dtype=torch.float32),
    ])

    # --- 3. SEQUENCE GENERATION LOOP ---
    print(f"Processing frames {cfg.frame_start} to {cfg.frame_end}...")
    
    generated_imgs = []
    generated_segs = []
    
    ref_path = get_filename(cfg, cfg.ref_frame_idx)
    ref_affine = None # Will store from first iteration

    with torch.no_grad():
        for t in range(cfg.frame_start, cfg.frame_end + 1):
            
            # A. Define Paths
            dri_path = get_filename(cfg, t)
            
            if not os.path.exists(dri_path):
                print(f"Warning: File not found {dri_path}. Skipping step.")
                continue

            print(f"  > Processing Frame {t}: Ref(0) -> Real Frame({t})")

            # B. Prepare Data Dictionary for Transforms
            # We process pairwise to ensure Ref and Dri are cropped/padded identically
            data_dict = {"ref": ref_path, "dri": dri_path}
            data_dict = inference_transforms(data_dict)
            
            if ref_affine is None:
                ref_affine = data_dict["ref_meta_dict"]["affine"]

            # C. Move to GPU
            ref_img = data_dict["ref"].unsqueeze(0).to(cfg.device) # (1, 1, H, W, D)
            dri_img = data_dict["dri"].unsqueeze(0).to(cfg.device) # (1, 1, H, W, D)

            # D. Encode
            z_ref_raw = encode_image(vae, ref_img)
            z_dri_raw = encode_image(vae, dri_img)
            
            z_ref = z_ref_raw * cfg.vae_scaling
            z_dri = z_dri_raw * cfg.vae_scaling
            
            # E. Calculate Motion (Ref -> Real Dri)
            # This is NOT interpolation. This is actual motion prediction.
            if t == cfg.ref_frame_idx:
                # For frame 0, motion is zero
                feature_motion = torch.zeros_like(z_ref)
            else:
                feature_motion = flow_predictor(z_ref, z_dri)
            
            # F. Apply Warp
            z_warped = z_ref + feature_motion
            
            # G. Decode
            z_in_decoder = z_warped / cfg.vae_scaling
            
            # 1. Image
            rec_img = vae.decoder_img(z_in_decoder)
            generated_imgs.append(rec_img.squeeze().cpu().numpy())
            
            # 2. Segmentation
            rec_logits = vae.decoder_seg(z_in_decoder)
            rec_seg = torch.argmax(rec_logits, dim=1).float()
            generated_segs.append(rec_seg.squeeze().cpu().numpy())

    # --- 4. SAVE OUTPUTS ---
    print("Saving 4D results...")
    
    save_nifti_4d(generated_imgs, ref_affine, os.path.join(cfg.output_dir, "generated_sequence_images.nii.gz"))
    save_nifti_4d(generated_segs, ref_affine, os.path.join(cfg.output_dir, "generated_sequence_segs.nii.gz"))
    
    print("Done.")

if __name__ == "__main__":
    main()