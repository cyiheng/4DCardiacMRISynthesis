import os
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
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
    print("Please ensure 'custom_AEKL.py' is in the repository root or on PYTHONPATH.")
    sys.exit(1)

from models.latent_motion import LatentFeatureMotionPredictor
from utils.common import save_nifti, encode_image

# =========================================================================================
# --- 1. CONFIGURATION ---
# =========================================================================================
class Config:
    # --- Input Data ---
    # Adjust these paths to the images you want to test
    ref_image_path = "data/images/ACDC_patient101_frame_000.nii.gz"
    dri_image_path = "data/images/ACDC_patient101_frame_013.nii.gz"

    # --- Paths to Trained Models ---
    # Output directory defined in training
    train_output_dir = "results/002_LatentMotion/" 
    
    # Path to the specific checkpoints you want to load
    # Example: Loading step 50000 or the final models
    flow_checkpoint = os.path.join(train_output_dir, "checkpoints/flow_3.pth")
    
    # Base VAE weights (needed for the Encoder, which was frozen during LFM training)
    base_vae_path = "results/001_VAE3D_dualdecoder/checkpoints/best_autoencoder_model.pth"

    # --- Output ---
    output_dir = os.path.join(train_output_dir, "inference")

    # --- Model & Image Parameters (MUST MATCH TRAINING) ---
    spatial_size = (192, 192, 16) 
    target_pixdim = (1.0, 1.0, 10.0) # Note: Z pixdim depends on your preprocessing
    latent_channels = 8
    num_classes = 4
    vae_scaling = 0.643433 # Match value from training config
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================================================
# --- 2. MAIN INFERENCE SCRIPT ---
# =========================================================================================
def main():
    print("--- Starting DualDecoder LFM Inference ---")
    cfg = Config()
    os.makedirs(cfg.output_dir, exist_ok=True)

    # --- 1. TRANSFORMS (MATCHING TRAINING) ---
    # We must replicate the preprocessing exactly (Spacing, Padding, Scaling)
    inference_transforms = Compose([
        LoadImaged(keys=["ref", "dri"], image_only=False, ensure_channel_first=True),
        Spacingd(keys=["ref", "dri"], pixdim=(1.0, 1.0, -1.0), mode="bilinear"), # Match training pixdim logic
        SpatialPadd(keys=["ref", "dri"], spatial_size=cfg.spatial_size, method="symmetric", mode="edge"),
        CenterSpatialCropd(keys=["ref", "dri"], roi_size=cfg.spatial_size),
        ScaleIntensityRangePercentilesd(keys=["ref", "dri"], lower=0.5, upper=99.5, b_min=-1.0, b_max=1.0, clip=True),
        EnsureTyped(keys=["ref", "dri"], dtype=torch.float32),
    ])

    # --- 2. LOAD DATA ---
    print(f"Loading Reference: {cfg.ref_image_path}")
    print(f"Loading Driving:   {cfg.dri_image_path}")
    
    data_dict = inference_transforms({"ref": cfg.ref_image_path, "dri": cfg.dri_image_path})
    
    ref_img = data_dict["ref"].unsqueeze(0).to(cfg.device)
    dri_img = data_dict["dri"].unsqueeze(0).to(cfg.device)
    
    # Save original affine for saving outputs
    # Note: This affine includes the Spacing/Padding transforms
    ref_affine = data_dict["ref_meta_dict"]["affine"]

    # --- 3. MODEL INITIALIZATION ---
    print("Initializing models...")

    # A. Dual Decoder VAE
    vae = DualDecoderVAE(latent_dim=cfg.latent_channels, num_classes=cfg.num_classes).to(cfg.device)
    
    # 1. Load Base Weights (for Encoder)
    print(f"Loading base VAE weights: {cfg.base_vae_path}")
    base_sd = torch.load(cfg.base_vae_path, map_location=cfg.device)
    vae.load_state_dict(base_sd, strict=False) # strict=False in case LFM introduced mismatches (unlikely)

    # B. Flow Predictor
    flow_predictor = LatentFeatureMotionPredictor(
        in_channels=2 * cfg.latent_channels, 
        latent_channels=cfg.latent_channels
    ).to(cfg.device)
    
    print(f"Loading Flow Predictor: {cfg.flow_checkpoint}")
    flow_predictor.load_state_dict(torch.load(cfg.flow_checkpoint, map_location=cfg.device))

    vae.eval()
    flow_predictor.eval()

    # --- 4. INFERENCE LOOP ---
    print("Running Inference...")
    with torch.no_grad():
        # Step 1: Encode Images -> Latents
        # The training script loaded pre-calculated latents. Here we encode on the fly.
        z_ref_raw = encode_image(vae, ref_img)
        z_dri_raw = encode_image(vae, dri_img)
        
        # Step 2: Apply Scaling (Critical!)
        z_ref = z_ref_raw * cfg.vae_scaling
        z_dri = z_dri_raw * cfg.vae_scaling
        
        # Step 3: Predict Motion (Feature Addition)
        # Note: Training logic: feature_motion = flow(ref, dri)
        feature_motion = flow_predictor(z_ref, z_dri)
        
        # Step 4: Apply Motion
        z_warped = z_ref + feature_motion
        
        # Step 5: Decode (Dual Outputs)
        # Remember to unscale!
        z_input_decoder = z_warped / cfg.vae_scaling
        
        img_recon = vae.decoder_img(z_input_decoder)
        seg_logits = vae.decoder_seg(z_input_decoder)
        seg_recon = torch.argmax(seg_logits, dim=1, keepdim=True).float()

    # --- 5. SAVING OUTPUTS ---
    print("Saving results...")
    
    # 1. Warped Image
    save_nifti(img_recon, ref_affine, os.path.join(cfg.output_dir, "warped_image.nii.gz"))
    
    # 2. Warped Segmentation
    save_nifti(seg_recon, ref_affine, os.path.join(cfg.output_dir, "warped_segmentation.nii.gz"))
    
    # 3. Reference Input (Preprocessed)
    save_nifti(ref_img, ref_affine, os.path.join(cfg.output_dir, "input_ref.nii.gz"))
    
    # 4. Driving Input (Preprocessed - Target)
    save_nifti(dri_img, ref_affine, os.path.join(cfg.output_dir, "input_dri.nii.gz"))
    
    # 5. Motion Energy (Visualizing where the change happened)
    motion_energy = torch.mean(torch.abs(feature_motion), dim=1, keepdim=True)
    # Resize motion energy to image size for visualization
    motion_energy_resized = F.interpolate(motion_energy, size=cfg.spatial_size, mode='trilinear', align_corners=False)
    save_nifti(motion_energy_resized, ref_affine, os.path.join(cfg.output_dir, "motion_energy.nii.gz"))

    print("Done.")

if __name__ == "__main__":
    main()