import os
import torch
import numpy as np
import imageio
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from monai.transforms import (
    Compose,
    LoadImaged,
    ScaleIntensityRangePercentilesd,
    EnsureTyped,
    Spacingd,
    SpatialPadd, CenterSpatialCropd
)
from accelerate import Accelerator
from diffusers.optimization import get_scheduler
from diffusers import DDIMScheduler, UNet3DConditionModel

# =========================================================================================
# --- 1. ARCHITECTURE DEFINITIONS ---
# =========================================================================================
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
import sys
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Shared motion and conditioning modules
from models.custom_AEKL import DualDecoderVAE
from models.latent_motion import LatentFeatureMotionPredictor
from models.conditioning import SinusoidalPositionalEmbedding, ClinicalConditioningModel
from utils.common import normalize_scalars

# =========================================================================================
# --- 2. CONFIGURATION ---
# =========================================================================================
class InferenceConfig:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Paths ---
    # Update these to your actual trained checkpoints
    data_dir = "data/images"
    vae_model_path = "results/001_VAE3D_dualdecoder/checkpoints/best_autoencoder_model.pth"
    lfm_flow_predictor_path = "results/002_LatentMotion/checkpoints/flow_3.pth"
    
    # These usually come from your Flow Diffusion output folder
    diffusion_unet_path = "results/004_FlowDiffusion/checkpoints/checkpoint_step_2" # Update path
    
    output_dir = "results/004_FlowDiffusion/inference_output"
    gif_path = os.path.join(output_dir, "structure_conditioned_motion.gif")

    # --- Reference Image ---
    # The static anatomy we want to animate (Frame 000)
    ref_image_path = os.path.join(data_dir, "ACDC_patient101_frame_000.nii.gz")
    
    # --- Clinical Inputs (The "Prompt") ---
    # We want to force the motion to look like this condition:
    target_diagnosis = "DCM"  # Options: "NOR", "MINF", "DCM", "HCM", "RV"
    target_ef = 23.0          # Ejection Fraction (Low for DCM)
    target_edv = 223.0        # End Diastolic Volume (Dilated)
    target_esv = 80.0        # End Systolic Volume
    target_slices = 14.0      # Number of slices (affects Z-motion/padding)
    
    # --- Generation Params ---
    seed = 42
    num_inference_steps = 30
    guidance_scale = 5.0  # CFG Scale (Higher = stricter adherence to scalars)

    # --- Data/Model Params ---
    vae_scaling = 0.643433
    flow_scaling = 1.65092982 # From your training logs
    
    latent_channels = 8
    num_diagnosis = 7
    motion_in_channels = 8 + 8 + 1 # Latent (8) + Ref Latent (8) + Mask (1) = 17 (Adjust if your model is different)
    motion_out_channels = 8     # Delta Z (8)
    cross_attention_dim = 512
    spatial_size = (192, 192, 16) # Padded size
    target_pixdim = (1.0, 1.0, 10.0)
    
    # Normalization Limits 
    MAX_EDV = 600.0
    MAX_ESV = 600.0
    MAX_SLICES = 20.0
    
    # Diagnosis Map
    DIAGNOSIS_MAP = { "Unknown": 0, "NOR": 1, "MINF": 2, "DCM": 3, "HCM": 4, "RV": 5 }
    
    # --- Timing for Patient 101 (Ground Truth for comparison) ---
    ed_frame_idx = 0
    es_frame_idx = 12
    num_total_frames = 30 

cfg = InferenceConfig()
os.makedirs(cfg.output_dir, exist_ok=True)

# =========================================================================================
# --- 3. UTILITIES ---
# =========================================================================================

def calculate_normalized_time(target_frame_idx, cfg):
    """
    Returns time in range 0.0 (ED) -> 1.0 (ES) -> 2.0 (Next ED).
    Matches the training logic exactly.
    """
    contraction_len = (cfg.es_frame_idx - cfg.ed_frame_idx + cfg.num_total_frames) % cfg.num_total_frames
    relaxation_len = cfg.num_total_frames - contraction_len
    
    dist = (target_frame_idx - cfg.ed_frame_idx + cfg.num_total_frames) % cfg.num_total_frames
    
    if dist <= contraction_len:
        # Systole (0.0 -> 1.0)
        return 1.0 * (dist / contraction_len)
    else:
        # Diastole (1.0 -> 2.0)
        return 1.0 + 1.0 * ((dist - contraction_len) / relaxation_len)

def get_transforms(cfg):
    return Compose([
        LoadImaged(keys=["image"], image_only=False, ensure_channel_first=True),
        Spacingd(keys=["image"], pixdim=cfg.target_pixdim, mode=("bilinear")),
        
        # 1. Pad with edge replication (avoids black border artifacts)
        SpatialPadd(keys=["image"], spatial_size=cfg.spatial_size, method="symmetric", mode="edge"),
        # 2. Crop to fixed size
        CenterSpatialCropd(keys=["image"], roi_size=cfg.spatial_size),

        ScaleIntensityRangePercentilesd(keys=["image"], lower=0.5, upper=99.5, b_min=-1.0, b_max=1.0, clip=True),
        EnsureTyped(keys=["image"], dtype=torch.float32),
    ])

@torch.no_grad()
def decode_and_split(vae, latent, scaling):
    """Decodes latent to Image and Segmentation."""
    # Latent: (B, C, D, H, W)
    img = vae.decoder_img(latent / scaling)
    decoded_seg = vae.decoder_seg(latent / scaling)
    seg = torch.argmax(decoded_seg, dim=1, keepdim=True).float()
    return img, seg

# =========================================================================================
# --- 4. MODEL LOADING ---
# =========================================================================================
def extract_motion_models(ckpt_path):
    """Reconstructs and loads Motion models from Accelerate checkpoint."""
    print(f"--- Extracting MOTION models from {ckpt_path} ---")
    accelerator = Accelerator(mixed_precision="no")
    
    # Re-init Models
    cond_model = ClinicalConditioningModel(embedding_dim=cfg.cross_attention_dim, num_diagnosis=cfg.num_diagnosis)
    unet = UNet3DConditionModel(
        in_channels=cfg.motion_in_channels, out_channels=cfg.motion_out_channels,
        block_out_channels=(256, 512, 768, 768), layers_per_block=2,
        cross_attention_dim=cfg.cross_attention_dim,
        attention_head_dim=64,
        down_block_types=("CrossAttnDownBlock3D", "CrossAttnDownBlock3D", "CrossAttnDownBlock3D", "DownBlock3D"),
        up_block_types=("UpBlock3D", "CrossAttnUpBlock3D", "CrossAttnUpBlock3D", "CrossAttnUpBlock3D"),
    )
    
    # Re-init Dummy Objects
    optimizer = torch.optim.AdamW(list(unet.parameters()) + list(cond_model.parameters()))
    scheduler = get_scheduler("constant", optimizer=optimizer)
    dummy_loader = DataLoader([0], batch_size=1)
    
    # Prepare
    unet, cond_model, optimizer, dummy_loader, scheduler = accelerator.prepare(
        unet, cond_model, optimizer, dummy_loader, scheduler
    )
    
    # Load
    accelerator.load_state(ckpt_path)
    return accelerator.unwrap_model(unet), accelerator.unwrap_model(cond_model)

@torch.no_grad()
def load_models(cfg):
    print("--- Loading models ---")
    
    # 1. VAE
    vae = DualDecoderVAE(latent_dim=cfg.latent_channels, num_classes=4)
    vae.load_state_dict(torch.load(cfg.vae_model_path, map_location="cpu"))
    vae.eval().to(cfg.device)

    # 2. LFM Predictor
    lfm = LatentFeatureMotionPredictor(in_channels=2 * cfg.latent_channels, latent_channels=cfg.latent_channels)
    lfm.load_state_dict(torch.load(cfg.lfm_flow_predictor_path, map_location="cpu"))
    lfm.eval().to(cfg.device)
    
    
    # 4. Time Embedder
    time_embedder = SinusoidalPositionalEmbedding(cfg.cross_attention_dim).to(cfg.device)

    unet, cond_model = extract_motion_models(cfg.diffusion_unet_path)
    cond_model.eval().to(cfg.device)
    unet.eval().to(cfg.device)
    
    # 6. Scheduler
    scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule="scaled_linear", clip_sample=False)
    scheduler.set_timesteps(cfg.num_inference_steps)

    return {"vae": vae, "lfm": lfm, "cond": cond_model, "time": time_embedder, "unet": unet, "sched": scheduler}

# =========================================================================================
# --- 5. MAIN ---
# =========================================================================================
def main():
    torch.manual_seed(cfg.seed)
    models = load_models(cfg)
    transforms = get_transforms(cfg)

    # 1. Load Reference (Frame 0)
    print(f"Loading reference: {cfg.ref_image_path}")
    ref_data = transforms({"image": cfg.ref_image_path})
    ref_img = ref_data["image"].to(cfg.device).unsqueeze(0) # (1, 1, 192, 192, 16)
    
    with torch.no_grad():
        _, _, ref_latent, _ = models["vae"](ref_img)
        ref_latent = ref_latent * cfg.vae_scaling

    _, _, h, w, d = ref_latent.shape
    mask = torch.zeros((1, 1, h, w, d), device=cfg.device)
    valid_slices = min(int(cfg.target_slices), d)
    valid_slices = max(1, valid_slices)
    diff = d - valid_slices
    start_idx = diff // 2
    end_idx = start_idx + valid_slices
    mask[..., start_idx:end_idx] = 1.0

    # 2. Prepare Conditioning (The "Prompt")
    print(f"Conditioning: {cfg.target_diagnosis}, EF={cfg.target_ef}%, EDV={cfg.target_edv}ml")
    
    # A. Normalize Scalars
    scalars_norm = normalize_scalars(cfg.target_ef, cfg.target_edv, cfg.target_esv, cfg.target_slices)
    scalars_tensor = torch.tensor([scalars_norm], dtype=torch.float32, device=cfg.device) # (1, 4)
    
    # B. Diagnosis ID
    diag_id = cfg.DIAGNOSIS_MAP.get(cfg.target_diagnosis, 0)
    diag_tensor = torch.tensor([diag_id], dtype=torch.long, device=cfg.device) # (1,)
    
    # C. Unconditional Inputs (for CFG)
    # Mask class is index 6, scalars are 0
    uncond_diag = torch.tensor([6], dtype=torch.long, device=cfg.device)
    uncond_scalars = torch.zeros_like(scalars_tensor)

    # D. Encode Clinical Features (Static part of context)
    with torch.no_grad():
        cond_clinical_emb = models["cond"](diag_tensor, scalars_tensor)     # (1, 5, 512)
        uncond_clinical_emb = models["cond"](uncond_diag, uncond_scalars)   # (1, 5, 512)

    # 3. Generate Loop
    gif_frames = []
    frames_to_gen = list(range(0, cfg.num_total_frames))
    
    # Latent shape for noise
    noise_shape = (1, cfg.latent_channels, *ref_latent.shape[2:])
    fixed_noise = torch.randn(noise_shape, device=cfg.device) 
    motion_vmax = None 

    for frame_idx in tqdm(frames_to_gen, desc="Generating Frames"):
        # A. Calculate Time
        time_val = calculate_normalized_time(frame_idx, cfg)
        time_tensor = torch.tensor([[time_val * 100]], dtype=torch.float32, device=cfg.device)
        # B. Diffusion Generation
        with torch.no_grad():
            # Embed Time
            time_emb = models["time"](time_tensor).unsqueeze(1) # (1, 1, 512)
            
            # Construct Full Context [Clinical tokens, Time token]
            # Cond Context: (1, 6, 512)
            context_cond = torch.cat([cond_clinical_emb, time_emb], dim=1)
            context_uncond = torch.cat([uncond_clinical_emb, time_emb], dim=1)
            
            # Batch for CFG
            context_batch = torch.cat([context_uncond, context_cond]) # (2, 6, 512)
            
            # Start Sampling
            latents = fixed_noise.clone()
            
            for t in models["sched"].timesteps:
                # Input: [Noisy Flow, Ref Latent]
                latent_input = torch.cat([latents] * 2)
                ref_input = torch.cat([ref_latent] * 2)
                mask_input = torch.cat([mask] * 2) # Mask required
                unet_input = torch.cat([latent_input, ref_input, mask_input], dim=1)
                
                # Predict Noise
                noise_pred = models["unet"](unet_input, t, encoder_hidden_states=context_batch).sample
                
                # CFG
                noise_uncond, noise_cond = noise_pred.chunk(2)
                noise_cfg = noise_uncond + cfg.guidance_scale * (noise_cond - noise_uncond)
                
                # Step
                latents = models["sched"].step(noise_cfg, t, latents).prev_sample
                
            # Final Flow
            pred_flow = latents / cfg.flow_scaling
            motion_energy_gen = torch.linalg.norm(pred_flow, ord=2, dim=1, keepdim=True)
            
            # Warp & Decode
            z_warped = ref_latent + pred_flow
            img_gen, seg_gen = decode_and_split(models["vae"], z_warped, cfg.vae_scaling)

        # C. Ground Truth Comparison (LFM)
        gt_path = os.path.join(cfg.data_dir, f"ACDC_patient101_frame_{frame_idx:03d}.nii.gz")
        if os.path.exists(gt_path):
            gt_data = transforms({"image": gt_path})
            gt_img = gt_data["image"].to(cfg.device).unsqueeze(0)
            with torch.no_grad():
                _, _, gt_latent, _ = models["vae"](gt_img)
                gt_latent = gt_latent * cfg.vae_scaling
                
                # LFM Prediction (Ref + Target -> Flow)
                lfm_flow = models["lfm"](ref_latent, gt_latent)
                z_lfm = ref_latent + lfm_flow
                img_lfm, seg_lfm = decode_and_split(models["vae"], z_lfm, cfg.vae_scaling)
                motion_energy_lfm = torch.linalg.norm(lfm_flow, ord=2, dim=1, keepdim=True)
        else:
            img_lfm = torch.zeros_like(img_gen)
            seg_lfm = torch.zeros_like(seg_gen)

        # D. Visualization
        def to_np(t, z_slice): 
            return np.rot90(t.squeeze().cpu().numpy()[:, :, z_slice])
            
        mid_slice = img_gen.shape[-1] // 2
        
        # Generated (Diffusion)
        i_gen = to_np(img_gen, mid_slice)
        s_gen = to_np(seg_gen, mid_slice)
        m_gen = to_np(motion_energy_gen, mid_slice)
        
        # Ground Truth (LFM)
        i_lfm = to_np(img_lfm, mid_slice)
        s_lfm = to_np(seg_lfm, mid_slice)
        m_lfm = to_np(motion_energy_lfm, mid_slice)

        # Plot
        fig, axs = plt.subplots(3, 2, figsize=(8, 12), facecolor='black')
        plt.suptitle(f"Frame {frame_idx} (t={time_val:.2f})", color='white')
        
        axs[0,0].imshow(i_gen, cmap='gray'); axs[0,0].set_title("Diff Gen Image", color='w')
        axs[0,1].imshow(i_lfm, cmap='gray'); axs[0,1].set_title("LFM GT Image", color='w')
        axs[1,0].imshow(s_gen, cmap='tab10', vmin=0, vmax=3); axs[1,0].set_title("Diff Gen Seg", color='w')
        axs[1,1].imshow(s_lfm, cmap='tab10', vmin=0, vmax=3); axs[1,1].set_title("LFM GT Seg", color='w')
        axs[2,0].imshow(m_gen, cmap='magma', vmin=0, vmax=motion_vmax); axs[2,0].set_title("Motion Energy Gen", color='w')
        axs[2,1].imshow(m_lfm, cmap='magma', vmin=0, vmax=motion_vmax); axs[2,1].set_title("Motion Energy LFM", color='w')
        
        for ax in axs.flatten(): ax.axis('off')
        
        fig.canvas.draw()
        arr = np.array(fig.canvas.renderer.buffer_rgba())
        gif_frames.append(arr)
        plt.close()

    print(f"Saving GIF to {cfg.gif_path}...")
    imageio.mimsave(cfg.gif_path, gif_frames, fps=10,loop=0)
    print("Done.")

if __name__ == "__main__":
    main()