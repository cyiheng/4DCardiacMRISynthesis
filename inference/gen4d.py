import os
import random
import numpy as np
import pandas as pd
import nibabel as nib
import torch
import imageio
from diffusers import DDIMScheduler, UNet3DConditionModel
from monai.networks.nets import DiffusionModelUNet
from safetensors.torch import load_file

try:
    from models.custom_AEKL import DualDecoderVAE
except ImportError:
    print("Error: Could not import DualDecoderVAE from custom_AEKL.")
    exit()

from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
import sys
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.conditioning import ClinicalConditioningModel, SinusoidalPositionalEmbedding
# =========================================================================================
# --- 1. CONFIGURATION ---
# =========================================================================================
class GenConfig:
    # --- General ---
    SEED = 42                 # Base seed (will increment if anatomy fails validity check)
    SAVE_4D = True            # True = Save (H,W,D,T), False = Save individual 3D files
    
    # --- Motion Control ---
    GENERATE_MOTION = True
    NUM_FRAMES = 30           # Total frames
    ONLY_SYSTOLE = False      # False = Full Cycle (ED->ES->ED)
    random_params = True      # True = Random clinical params, False = Sample from ACDC stats
    
    # --- Paths ---
    acdc_train_csv = "data/ACDC_Preprocessed/train_metadata.csv" 
    output_base_dir = "./results/full_generation/" 
    img_dir = os.path.join(output_base_dir, "images")
    lbl_dir = os.path.join(output_base_dir, "labels")
    vis_dir = os.path.join(output_base_dir, "visuals")
    csv_path = os.path.join(output_base_dir, "metadata.csv")
    
    # --- Checkpoints ---
    vae_path = "results/001_VAE3D_dualdecoder/checkpoints/best_autoencoder_model.pth"
    anat_ckpt_dir = "results/003_FirstFrame3D/checkpoints/ckpt_20" 
    motion_ckpt_dir = "results/004_FlowDiffusion/checkpoints/checkpoint_step_2"
    # vae_path = "demo/weights/vae.pth"
    # anat_ckpt_dir = "demo/weights/anatomy" 
    # motion_ckpt_dir = "demo/weights/motion"
    
    # --- Model Params ---
    spatial_size = (192, 192, 16)
    latent_shape = (8, 24, 24, 16)
    latent_channels = 8
    
    # Anatomy
    anat_in_channels = 9      # 8 Latent + 1 Mask
    cross_attention_dim = 512
    num_diagnosis = 7
    num_classes = 4        
    
    # Motion
    motion_in_channels = 17   # 8 Latent + 8 Ref + 1 Mask
    motion_out_channels = 8
    
    # Scales
    vae_scale = 0.643433
    flow_scale = 1.65115030
    
    # Limits for Normalization
    MAX_EDV = 600.0; MAX_ESV = 600.0; MAX_SLICES = 20.0
    
    # Inference
    num_inference_steps = 30
    guidance_scale_anat = 3.0
    guidance_scale_motion = 3.0
    
    target_affine = np.eye(4)
    target_affine[0,0]=1; target_affine[1,1]=1; target_affine[2,2]=10.0

cfg = GenConfig()
os.makedirs(cfg.img_dir, exist_ok=True)
os.makedirs(cfg.lbl_dir, exist_ok=True)
os.makedirs(cfg.vis_dir, exist_ok=True)

DIAGNOSIS_MAP = {"Unknown": 0, "NOR": 1, "MINF": 2, "DCM": 3, "HCM": 4, "RV": 5}

def get_acdc_stats(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"ACDC CSV not found at {csv_path}")
    
    df = pd.read_csv(csv_path)
    df['pathology'] = df['pathology'].str.upper().str.strip()
    stats = {}
    valid_paths = ["NOR", "MINF", "DCM", "HCM", "RV"]
    
    for p in valid_paths:
        subset = df[df['pathology'] == p]
        stats[p] = {
            "min_edv": subset['edv'].min(), "max_edv": subset['edv'].max(),
            "min_ef": subset['ef'].min(),   "max_ef": subset['ef'].max(),
            "min_slices": int(subset['frame_00_n_slices'].min()) if 'frame_00_n_slices' in df.columns else 6,
            "max_slices": int(subset['frame_00_n_slices'].max()) if 'frame_00_n_slices' in df.columns else 14,
        }
    return stats

# =========================================================================================
# --- 2. MODEL LOADING ---
# =========================================================================================

def extract_anatomy_models(ckpt_path):
    """Loads Anatomy models directly from .safetensors files."""
    print(f"--- Loading ANATOMY models from {ckpt_path} ---")
    
    cond_model = ClinicalConditioningModel(
        embedding_dim=cfg.cross_attention_dim, 
        num_diagnosis=cfg.num_diagnosis
    )
    unet = DiffusionModelUNet(
        spatial_dims=3, in_channels=cfg.anat_in_channels, out_channels=cfg.latent_channels,
        channels=(256, 512, 1024), attention_levels=(False, True, True),
        num_res_blocks=2, num_head_channels=32, with_conditioning=True,
        cross_attention_dim=cfg.cross_attention_dim
    )
    
    # Define safetensors paths
    unet_weights_path = os.path.join(ckpt_path, "model.safetensors")
    cond_weights_path = os.path.join(ckpt_path, "model_1.safetensors")
    
    if not os.path.exists(unet_weights_path) or not os.path.exists(cond_weights_path):
        raise FileNotFoundError(
            f"Could not find model weight safetensors in {ckpt_path}. "
            "Ensure 'model.safetensors' and 'model_1.safetensors' exist."
        )

    # Load states using safetensors.torch.load_file
    unet.load_state_dict(load_file(unet_weights_path, device="cpu"))
    cond_model.load_state_dict(load_file(cond_weights_path, device="cpu"))
    
    print(" -> Anatomy safetensors loaded successfully.")
    return unet, cond_model


def extract_motion_models(ckpt_path):
    """Loads Motion models directly from .safetensors files."""
    print(f"--- Loading MOTION models from {ckpt_path} ---")
    
    cond_model = ClinicalConditioningModel(
        embedding_dim=cfg.cross_attention_dim, 
        num_diagnosis=cfg.num_diagnosis
    )
    unet = UNet3DConditionModel(
        in_channels=cfg.motion_in_channels, out_channels=cfg.motion_out_channels,
        block_out_channels=(256, 512, 768, 768), layers_per_block=2,
        cross_attention_dim=cfg.cross_attention_dim, attention_head_dim=64,
        down_block_types=("CrossAttnDownBlock3D", "CrossAttnDownBlock3D", "CrossAttnDownBlock3D", "DownBlock3D"),
        up_block_types=("UpBlock3D", "CrossAttnUpBlock3D", "CrossAttnUpBlock3D", "CrossAttnUpBlock3D"),
    )
    
    # Define safetensors paths
    unet_weights_path = os.path.join(ckpt_path, "model.safetensors")
    cond_weights_path = os.path.join(ckpt_path, "model_1.safetensors")
    
    if not os.path.exists(unet_weights_path) or not os.path.exists(cond_weights_path):
        raise FileNotFoundError(
            f"Could not find model weight safetensors in {ckpt_path}. "
            "Ensure 'model.safetensors' and 'model_1.safetensors' exist."
        )

    # Load states using safetensors.torch.load_file
    unet.load_state_dict(load_file(unet_weights_path, device="cpu"))
    cond_model.load_state_dict(load_file(cond_weights_path, device="cpu"))
    
    print(" -> Motion safetensors loaded successfully.")
    return unet, cond_model

# =========================================================================================
# --- 3. DATA UTILS ---
# =========================================================================================
def generate_random_clinical_data():
    diag_choice = random.choice(["NOR", "DCM", "HCM", "MINF", "RV"])
    if diag_choice == "NOR": edv, ef = random.uniform(100, 180), random.uniform(55, 75)
    elif diag_choice == "DCM": edv, ef = random.uniform(200, 350), random.uniform(15, 45)
    elif diag_choice == "HCM": edv, ef = random.uniform(80, 150), random.uniform(60, 80)
    elif diag_choice == "MINF": edv, ef = random.uniform(130, 220), random.uniform(30, 50)
    else: edv, ef = random.uniform(120, 200), random.uniform(40, 60)

    esv = edv * (1 - ef/100.0)
    n_slices = random.randint(6, 14)
    ef_n = max(0.0, min(1.0, ef / 100.0))
    edv_n = max(0.0, min(1.0, edv / cfg.MAX_EDV))
    esv_n = max(0.0, min(1.0, esv / cfg.MAX_ESV))
    slices_n = max(0.0, min(1.0, n_slices / cfg.MAX_SLICES))

    return {
        "diagnosis": diag_choice, "diag_id": DIAGNOSIS_MAP[diag_choice],
        "ef": ef, "edv": edv, "esv": esv, "n_slices": n_slices,
        "normalized": [ef_n, edv_n, esv_n, slices_n]
    }

def sample_clinical_data(pathology, stats_dict):
    s = stats_dict[pathology]
    pad_edv = (s['max_edv'] - s['min_edv']) * 0.05
    pad_ef = (s['max_ef'] - s['min_ef']) * 0.05
    
    edv = random.uniform(max(50, s['min_edv'] - pad_edv), s['max_edv'] + pad_edv)
    ef = random.uniform(max(10, s['min_ef'] - pad_ef), min(90, s['max_ef'] + pad_ef))
    n_slices = random.randint(s['min_slices'], min(s['max_slices'], 18))
    esv = edv * (1 - ef/100.0)
    
    ef_n = max(0.0, min(1.0, ef / 100.0))
    edv_n = max(0.0, min(1.0, edv / cfg.MAX_EDV))
    esv_n = max(0.0, min(1.0, esv / cfg.MAX_ESV))
    slices_n = max(0.0, min(1.0, n_slices / cfg.MAX_SLICES))
    return {
        "diagnosis": pathology, "diag_id": DIAGNOSIS_MAP[pathology],
        "ef": ef, "edv": edv, "esv": esv, "n_slices": n_slices,
        "normalized": [ef_n, edv_n, esv_n, slices_n]
    }

def is_anatomically_valid(seg_np):
    lv_mask = (seg_np == 3); myo_mask = (seg_np == 2)
    if np.sum(lv_mask) < 100 or np.sum(myo_mask) < 100: return False
    z_indices = np.where(np.any(lv_mask, axis=(0,1)))[0]
    if len(z_indices) < 2: return False
    if (z_indices.max() - z_indices.min() + 1) != len(z_indices): return False
    return True

def create_visual_frame(vol_np, seg_np):
    mid = vol_np.shape[2] // 2
    img = np.rot90(vol_np[:, :, mid])
    seg = np.rot90(seg_np[:, :, mid])
    img_n = (img - img.min())/(img.max()-img.min()+1e-8)
    img_rgb = np.stack([img_n]*3, axis=-1)
    seg_rgb = np.zeros_like(img_rgb)
    seg_rgb[seg==1]=[1,0,0]; seg_rgb[seg==2]=[0,1,0]; seg_rgb[seg==3]=[0,0,1]
    ov = img_rgb.copy()
    ov[seg>0] = 0.6*img_rgb[seg>0] + 0.4*seg_rgb[seg>0]
    return (np.concatenate([img_rgb, seg_rgb, ov], axis=1)*255).astype(np.uint8)

# =========================================================================================
# --- 4. MAIN PIPELINE ---
# =========================================================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acdc_stats = get_acdc_stats(cfg.acdc_train_csv) if not cfg.random_params else None
    
    # 1. Load Models
    vae = DualDecoderVAE(latent_dim=cfg.latent_channels, num_classes=cfg.num_classes)
    vae.load_state_dict(torch.load(cfg.vae_path, map_location="cpu"))
    vae.to(device).eval()

    unet_anat, cond_model_anat = extract_anatomy_models(cfg.anat_ckpt_dir)
    unet_anat.to(device).eval()
    cond_model_anat.to(device).eval()

    unet_motion, cond_model_motion = extract_motion_models(cfg.motion_ckpt_dir)
    unet_motion.to(device).eval()
    cond_model_motion.to(device).eval()
    
    time_embedder = SinusoidalPositionalEmbedding(cfg.cross_attention_dim).to(device)
    scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule="scaled_linear", clip_sample=False)
    
    # 2. Rejection Sampling Loop for Exactly 1 Valid Case
    success = False
    attempt_count = 0
    
    print("\n--- Starting Generation ---")
    while not success:
        seed = cfg.SEED + attempt_count
        torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
        attempt_count += 1
        
        print(f"Attempting Case Generation (Seed: {seed})...")
        
        if cfg.random_params:
            params = generate_random_clinical_data()
        else:
            # Fallback to sampling NOR from ACDC Stats
            params = sample_clinical_data("NOR", acdc_stats)

        diag_t = torch.tensor([params['diag_id']], device=device)
        scal_t = torch.tensor([params['normalized']], device=device, dtype=torch.float32)
        
        with torch.no_grad():
            cond_emb_a = cond_model_anat(diag_t, scal_t)
            uncond_emb_a = cond_model_anat(torch.tensor([0], device=device), torch.zeros_like(scal_t))
            ctx_anat = torch.cat([uncond_emb_a, cond_emb_a])

        l_h, l_w, l_d = cfg.latent_shape[1:]
        mask = torch.zeros((1, 1, l_h, l_w, l_d), device=device)
        valid_slices = min(int(params['n_slices']), l_d)
        s_idx = (l_d - valid_slices) // 2
        e_idx = s_idx + valid_slices
        mask[..., s_idx:e_idx] = 1.0
        
        # Sample Anatomy
        latents_0 = torch.randn((1, cfg.latent_channels, l_h, l_w, l_d), device=device)
        scheduler.set_timesteps(cfg.num_inference_steps)
        
        with torch.no_grad():
            for t in scheduler.timesteps:
                inp = torch.cat([latents_0]*2)
                m_inp = torch.cat([mask]*2)
                x_in = torch.cat([inp, m_inp], dim=1)
                
                noise = unet_anat(x=x_in, context=ctx_anat, timesteps=t.expand(2).to(device))
                u, c = noise.chunk(2)
                noise = u + cfg.guidance_scale_anat * (c - u)
                latents_0 = scheduler.step(noise, t, latents_0).prev_sample
            
            dec_img_0 = vae.decoder_img(latents_0 / cfg.vae_scale).squeeze().cpu().numpy()
            dec_seg_0 = torch.argmax(vae.decoder_seg(latents_0 / cfg.vae_scale), dim=1).squeeze().cpu().numpy().astype(np.uint8)

        # Anatomy Check
        if not is_anatomically_valid(dec_seg_0):
            print(f" -> Invalid anatomy layout for seed {seed}. Re-sampling...")
            continue

        # If valid, generate motion and break out of loop
        print(f" -> Anatomy layout valid! Proceeding with motion generation...")
        success = True
        case_id = f"Syn_E2E_0000"
        
        with torch.no_grad():
            cond_emb_m = cond_model_motion(diag_t, scal_t)
            uncond_emb_m = cond_model_motion(torch.tensor([0], device=device), torch.zeros_like(scal_t))

        # Generate linear progression time signal
        num_systole_frames = max(2, int(cfg.NUM_FRAMES * 0.4))
        num_diastole_frames = cfg.NUM_FRAMES - num_systole_frames
        systole_times = np.linspace(0, 100, num_systole_frames, endpoint=False)
        diastole_times = np.linspace(100, 200, num_diastole_frames)
        time_signal = np.concatenate([systole_times, diastole_times])
        time_signal[0], time_signal[-1] = 0.0, 200.0
        time_signal = torch.from_numpy(time_signal).float()
        
        img_frames = [dec_img_0[..., s_idx:e_idx]] 
        seg_frames = [dec_seg_0[..., s_idx:e_idx]]
        fixed_noise_m = torch.randn_like(latents_0)

        for f_idx, val in enumerate(time_signal):
            if f_idx == 0: continue
            
            t_val = torch.tensor([[val]], device=device, dtype=torch.float32)
            t_emb = time_embedder(t_val).unsqueeze(1)
            ctx_m_full = torch.cat([
                torch.cat([uncond_emb_m, t_emb], dim=1),
                torch.cat([cond_emb_m, t_emb], dim=1)
            ])
            
            latents_flow = fixed_noise_m.clone()
            scheduler.set_timesteps(cfg.num_inference_steps)
            
            ref_in = torch.cat([latents_0]*2)
            msk_in = torch.cat([mask]*2)

            with torch.no_grad():
                for t in scheduler.timesteps:
                    lin = torch.cat([latents_flow]*2)
                    minp = torch.cat([lin, ref_in, msk_in], dim=1)
                    
                    pred = unet_motion(minp, t.expand(2).to(device), encoder_hidden_states=ctx_m_full).sample
                    u, c = pred.chunk(2)
                    pred = u + cfg.guidance_scale_motion * (c - u)
                    latents_flow = scheduler.step(pred, t, latents_flow).prev_sample
                
                latents_t = latents_0 + (latents_flow / cfg.flow_scale)
                img_t = vae.decoder_img(latents_t / cfg.vae_scale).squeeze().cpu().numpy()[..., s_idx:e_idx]
                seg_t = torch.argmax(vae.decoder_seg(latents_t / cfg.vae_scale), dim=1).squeeze().cpu().numpy().astype(np.uint8)[..., s_idx:e_idx]
                
                img_frames.append(img_t)
                seg_frames.append(seg_t)

        # Save 4D outputs
        vol_4d = np.stack(img_frames, axis=-1)
        seg_4d = np.stack(seg_frames, axis=-1)
        
        nib.save(nib.Nifti1Image(vol_4d, cfg.target_affine), os.path.join(cfg.img_dir, f"{case_id}.nii.gz"))
        nib.save(nib.Nifti1Image(seg_4d, cfg.target_affine), os.path.join(cfg.lbl_dir, f"{case_id}.nii.gz"))

        # Save animated GIF visualization
        gif_frames = [create_visual_frame(i, s) for i, s in zip(img_frames, seg_frames)]
        gif_path = os.path.join(cfg.vis_dir, f"{case_id}.gif")
        imageio.mimsave(gif_path, gif_frames, fps=10, loop=0)

        params['case_id'] = case_id
        pd.DataFrame([params]).to_csv(cfg.csv_path, index=False)
        print(f"Successfully generated and saved single synthetic heart to {cfg.output_base_dir}")

if __name__ == "__main__":
    main()