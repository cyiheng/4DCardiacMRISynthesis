import os
import random
import numpy as np
import nibabel as nib
import torch
from tqdm.auto import tqdm
import imageio
import tempfile
import gradio as gr
import matplotlib.pyplot as plt
from PIL import Image
from safetensors.torch import load_file
from diffusers import DDIMScheduler, UNet3DConditionModel
from monai.networks.nets import DiffusionModelUNet

# --- IMPORTS FROM YOUR PROJECT ---
try:
    from models.custom_AEKL import DualDecoderVAE
    from models.conditioning import ClinicalConditioningModel, SinusoidalPositionalEmbedding
except ImportError as e:
    print(f"Import Error: {e}")
    print("Please ensure you are running this script from the root of your project directory.")


# =========================================================================================
# --- 1. CONFIGURATION ---
# =========================================================================================
class InferenceConfig:
    vae_path = "demo/weights/vae.pth"
    anat_ckpt_dir = "demo/weights/anatomy" 
    motion_ckpt_dir = "demo/weights/motion"
    
    spatial_size = (192, 192, 16)
    latent_shape = (8, 24, 24, 16)
    latent_channels = 8
    
    anat_in_channels = 9      
    cross_attention_dim = 512
    num_diagnosis = 7
    num_classes = 4        
    
    motion_in_channels = 17   
    motion_out_channels = 8
    
    vae_scale = 0.643433
    flow_scale = 1.65115030
    
    MAX_EDV = 600.0
    MAX_ESV = 600.0
    MAX_SLICES = 20.0
    
    target_affine = np.eye(4)
    target_affine[0,0] = 1.0; target_affine[1,1] = 1.0; target_affine[2,2] = 10.0
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = InferenceConfig()

DIAGNOSIS_MAP = {"Unknown": 0, "NOR": 1, "MINF": 2, "DCM": 3, "HCM": 4, "RV": 5}

DEFAULT_PARAMS = {
    "NOR (Normal function)": (65.0, 140.0, 10),
    "DCM (Dilated Cardiomyopathy)": (25.0, 280.0, 10),
    "HCM (Hypertrophic Cardiomyopathy)": (70.0, 110.0, 12),
    "MINF (Myocardial Infarction)": (40.0, 180.0, 10),
    "RV (Abnormal Right Ventricle)": (55.0, 140.0, 10)
}

MODELS = {}

# =========================================================================================
# --- 2. OPTIMIZED DIRECT LOADERS ---
# =========================================================================================

def load_anatomy_models(ckpt_dir):
    cond_model = ClinicalConditioningModel(embedding_dim=cfg.cross_attention_dim, num_diagnosis=cfg.num_diagnosis)
    unet = DiffusionModelUNet(
        spatial_dims=3, in_channels=cfg.anat_in_channels, out_channels=cfg.latent_channels,
        channels=(256, 512, 1024), attention_levels=(False, True, True),
        num_res_blocks=2, num_head_channels=32, with_conditioning=True,
        cross_attention_dim=cfg.cross_attention_dim
    )
    unet.load_state_dict(load_file(os.path.join(ckpt_dir, "model.safetensors"), device="cpu"))
    cond_model.load_state_dict(load_file(os.path.join(ckpt_dir, "model_1.safetensors"), device="cpu"))
    return unet, cond_model

def load_motion_models(ckpt_dir):
    cond_model = ClinicalConditioningModel(embedding_dim=cfg.cross_attention_dim, num_diagnosis=cfg.num_diagnosis)
    unet = UNet3DConditionModel(
        in_channels=cfg.motion_in_channels, out_channels=cfg.motion_out_channels,
        block_out_channels=(256, 512, 768, 768), layers_per_block=2,
        cross_attention_dim=cfg.cross_attention_dim, attention_head_dim=64,
        down_block_types=("CrossAttnDownBlock3D", "CrossAttnDownBlock3D", "CrossAttnDownBlock3D", "DownBlock3D"),
        up_block_types=("UpBlock3D", "CrossAttnUpBlock3D", "CrossAttnUpBlock3D", "CrossAttnUpBlock3D"),
    )
    unet.load_state_dict(load_file(os.path.join(ckpt_dir, "model.safetensors"), device="cpu"))
    cond_model.load_state_dict(load_file(os.path.join(ckpt_dir, "model_1.safetensors"), device="cpu"))
    return unet, cond_model

def load_models():
    if MODELS: return
    print("Loading 4D pre-trained models from safetensors...")
    
    vae = DualDecoderVAE(latent_dim=cfg.latent_channels, num_classes=cfg.num_classes)
    vae.load_state_dict(torch.load(cfg.vae_path, map_location="cpu", weights_only=True))
    MODELS['vae'] = vae.to(cfg.device).eval()

    unet_anat, cond_model_anat = load_anatomy_models(cfg.anat_ckpt_dir)
    MODELS['unet_anat'] = unet_anat.to(cfg.device).eval()
    MODELS['cond_model_anat'] = cond_model_anat.to(cfg.device).eval()

    unet_motion, cond_model_motion = load_motion_models(cfg.motion_ckpt_dir)
    MODELS['unet_motion'] = unet_motion.to(cfg.device).eval()
    MODELS['cond_model_motion'] = cond_model_motion.to(cfg.device).eval()
    
    MODELS['time_embedder'] = SinusoidalPositionalEmbedding(cfg.cross_attention_dim).to(cfg.device).eval()
    MODELS['scheduler'] = DDIMScheduler(num_train_timesteps=1000, beta_schedule="scaled_linear", clip_sample=False)
    print("All models loaded successfully.")

# =========================================================================================
# --- 3. LOGIC & PLOTTING ---
# =========================================================================================
def normalize_scalars(ef, edv, esv, n_slices):
    ef_n = max(0.0, min(1.0, ef / 100.0))
    edv_n = max(0.0, min(1.0, edv / cfg.MAX_EDV))
    esv_n = max(0.0, min(1.0, esv / cfg.MAX_ESV))
    slices_n = max(0.0, min(1.0, n_slices / cfg.MAX_SLICES))
    return [ef_n, edv_n, esv_n, slices_n]

def generate_cardiac_time_signal(num_frames: int, only_systole: bool = False) -> torch.Tensor:
    if only_systole:
        final_signal = np.linspace(0, 100, num_frames)
    else:
        final_signal = np.linspace(0, 200, num_frames)
    final_signal[0] = 0.0
    final_signal[-1] = 100.0 if only_systole else 200.0
    return torch.from_numpy(final_signal).float()

def create_visual_frame(vol_np, seg_np):
    mid = vol_np.shape[2] // 2
    img = np.rot90(vol_np[:, :, mid])
    seg = np.rot90(seg_np[:, :, mid])
    img_n = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img_rgb = np.stack([img_n]*3, axis=-1)
    seg_rgb = np.zeros_like(img_rgb)
    seg_rgb[seg==1] = [1, 0, 0]; seg_rgb[seg==2] = [0, 1, 0]; seg_rgb[seg==3] = [0, 0, 1]
    ov = img_rgb.copy()
    ov[seg>0] = 0.6 * img_rgb[seg>0] + 0.4 * seg_rgb[seg>0]
    return (np.concatenate([img_rgb, seg_rgb, ov], axis=1) * 255).astype(np.uint8)

def update_defaults(diagnosis_selection):
    ef, edv, slices = DEFAULT_PARAMS.get(diagnosis_selection, (65.0, 140.0, 10))
    return ef, edv, slices

def reset_all_defaults(diagnosis_selection):
    ef, edv, slices = DEFAULT_PARAMS.get(diagnosis_selection, (65.0, 140.0, 10))
    return ef, edv, slices, 30, False, 42, 30, 30, 3.0, 3.0

# --- Tri-Planar Viewer Logic ---
def update_viewer(vol_4d, seg_4d, t, x, y, z, show_seg):
    if vol_4d is None or seg_4d is None:
        return None, None, None
    
    # Extract slices based on coordinates
    axial_img = vol_4d[:, :, z, t]
    axial_seg = seg_4d[:, :, z, t]
    
    coronal_img = vol_4d[:, y, :, t]
    coronal_seg = seg_4d[:, y, :, t]
    
    sagittal_img = vol_4d[x, :, :, t]
    sagittal_seg = seg_4d[x, :, :, t]
    
    def render(img_slice, seg_slice, is_z_axis=False):
        img_n = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min() + 1e-8)
        img_rgb = (np.stack([img_n]*3, axis=-1) * 255).astype(np.uint8)
        
        if show_seg:
            seg_rgb = np.zeros_like(img_rgb)
            seg_rgb[seg_slice==1] = [255, 50, 50]   # RV (Red)
            seg_rgb[seg_slice==2] = [50, 255, 50]   # Myo (Green)
            seg_rgb[seg_slice==3] = [50, 50, 255]   # LV (Blue)
            mask = seg_slice > 0
            img_rgb[mask] = (0.5 * img_rgb[mask] + 0.5 * seg_rgb[mask]).astype(np.uint8)
        
        img_rgb = np.rot90(img_rgb)
        pil_img = Image.fromarray(img_rgb)
        
        if is_z_axis:
            # Stretch the Z-axis by 10x to match realistic physical proportions
            w, h = pil_img.size
            pil_img = pil_img.resize((w, h * 10), resample=Image.NEAREST if show_seg else Image.BILINEAR)
            
        return np.array(pil_img)

    return render(axial_img, axial_seg, False), render(coronal_img, coronal_seg, True), render(sagittal_img, sagittal_seg, True)

# =========================================================================================
# --- 4. GENERATION PIPELINE ---
# =========================================================================================
def generate_4d_cine(
    diag_choice, ef, edv, n_slices, num_frames, only_systole, 
    steps_anat, steps_motion, guidance_anat, guidance_motion, seed,
    progress=gr.Progress(track_tqdm=True)
):
    output_dir = tempfile.mkdtemp()
    
    if seed == -1: seed = np.random.randint(0, 1_000_000)
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    
    esv = edv * (1 - ef / 100.0)
    norm_vals = normalize_scalars(ef, edv, esv, n_slices)
    diag_id = DIAGNOSIS_MAP.get(diag_choice.split(" ")[0], 1)
    diag_t = torch.tensor([diag_id], device=cfg.device)
    scal_t = torch.tensor([norm_vals], device=cfg.device, dtype=torch.float32)
    
    # --- Stage 1: ED Anatomy ---
    with torch.no_grad():
        cond_emb_a = MODELS['cond_model_anat'](diag_t, scal_t)
        uncond_emb_a = MODELS['cond_model_anat'](torch.tensor([0], device=cfg.device), torch.zeros_like(scal_t))
        ctx_anat = torch.cat([uncond_emb_a, cond_emb_a])

    l_h, l_w, l_d = cfg.latent_shape[1:]
    mask = torch.zeros((1, 1, l_h, l_w, l_d), device=cfg.device)
    valid_slices = min(int(n_slices), l_d)
    s_idx = (l_d - valid_slices) // 2
    e_idx = s_idx + valid_slices
    mask[..., s_idx:e_idx] = 1.0
    
    latents_0 = torch.randn((1, cfg.latent_channels, l_h, l_w, l_d), device=cfg.device)
    MODELS['scheduler'].set_timesteps(steps_anat)
    
    with torch.no_grad(), torch.autocast("cuda"):
        for t in tqdm(MODELS['scheduler'].timesteps, desc="Anatomy Steps"):
            inp = torch.cat([latents_0] * 2); m_inp = torch.cat([mask] * 2)
            noise = MODELS['unet_anat'](x=torch.cat([inp, m_inp], dim=1), context=ctx_anat, timesteps=t.expand(2).to(cfg.device))
            u, c = noise.chunk(2)
            latents_0 = MODELS['scheduler'].step(u + guidance_anat * (c - u), t, latents_0).prev_sample
        
        dec_img_0 = MODELS['vae'].decoder_img(latents_0 / cfg.vae_scale).squeeze().cpu().numpy()
        dec_seg_0 = torch.argmax(MODELS['vae'].decoder_seg(latents_0 / cfg.vae_scale), dim=1).squeeze().cpu().numpy().astype(np.uint8)

    # --- Stage 2: 4D Motion ---
    with torch.no_grad():
        cond_emb_m = MODELS['cond_model_motion'](diag_t, scal_t)
        uncond_emb_m = MODELS['cond_model_motion'](torch.tensor([0], device=cfg.device), torch.zeros_like(scal_t))

    time_signal = generate_cardiac_time_signal(num_frames, only_systole)
    img_frames, seg_frames = [dec_img_0[..., s_idx:e_idx]], [dec_seg_0[..., s_idx:e_idx]]
    fixed_noise_m = torch.randn_like(latents_0)

    for f_idx, val in enumerate(tqdm(time_signal, desc="Generating Frames")):
        if f_idx == 0: continue
        t_emb = MODELS['time_embedder'](torch.tensor([[val]], device=cfg.device, dtype=torch.float32)).unsqueeze(1)
        ctx_m_full = torch.cat([torch.cat([uncond_emb_m, t_emb], dim=1), torch.cat([cond_emb_m, t_emb], dim=1)])
        
        latents_flow = fixed_noise_m.clone()
        MODELS['scheduler'].set_timesteps(steps_motion)
        ref_in, msk_in = torch.cat([latents_0] * 2), torch.cat([mask] * 2)

        with torch.no_grad(), torch.autocast("cuda"):
            for t in MODELS['scheduler'].timesteps:
                minp = torch.cat([torch.cat([latents_flow] * 2), ref_in, msk_in], dim=1)
                pred = MODELS['unet_motion'](minp, t.expand(2).to(cfg.device), encoder_hidden_states=ctx_m_full).sample
                u, c = pred.chunk(2)
                latents_flow = MODELS['scheduler'].step(u + guidance_motion * (c - u), t, latents_flow).prev_sample
            
            latents_t = latents_0 + (latents_flow / cfg.flow_scale)
            img_frames.append(MODELS['vae'].decoder_img(latents_t / cfg.vae_scale).squeeze().cpu().numpy()[..., s_idx:e_idx])
            seg_frames.append(torch.argmax(MODELS['vae'].decoder_seg(latents_t / cfg.vae_scale), dim=1).squeeze().cpu().numpy().astype(np.uint8)[..., s_idx:e_idx])

    # --- Calculations & Volumes ---
    voxel_vol_ml = 1.0 * 1.0 * 10.0 / 1000.0  
    lv_vols, rv_vols, myo_vols = [], [], []
    for seg in seg_frames:
        rv_vols.append(np.sum(seg == 1) * voxel_vol_ml)
        myo_vols.append(np.sum(seg == 2) * voxel_vol_ml)
        lv_vols.append(np.sum(seg == 3) * voxel_vol_ml)

    gen_edv, gen_esv = max(lv_vols), min(lv_vols)
    gen_ef = ((gen_edv - gen_esv) / gen_edv * 100) if gen_edv > 0 else 0.0

    metrics_md = f"### 📊 Generated Metrics\n* **EDV**: {gen_edv:.1f} mL (Target: {edv} mL)\n* **ESV**: {gen_esv:.1f} mL\n* **EF**: {gen_ef:.1f} % (Target: {ef}%)"

    plt.figure(figsize=(8, 4))
    plt.plot(lv_vols, label="LV Cavity", color='blue', linewidth=2)
    plt.plot(rv_vols, label="RV Cavity", color='red', linewidth=2)
    plt.plot(myo_vols, label="Myocardium", color='green', linewidth=2)
    plt.xlabel("Time Frame"); plt.ylabel("Volume (mL)"); plt.title("Generated Cardiac Volume Curves over Time")
    plt.grid(True, linestyle="--", alpha=0.6); plt.legend(); plt.tight_layout()
    plot_path = os.path.join(output_dir, "volume_curves.png")
    plt.savefig(plot_path); plt.close()

    # Float32 cast fixes nibabel float16 bug
    vol_4d = np.stack(img_frames, axis=-1).astype(np.float32)
    seg_4d = np.stack(seg_frames, axis=-1).astype(np.uint8)
    nifti_vol_path = os.path.join(output_dir, "generated_vol.nii.gz")
    nifti_seg_path = os.path.join(output_dir, "generated_seg.nii.gz")
    nib.save(nib.Nifti1Image(vol_4d, cfg.target_affine), nifti_vol_path)
    nib.save(nib.Nifti1Image(seg_4d, cfg.target_affine), nifti_seg_path)

    gif_path = os.path.join(output_dir, "visualization.gif")
    imageio.mimsave(gif_path, [create_visual_frame(i, s) for i, s in zip(img_frames, seg_frames)], fps=10, loop=0)

    # Initial views for the tri-planar viewer
    mid_t, mid_x, mid_y, mid_z = num_frames // 2, 96, 96, valid_slices // 2
    ax, cor, sag = update_viewer(vol_4d, seg_4d, mid_t, mid_x, mid_y, mid_z, show_seg=False)

    return (
        # Files/Plots
        gif_path, nifti_vol_path, nifti_seg_path, plot_path, metrics_md,
        # States for viewer
        vol_4d, seg_4d,
        # Slider updates (adjust max limits dynamically based on generation)
        gr.update(maximum=num_frames-1, value=0),
        gr.update(maximum=191, value=mid_x),
        gr.update(maximum=191, value=mid_y),
        gr.update(maximum=valid_slices-1, value=mid_z),
        # Initial Images
        ax, cor, sag
    )

# =========================================================================================
# --- 5. GRADIO UI ---
# =========================================================================================
def create_ui():
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 4D Clinical-to-Cine MRI Generation")
        gr.Markdown(
            """
            > **⚠️ Note to Users:** 
            > * This is a generative AI model still under active research. The generated anatomy and motion may occasionally exhibit artifacts or unphysiological characteristics.
            > * The sliders are constrained to realistic human boundaries for stability, but these ranges can be extended directly in the application code if extreme edge cases are desired.
            > * It was tested using a RTX4090 with 24GB, but uses around 9GB of memory with default settings.
            """
        )

        # --- STATE VARIABLES FOR 3D VIEWER ---
        state_vol = gr.State(None)
        state_seg = gr.State(None)

        with gr.Row():
            # --- LEFT COLUMN: INPUTS ---
            with gr.Column(scale=1):
                gr.Markdown("### Clinical Parameters")
                diagnosis = gr.Dropdown(choices=list(DEFAULT_PARAMS.keys()), value="NOR (Normal function)", label="Diagnosis")
                ef = gr.Slider(minimum=10.0, maximum=90.0, value=65.0, step=1.0, label="Ejection Fraction (EF) (%)")
                edv = gr.Slider(minimum=50.0, maximum=400.0, value=140.0, step=5.0, label="End-Diastolic Volume (EDV) (mL)")
                n_slices = gr.Slider(minimum=6, maximum=16, value=10, step=1, label="Number of Z-Slices")
                
                diagnosis.change(fn=update_defaults, inputs=[diagnosis], outputs=[ef, edv, n_slices])
                
                gr.Markdown("### Sequence Configuration")
                num_frames = gr.Slider(minimum=10, maximum=50, value=30, step=1, label="Time Frames")
                only_systole = gr.Checkbox(label="Only generate Systole (ED to ES)", value=False)
                
                with gr.Accordion("Advanced AI Settings", open=False):
                    seed = gr.Number(label="Seed (-1 for random)", value=42)
                    steps_anat = gr.Slider(minimum=10, maximum=100, value=30, step=1, label="Anatomy Steps")
                    steps_motion = gr.Slider(minimum=10, maximum=100, value=30, step=1, label="Motion Steps")
                    guidance_anat = gr.Slider(minimum=1.0, maximum=10.0, value=3.0, step=0.5, label="Anatomy CFG")
                    guidance_motion = gr.Slider(minimum=1.0, maximum=10.0, value=3.0, step=0.5, label="Motion CFG")

                with gr.Row():
                    generate_btn = gr.Button("Generate 4D Cine", variant="primary")
                    reset_btn = gr.Button("Reset Defaults")
                
                reset_btn.click(
                    fn=reset_all_defaults, inputs=[diagnosis],
                    outputs=[ef, edv, n_slices, num_frames, only_systole, seed, steps_anat, steps_motion, guidance_anat, guidance_motion]
                )

            # --- RIGHT COLUMN: OUTPUTS ---
            with gr.Column(scale=2):
                with gr.Tabs():
                    # TAB 1: GIF Summary
                    with gr.TabItem("Middle-Slice Summary"):
                        output_gif = gr.Image(label="Cine MRI (Mid-Slice)", type="filepath")

                    # TAB 2: 3D Tri-Planar Viewer (ITK-SNAP style)
                    with gr.TabItem("Interactive 3D Viewer"):
                        gr.Markdown("Explore the generated 4D volume. Use the sliders to move through space and time.")
                        with gr.Row():
                            viewer_time = gr.Slider(minimum=0, maximum=29, value=0, step=1, label="Time Frame (T)")
                            show_seg_toggle = gr.Checkbox(label="Overlay Segmentation Masks", value=False)
                        
                        with gr.Row():
                            with gr.Column():
                                viewer_z = gr.Slider(minimum=0, maximum=15, value=7, step=1, label="Axial Slice (Z)")
                                view_axial = gr.Image(label="Axial (XY)", interactive=False)
                            with gr.Column():
                                viewer_y = gr.Slider(minimum=0, maximum=191, value=96, step=1, label="Coronal Slice (Y)")
                                view_coronal = gr.Image(label="Coronal (XZ)", interactive=False)
                            with gr.Column():
                                viewer_x = gr.Slider(minimum=0, maximum=191, value=96, step=1, label="Sagittal Slice (X)")
                                view_sagittal = gr.Image(label="Sagittal (YZ)", interactive=False)
                        
                        # Hook up viewer interactions
                        viewer_inputs = [state_vol, state_seg, viewer_time, viewer_x, viewer_y, viewer_z, show_seg_toggle]
                        viewer_outputs = [view_axial, view_coronal, view_sagittal]
                        
                        viewer_time.change(fn=update_viewer, inputs=viewer_inputs, outputs=viewer_outputs)
                        viewer_x.change(fn=update_viewer, inputs=viewer_inputs, outputs=viewer_outputs)
                        viewer_y.change(fn=update_viewer, inputs=viewer_inputs, outputs=viewer_outputs)
                        viewer_z.change(fn=update_viewer, inputs=viewer_inputs, outputs=viewer_outputs)
                        show_seg_toggle.change(fn=update_viewer, inputs=viewer_inputs, outputs=viewer_outputs)

                    
                    # TAB 3: Metrics & Plots
                    with gr.TabItem("Volumetric Metrics"):
                        output_metrics = gr.Markdown("*(Metrics will appear here after generation)*")
                        output_plot = gr.Image(label="Volume Curve Analysis", type="filepath")
                    
                    # TAB 4: Downloads
                    with gr.TabItem("Download 4D NIfTI"):
                        gr.Markdown("Download the raw 4D generated volumes to view in ITK-SNAP or 3D Slicer.")
                        output_nifti_vol = gr.File(label="4D MRI Volume (NIfTI)")
                        output_nifti_seg = gr.File(label="4D Segmentation Mask (NIfTI)")

        # Main Generate Hook
        gen_inputs = [
            diagnosis, ef, edv, n_slices, num_frames, only_systole,
            steps_anat, steps_motion, guidance_anat, guidance_motion, seed
        ]
        gen_outputs = [
            output_gif, output_nifti_vol, output_nifti_seg, output_plot, output_metrics,
            state_vol, state_seg,
            viewer_time, viewer_x, viewer_y, viewer_z,
            view_axial, view_coronal, view_sagittal
        ]
        
        generate_btn.click(fn=generate_4d_cine, inputs=gen_inputs, outputs=gen_outputs)

    return demo

if __name__ == "__main__":
    try:
        load_models()
        app = create_ui()
        app.launch()
    except Exception as e:
        print(f"Failed to start application: {e}")