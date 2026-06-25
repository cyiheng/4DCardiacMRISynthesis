import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import torch
from monai.transforms import Compose, LoadImaged, Spacingd

def plot_and_save_learning_curve(g_loss_history, d_loss_history_or_path, save_path=None, smoothing_window=100):
    """
    Unified, polymorphic plot generator.
    
    Compatible with:
      - Single Loss: plot_and_save_learning_curve(loss_history, save_path, smoothing_window)
      - Dual Loss:   plot_and_save_learning_curve(g_loss_history, d_loss_history, save_path, smoothing_window)
    """
    if not g_loss_history:
        return

    # Check if the second argument is a string/path (Single-Loss Mode)
    if isinstance(d_loss_history_or_path, (str, os.PathLike)):
        # --- SINGLE LOSS MODE (e.g., Diffusion) ---
        loss_history = g_loss_history
        actual_save_path = d_loss_history_or_path
        actual_smoothing_window = save_path if isinstance(save_path, int) else smoothing_window
        
        plt.figure("Learning Curve", figsize=(12, 6))
        plt.title("Diffusion Model Training Loss")
        plt.plot(loss_history, label="Step Loss", alpha=0.3, color='dodgerblue')
        
        if len(loss_history) > actual_smoothing_window:
            smoothed_loss = np.convolve(loss_history, np.ones(actual_smoothing_window)/actual_smoothing_window, mode='valid')
            plt.plot(np.arange(actual_smoothing_window - 1, len(loss_history)), smoothed_loss, color='crimson', label="Smoothed")
            
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(actual_save_path)
        plt.close()
        
    else:
        # --- DUAL LOSS MODE (e.g., GAN / VAE / LFM) ---
        d_loss_history = d_loss_history_or_path
        actual_save_path = save_path if save_path else "learning_curve.png"
        actual_smoothing_window = smoothing_window
        
        plt.figure("Learning Curve", figsize=(16, 6))
        
        # Generator/Main Loss Subplot
        plt.subplot(1, 2, 1)
        plt.title("Generator Loss")
        plt.plot(g_loss_history, label="Step Loss", alpha=0.3)
        if len(g_loss_history) > actual_smoothing_window:
            smoothed = np.convolve(g_loss_history, np.ones(actual_smoothing_window) / actual_smoothing_window, mode="valid")
            plt.plot(np.arange(actual_smoothing_window - 1, len(g_loss_history)), smoothed, color="crimson", label="Smoothed")
        plt.yscale("log")
        plt.grid(True)
        plt.legend()

        # Discriminator/Adversarial Loss Subplot
        plt.subplot(1, 2, 2)
        plt.title("Discriminator Loss")
        if d_loss_history:
            plt.plot(d_loss_history, label="Step Loss", alpha=0.3, color="orange")
            if len(d_loss_history) > actual_smoothing_window:
                smoothed = np.convolve(d_loss_history, np.ones(actual_smoothing_window) / actual_smoothing_window, mode="valid")
                plt.plot(np.arange(actual_smoothing_window - 1, len(d_loss_history)), smoothed, color="darkorange", label="Smoothed")
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(actual_save_path)
        plt.close()



def visualize_fixed_pair(
    flow_predictor,
    vae,
    transforms,
    device,
    ref_latent_path,
    dri_latent_path,
    ref_image_path,
    dri_image_path,
    ref_label_path,
    dri_label_path,
    step,
    output_dir,
    vae_scaling,
):
    print(f"  -> Generating visual step {step}...")
    flow_predictor.eval()
    vae.eval()

    ref_latent = torch.load(ref_latent_path, weights_only=False, map_location=device).unsqueeze(0) * vae_scaling
    dri_latent_gt = torch.load(dri_latent_path, weights_only=False, map_location=device).unsqueeze(0) * vae_scaling

    input_dict = {
        "ref_image": ref_image_path,
        "dri_image": dri_image_path,
        "ref_label": ref_label_path if ref_label_path and os.path.exists(ref_label_path) else None,
        "dri_label": dri_label_path if dri_label_path and os.path.exists(dri_label_path) else None,
    }
    input_dict = {k: v for k, v in input_dict.items() if v is not None}
    gt_data = transforms(input_dict)

    ref_img = gt_data["ref_image"].to(device)
    dri_img = gt_data["dri_image"].to(device)
    ref_lbl = gt_data.get("ref_label", torch.zeros_like(ref_img)).to(device)
    dri_lbl = gt_data.get("dri_label", torch.zeros_like(dri_img)).to(device)

    with torch.no_grad():
        feature_motion = flow_predictor(ref_latent, dri_latent_gt)
        z_warped = ref_latent + feature_motion
        img_warped = vae.decoder_img(z_warped / vae_scaling)
        logits_warped = vae.decoder_seg(z_warped / vae_scaling)
        seg_warped = torch.argmax(logits_warped, dim=1, keepdim=True).float()

    def to_np(tensor):
        return tensor.squeeze().cpu().numpy()

    img_np = to_np(ref_img)
    id0, id1, id2 = img_np.shape
    is0, is1, is2 = id0 // 2, id1 // 2, id2 // 4

    fig, axes = plt.subplots(6, 3, figsize=(12, 24))
    plt.suptitle(f"LFM DualDecoder - Step {step}", fontsize=16)
    col_titles = ["Axial", "Coronal", "Sagittal"]
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title)

    def plot_row(row_idx, vol, title, cmap="gray", vmin=None, vmax=None):
        axes[row_idx, 0].set_ylabel(title, fontsize=10)
        axes[row_idx, 0].imshow(np.rot90(vol[is0, :, :]), cmap=cmap, vmin=vmin, vmax=vmax)
        axes[row_idx, 1].imshow(np.rot90(vol[:, is1, :]), cmap=cmap, vmin=vmin, vmax=vmax)
        axes[row_idx, 2].imshow(np.rot90(vol[:, :, is2]), cmap=cmap, vmin=vmin, vmax=vmax)
        for ax in axes[row_idx]:
            ax.axis("off")

    plot_row(0, to_np(ref_img), "Ref GT Img")
    plot_row(1, to_np(dri_img), "Dri GT Img")
    plot_row(2, to_np(img_warped), "Warped Img")
    plot_row(3, to_np(dri_lbl), "Dri GT Seg", cmap="tab10", vmin=0, vmax=3)
    plot_row(4, to_np(seg_warped), "Warped Seg", cmap="tab10", vmin=0, vmax=3)

    def plot_nrj_row(row_idx, vol, title, cmap="gray", vmin=None, vmax=None):
        axes[row_idx, 0].set_ylabel(title, fontsize=10)
        axes[row_idx, 0].imshow(np.rot90(vol[12, :, :]), cmap=cmap, vmin=vmin, vmax=vmax)
        axes[row_idx, 1].imshow(np.rot90(vol[:, 12, :]), cmap=cmap, vmin=vmin, vmax=vmax)
        axes[row_idx, 2].imshow(np.rot90(vol[:, :, 8]), cmap=cmap, vmin=vmin, vmax=vmax)
        for ax in axes[row_idx]:
            ax.axis("off")

    motion_energy = torch.mean(torch.abs(feature_motion), dim=1, keepdim=True)
    plot_nrj_row(5, to_np(motion_energy), "Motion Energy (R)", cmap="magma")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"vis_step_{step:06d}.png"))
    plt.close()

    raw_loader = Compose([
        LoadImaged(keys=["image"], image_only=False, ensure_channel_first=True),
        Spacingd(keys=["image"], pixdim=(1.0, 1.0, -1.0), mode="bilinear"),
    ])
    temp_data = raw_loader({"image": dri_image_path})
    true_shape = temp_data["image"].shape[1:]
    true_affine = temp_data["image"].meta["affine"].numpy()
    padded_shape = img_warped.shape[2:]
    slices = []
    for t_dim, p_dim in zip(true_shape, padded_shape):
        if t_dim >= p_dim:
            slices.append(slice(0, p_dim))
        else:
            diff = p_dim - t_dim
            start = diff // 2
            slices.append(slice(start, start + t_dim))
    crop_slices = (..., slices[0], slices[1], slices[2])
    img_warped_cropped = img_warped[crop_slices]
    seg_warped_cropped = seg_warped[crop_slices]
    nib.save(nib.Nifti1Image(to_np(img_warped_cropped), true_affine), os.path.join(output_dir, f"warped_img_{step:06d}.nii.gz"))
    nib.save(nib.Nifti1Image(to_np(seg_warped_cropped), true_affine), os.path.join(output_dir, f"warped_seg_{step:06d}.nii.gz"))
    print(f"  -> Saved fixed visualization (Cropped to {true_shape}) to {output_dir}")
