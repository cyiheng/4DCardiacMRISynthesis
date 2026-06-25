import numpy as np
import torch
import nibabel as nib
import os
import re

def get_patient_id_from_filename(filepath):
    basename = os.path.basename(filepath)
    match = re.match(r"^(ACDC_patient\d+|DSB_\d+)", basename)
    if match:
        return match.group(1)
    return None

def get_lv_volume_from_logits(seg_logits, voxel_vol=0.01):
    seg_pred = torch.argmax(seg_logits, dim=1) 
    lv_voxels = torch.sum(seg_pred == 3).item()
    return lv_voxels * voxel_vol

def get_frame_number(filepath):
    match = re.search(r"frame_?0*([0-9]+)\.nii\.gz$", filepath)
    if match: return int(match.group(1))
    return -1

def normalize_scalars(ef, edv, esv, n_slices, max_edv=600.0, max_esv=600.0, max_slices=20.0):
    ef_n = max(0.0, min(1.0, ef / 100.0))
    edv_n = max(0.0, min(1.0, edv / max_edv))
    esv_n = max(0.0, min(1.0, esv / max_esv))
    slices_n = max(0.0, min(1.0, n_slices / max_slices))
    return [ef_n, edv_n, esv_n, slices_n]


def save_nifti(tensor, affine, filename):
    data = tensor.squeeze().cpu().numpy()
    img = nib.Nifti1Image(data, affine)
    nib.save(img, filename)
    print(f"Saved: {filename}")


def save_nifti_4d(frame_list, affine, filename):
    vol_4d = torch.stack(frame_list, dim=-1).cpu().numpy() if isinstance(frame_list[0], torch.Tensor) else np.stack(frame_list, axis=-1)
    img = nib.Nifti1Image(vol_4d, affine)
    nib.save(img, filename)
    print(f"Saved 4D volume: {filename} | Shape: {vol_4d.shape}")


def encode_image(vae, img_tensor):
    with torch.no_grad():
        _, _, z_mu, _ = vae(img_tensor)
    return z_mu
