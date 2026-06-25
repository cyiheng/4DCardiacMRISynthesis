import os
import glob
import re
import random
from tqdm import tqdm
import pandas as pd
import numpy as np
import nibabel as nib
import torch

from monai.transforms import (
    Compose, LoadImaged, Spacingd, ScaleIntensityRangePercentilesd,
    EnsureTyped, SpatialPadd, CenterSpatialCropd, RandAffined
)
from models.custom_AEKL import DualDecoderVAE 
from utils.common import get_patient_id_from_filename, get_lv_volume_from_logits, get_frame_number

# ==============================================================================
# --- 1. CONFIGURATION ---
# ==============================================================================
MODEL_PATH = "results/001_VAE3D_dualdecoder/checkpoints/best_autoencoder_model.pth"
data_dir = 'data'
DATA_DIR = os.path.join(data_dir, 'images')
OUTPUT_DIR = os.path.join(data_dir, 'latents_aug')
TRAIN_DIR = os.path.join(OUTPUT_DIR, "train")
TEST_DIR = os.path.join(OUTPUT_DIR, "test")
DEBUG_DIR = os.path.join(OUTPUT_DIR, "debug")

# Clean Consolidated CSV Paths (Bad cases already excluded)
TRAIN_CSV= os.path.join(data_dir, 'train_split_final.csv')
TEST_CSV = os.path.join(data_dir, 'test_split_final.csv')

# Params
LATENT_DIM = 8
NUM_CLASSES = 4
TARGET_SPATIAL_SIZE = (192, 192, 16) 
SPACING = (1.0, 1.0, 10.0) 
AUGMENTATIONS_PER_IMAGE = 5 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ==============================================================================
# --- 2. HELPERS ---
# ==============================================================================

def normalize_metadata(row, pid_prefix):
    try:
        # Check 'edv' / 'diastole_volume' safely
        edv = row.get('edv')
        if pd.isna(edv) or edv == '':
            edv = row.get('diastole_volume')
            
        # Check 'esv' / 'systole_volume' safely
        esv = row.get('esv')
        if pd.isna(esv) or esv == '':
            esv = row.get('systole_volume')
            
        # Convert to float safely
        edv = float(edv) if pd.notna(edv) else -1.0
        esv = float(esv) if pd.notna(esv) else -1.0
        
        # Calculate EF using the valid numerical values
        if edv > 0 and esv >= 0:
            ef = (edv - esv) / edv * 100.0
        else:
            ef = -1.0
            
        pathology = row.get('pathology', 'Unknown')
        if pd.isna(pathology) or str(pathology).strip() == "":
            pathology = 'Unknown'
                
        return {
            "pid": pid_prefix,
            "diagnosis": pathology,
            "edv": edv, "esv": esv, "ef": ef,
            "n_slices": float(row.get('n_slices', 10.0))
        }
    except Exception as e:
        return None

def load_and_split_metadata():
    print("--- Parsing Metadata & creating Split (Combined CSV) ---")
    meta_dict = {}
    
    # Helper to parse combined CSVs
    def process_csv(csv_path):
        if not os.path.exists(csv_path): return set()
        df = pd.read_csv(csv_path)
        pids = set()
        for _, row in df.iterrows():
            dataset = row['dataset']
            pid_raw = str(row['pid'])
            prefix = f"{dataset}_{pid_raw}" if dataset == "ACDC" else f"DSB_{pid_raw}"
            
            # Filter DSB slice count
            n_slices = float(row.get('n_slices', 10.0))
            if dataset == "DSB" and n_slices <= 4:
                continue
            
            data = normalize_metadata(row, prefix)
            if data and data['edv'] > 0:
                meta_dict[prefix] = data
                pids.add(prefix)
        return pids

    # Load clean final CSV splits directly (No bad case filtering needed)
    train_pids = process_csv(TRAIN_CSV)  
    val_pids = process_csv(TEST_CSV)    

    print(f"Total Metadata Loaded: {len(meta_dict)}")
    print(f"Final Train Count: {len(train_pids)}")
    print(f"Final Test/Val Count: {len(val_pids)}")
    
    return meta_dict, train_pids, val_pids

# ==============================================================================
# --- 3. MAIN EXECUTION ---
# ==============================================================================
def main():
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)
    os.makedirs(DEBUG_DIR, exist_ok=True)
    
    print(f"Loading VAE from {MODEL_PATH}...")
    vae = DualDecoderVAE(latent_dim=LATENT_DIM, num_classes=NUM_CLASSES).to(DEVICE)
    vae.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    vae.eval()

    meta_map, train_pids, test_pids = load_and_split_metadata()

    base_transforms = Compose([
        LoadImaged(keys=["image"], image_only=False, ensure_channel_first=True),
        Spacingd(keys=["image"], pixdim=SPACING, mode="bilinear"),
        SpatialPadd(keys=["image"], spatial_size=TARGET_SPATIAL_SIZE, method="symmetric", mode="edge"),
        CenterSpatialCropd(keys=["image"], roi_size=TARGET_SPATIAL_SIZE),
        ScaleIntensityRangePercentilesd(keys=["image"], lower=0.5, upper=99.5, b_min=-1.0, b_max=1.0, clip=True),
        EnsureTyped(keys=["image"], dtype=torch.float32),
    ])

    aug_transforms = Compose([
        RandAffined(
            keys=["image"],
            prob=1.0, 
            rotate_range=None, 
            shear_range=None,
            translate_range=None,
            scale_range=(0.09, 0.09, 0.05), # Scale +/- 9% in X, Y and 5% in Z
            mode="bilinear",
            padding_mode="border" 
        )
    ])

    raw_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.nii.gz")))[:100]
    all_files = []
    for f in raw_files:
        if re.search(r"frame_?0*[0-2]\.nii\.gz$", f):
            all_files.append(f)
            
    print(f"Filtered to {len(all_files)} files (Frames 0, 1, 2 only).")

    patient_frame0_vae_vol = {}
    latents_for_scaling = []
    new_train_rows = []
    new_test_rows = []

    for fpath in tqdm(all_files):
        fname = os.path.basename(fpath)
        pid = get_patient_id_from_filename(fname)
        frame_idx = get_frame_number(fname)
        
        if pid not in meta_map: continue
        is_train = pid in train_pids
        is_test = pid in test_pids
        if not is_train and not is_test: continue 

        try:
            data = base_transforms({"image": fpath})
        except: continue
        img_tensor = data["image"].unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            rec_img, rec_seg, mu, _ = vae(img_tensor)
            
            save_dir = TRAIN_DIR if is_train else TEST_DIR
            save_name = fname.replace(".nii.gz", ".pt")
            torch.save(mu.squeeze(0).cpu(), os.path.join(save_dir, save_name))
            
            orig_meta = meta_map[pid]
            vol_vae_current = get_lv_volume_from_logits(rec_seg)
            if vol_vae_current == 0: vol_vae_current = 0.001

            current_edv = orig_meta['edv']
            current_esv = orig_meta['esv']

            if frame_idx == 0:
                patient_frame0_vae_vol[pid] = vol_vae_current
            elif frame_idx > 0:
                if pid in patient_frame0_vae_vol:
                    baseline = patient_frame0_vae_vol[pid]
                    if baseline > 0:
                        ratio = vol_vae_current / baseline
                        current_edv = orig_meta['edv'] * ratio
                        current_esv = orig_meta['esv'] * ratio

            current_ef = (current_edv - current_esv) / current_edv * 100.0 if current_edv > 0 else 0

            row_data = {
                "filename": save_name, "pid": pid, "diagnosis": orig_meta['diagnosis'],
                "edv": current_edv, "esv": current_esv, "ef": current_ef,
                "n_slices": orig_meta['n_slices'], "is_augmented": False,
                "frame_idx": frame_idx
            }
            
            if is_train:
                new_train_rows.append(row_data)
                latents_for_scaling.append(mu.squeeze(0).cpu())
            else:
                new_test_rows.append(row_data)

        if is_train:
            for i in range(AUGMENTATIONS_PER_IMAGE):
                aug_input_dict = {"image": img_tensor.squeeze(0)} 
                aug_out_dict = aug_transforms(aug_input_dict)
                aug_tensor = aug_out_dict["image"].unsqueeze(0).to(DEVICE)
                
                with torch.no_grad():
                    rec_img_aug, rec_seg_aug, mu_aug, _ = vae(aug_tensor)
                    vol_vae_aug = get_lv_volume_from_logits(rec_seg_aug)
                    
                    if vol_vae_current == 0: vol_vae_current = 0.001
                    ratio = vol_vae_aug / vol_vae_current
                    if ratio < 0.5 or ratio > 2.0: continue
                    
                    new_edv = current_edv * ratio
                    new_esv = current_esv * ratio
                    new_ef = (new_edv - new_esv) / new_edv * 100.0 if new_edv > 0 else 0
                    
                    aug_name = fname.replace(".nii.gz", f"_aug{i}.pt")
                    torch.save(mu_aug.squeeze(0).cpu(), os.path.join(TRAIN_DIR, aug_name))
                    
                    aug_row = {
                        "filename": aug_name, "pid": pid, "diagnosis": orig_meta['diagnosis'],
                        "edv": new_edv, "esv": new_esv, "ef": new_ef,
                        "n_slices": orig_meta['n_slices'], "is_augmented": True,
                        "frame_idx": frame_idx
                    }
                    new_train_rows.append(aug_row)
                    latents_for_scaling.append(mu_aug.squeeze(0).cpu())

                    if i == 0: 
                        img_np = rec_img_aug.squeeze().cpu().numpy()
                        seg_np = torch.argmax(rec_seg_aug, dim=1).squeeze().cpu().numpy().astype(np.float32)
                        
                        affine = np.eye(4); affine[2,2] = 10.0 
                        nib.save(nib.Nifti1Image(img_np, affine), 
                                 os.path.join(DEBUG_DIR, aug_name.replace(".pt", "_img.nii.gz")))
                        nib.save(nib.Nifti1Image(seg_np, affine), 
                                 os.path.join(DEBUG_DIR, aug_name.replace(".pt", "_seg.nii.gz")))

    if len(latents_for_scaling) > 0:
        full = torch.stack(latents_for_scaling)
        mean, std = torch.mean(full), torch.std(full)
        print(f"\nStats - Mean: {mean:.4f}, Std: {std:.4f}, Scale Factor: {1.0/std:.6f}")
        with open(os.path.join(OUTPUT_DIR, "scaling_stats.txt"), "w") as f:
            f.write(f"Mean: {mean}\nStd: {std}\nScale: {1.0/std}\n")

    pd.DataFrame(new_train_rows).to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False)
    pd.DataFrame(new_test_rows).to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False)
    print("Done!")

if __name__ == "__main__":
    main()