import os
import glob
import re
import random
from collections import defaultdict
import pandas as pd

def parse_combined_metadata(csv_path):
    """
    Parses consolidated CSV to identify valid patients.
    Bad cases are assumed to already be filtered out by `split.py`.
    """
    if not os.path.exists(csv_path):
        print(f"Warning: CSV {csv_path} not found.")
        return set()

    df = pd.read_csv(csv_path)
    valid_pids = set()
    
    for _, row in df.iterrows():
        dataset = row['dataset']
        pid_raw = str(row['pid'])
        
        # Consistent Prefix naming logic matching `split.py` / Data files
        prefix = f"{dataset}_{pid_raw}" if dataset == "ACDC" else f"DSB_{pid_raw}"
        
        # Apply the DSB slice threshold filter
        if dataset == "DSB" and int(row["n_slices"]) <= 4:
            continue
            
        valid_pids.add(prefix)
            
    return valid_pids


def prepare_paired_datalist(cfg):
    print("--- Preparing Paired Latent Datalist (Combined CSV) ---")
    
    # 1. Parse valid PIDs using new combined CSV
    train_valid_pids = parse_combined_metadata(cfg.train_csv)
    val_valid_pids = parse_combined_metadata(cfg.test_csv)
    
    # 2. Scan Latent Files on disk
    all_latent_files = glob.glob(os.path.join(cfg.latent_data_dir, "*.pt"))
    patient_file_map = defaultdict(list)
    pid_pattern = re.compile(r"^(ACDC_patient\d+|DSB_\d+)")
    
    for file_path in all_latent_files:
        basename = os.path.basename(file_path)
        match = pid_pattern.match(basename)
        if match:
            pid = match.group(1)
            patient_file_map[pid].append(file_path)

    # 3. Pair building logic
    def create_pairs_from_pids(valid_pids_set, is_validation=False):
        datalist = []
        sorted_pids = sorted(list(valid_pids_set))
        
        # Subsampling for Validation to match original pipeline ratio
        if is_validation:
            dsb_pids_in_set = [p for p in sorted_pids if "DSB" in p]
            acdc_pids_in_set = [p for p in sorted_pids if "ACDC" in p]
            
            if len(dsb_pids_in_set) > 0:
                random.seed(42)
                random.shuffle(dsb_pids_in_set)
                subset_size = int(len(dsb_pids_in_set) * 0.10)
                dsb_pids_in_set = dsb_pids_in_set[:subset_size]
            
            sorted_pids = sorted(acdc_pids_in_set + dsb_pids_in_set)

        for pid in sorted_pids:
            if pid not in patient_file_map: continue
            
            frames = sorted(patient_file_map[pid])
            
            ref_latent_path = None
            for f in frames:
                if "frame_000" in f or "frame_00.pt" in f or "frame_0.pt" in f:
                     ref_latent_path = f
                     break
            
            if not ref_latent_path and len(frames) > 0:
                ref_latent_path = frames[0]
            
            if not ref_latent_path: continue

            for dri_latent_path in frames:
                if dri_latent_path == ref_latent_path: continue
                
                dri_fname = os.path.basename(dri_latent_path).replace(".pt", ".nii.gz")
                ref_fname = os.path.basename(ref_latent_path).replace(".pt", ".nii.gz")
                
                dri_img_path = os.path.join(cfg.original_image_data_dir, dri_fname)
                ref_img_path = os.path.join(cfg.original_image_data_dir, ref_fname)
                
                dri_lbl_path = os.path.join(cfg.label_data_dir, dri_fname)
                ref_lbl_path = os.path.join(cfg.label_data_dir, ref_fname)
                
                if not os.path.exists(dri_lbl_path): dri_lbl_path = None
                if not os.path.exists(ref_lbl_path): ref_lbl_path = None

                if os.path.exists(dri_img_path) and os.path.exists(ref_img_path):
                    datalist.append({
                        "ref_latent": ref_latent_path, "dri_latent": dri_latent_path,
                        "ref_image": ref_img_path, "dri_image": dri_img_path,
                        "dri_label": dri_lbl_path, "ref_label": ref_lbl_path
                    })
        return datalist
    
    train_pairs = create_pairs_from_pids(train_valid_pids, is_validation=False)
    val_pairs = create_pairs_from_pids(val_valid_pids, is_validation=True)
    
    print(f"Total Train Pairs: {len(train_pairs)}")
    print(f"Total Val Pairs:   {len(val_pairs)}")
    
    return train_pairs, val_pairs
