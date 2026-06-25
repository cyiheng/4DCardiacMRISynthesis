import os
import glob
import random
import re
import pandas as pd
import numpy as np
import torch
from monai.data import CacheDataset, list_data_collate, MetaTensor
from monai.transforms import (
    Compose,
    LoadImaged,
    Spacingd,
    ScaleIntensityRangePercentilesd,
    EnsureTyped,
    EnsureChannelFirstd,
    SpatialPadd,
    RandSpatialCropd,
    CenterSpatialCropd,
)
from monai.transforms import MapTransform
import logging

logger = logging.getLogger(__name__)

class SafeCompose(Compose):
    def __call__(self, input_, start=0, end=None, **kwargs):
        if end is None:
            end = len(self.transforms)
        for _transform in self.transforms[start:end]:
            if input_ is None:
                return None
            input_ = _transform(input_)
        return input_

class HandleMissingLabeld(MapTransform):
    def __init__(self, keys, image_key="image"):
        super().__init__(keys)
        self.image_key = image_key

    def __call__(self, data):
        d = dict(data)
        if "label" not in d or isinstance(d.get("label"), str):
            img = d[self.image_key]
            img_shape = img.shape[1:]
            zero_tensor = torch.zeros((1,) + img_shape, dtype=torch.float32)
            if isinstance(img, MetaTensor):
                meta_dict = dict(img.meta)
                if "affine" in meta_dict:
                    del meta_dict["affine"]
                d["label"] = MetaTensor(zero_tensor, affine=img.affine, meta=meta_dict)
            else:
                d["label"] = zero_tensor
            d["has_label"] = torch.tensor(0.0, dtype=torch.float32)
        else:
            d["has_label"] = torch.tensor(1.0, dtype=torch.float32)
        return d

def collate_fn_ignore_none(batch):
    batch = [x for x in batch if x is not None]
    if len(batch) == 0:
        return None
    return list_data_collate(batch)

def parse_metadata_rules(csv_path, dataset_type, exclude_set=None):
    df = pd.read_csv(csv_path)
    valid_pids = set()
    labeled_target_filenames = set()
    for _, row in df.iterrows():
        pid_raw = str(row['pid'])
        if dataset_type == "ACDC":
            prefix = f"ACDC_{pid_raw}"
            ed_frame = int(row.get('ed_frame', 1)) - 1
            es_frame = int(row.get('es_frame', ed_frame + 1)) - 1
        else:
            prefix = f"DSB_{pid_raw}"
            if exclude_set is not None and prefix in exclude_set:
                continue
            if int(row.get('n_slices', 0)) <= 4:
                continue
            ed_frame = 0
            try:
                es_frame = int(row.get('es_frame', 9)) - 1
            except Exception:
                es_frame = 8
        valid_pids.add(prefix)
        fname_ed = f"{prefix}_frame_{ed_frame:03d}.nii.gz"
        fname_es = f"{prefix}_frame_{es_frame:03d}.nii.gz"
        labeled_target_filenames.add(fname_ed)
        labeled_target_filenames.add(fname_es)
    return valid_pids, labeled_target_filenames

def build_transforms(patch_size, num_classes):
    train_transforms = SafeCompose([
        LoadImaged(keys=["image", "label"], image_only=False, allow_missing_keys=True),
        EnsureChannelFirstd(keys=["image", "label"], allow_missing_keys=True),
        HandleMissingLabeld(keys=["label"], image_key="image"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, -1.0), mode=("bilinear", "nearest")),
        SpatialPadd(keys=["image", "label"], spatial_size=patch_size, method="symmetric", mode="edge"),
        RandSpatialCropd(keys=["image", "label"], roi_size=patch_size, random_center=True, random_size=False),
        ScaleIntensityRangePercentilesd(keys=["image"], lower=0.5, upper=99.5, b_min=-1.0, b_max=1.0, clip=True),
        EnsureTyped(keys=["image", "label", "has_label"], dtype=torch.float32),
    ])

    val_transforms = SafeCompose([
        LoadImaged(keys=["image", "label"], image_only=False, allow_missing_keys=True),
        EnsureChannelFirstd(keys=["image", "label"], allow_missing_keys=True),
        HandleMissingLabeld(keys=["label"], image_key="image"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, -1.0), mode=("bilinear", "nearest")),
        SpatialPadd(keys=["image", "label"], spatial_size=patch_size, method="symmetric", mode="edge"),
        CenterSpatialCropd(keys=["image", "label"], roi_size=patch_size),
        ScaleIntensityRangePercentilesd(keys=["image"], lower=0.5, upper=99.5, b_min=-1.0, b_max=1.0, clip=True),
        EnsureTyped(keys=["image", "label", "has_label"], dtype=torch.float32),
    ])

    return train_transforms, val_transforms

def parse_combined_csv(csv_path):
    """
    Parses the consolidated CSV to identify valid patient prefixes (PIDs) 
    and expected labeled filenames.
    """
    if csv_path is None or not os.path.exists(csv_path):
        return set(), set()
        
    df = pd.read_csv(csv_path)
    valid_pids = set()
    labeled_target_filenames = set()
    
    for _, row in df.iterrows():
        dataset = row['dataset']
        pid_raw = str(row['pid'])
        
        # Build prefix to match files: e.g., ACDC_patient101 or DSB_1
        prefix = f"{dataset}_{pid_raw}" if dataset == "ACDC" else f"DSB_{pid_raw}"
        
        # Apply slice filter to DSB cases
        if dataset == "DSB" and int(row["n_slices"]) <= 4:
            continue
            
        valid_pids.add(prefix)
        
        # Determine 0-based ED and ES frames
        if dataset == "ACDC":
            ed_frame = int(row['ed_frame']) - 1
            es_frame = int(row['es_frame']) - 1
        else:  # DSB
            ed_frame = 0
            try:
                es_frame = int(row['es_frame']) - 1
            except (ValueError, TypeError):
                es_frame = 8 # Default fallback if ES calculation failed
                
        fname_ed = f"{prefix}_frame_{ed_frame:03d}.nii.gz"
        fname_es = f"{prefix}_frame_{es_frame:03d}.nii.gz"
        
        labeled_target_filenames.add(fname_ed)
        labeled_target_filenames.add(fname_es)
        
    return valid_pids, labeled_target_filenames

def build_dataloaders(
    data_dir='data',
    label_dir=None,
    patch_size=(192,192,16),
    batch_size=1,
    num_workers=4,
    train_csv=None,  # Consolidated train split
    test_csv=None,   # Consolidated validation/test split
):
    if label_dir is None:
        label_dir = os.path.join(data_dir, 'labels')

    # Load parameters and labels from precomputed CSV files
    train_valid_pids, train_target_labeled_names = parse_combined_csv(train_csv)
    val_valid_pids, _ = parse_combined_csv(test_csv)

    # Scan files in directory
    all_nii_files = sorted(glob.glob(os.path.join(data_dir, "*.nii.gz")))

    train_files_all = []
    train_files_labeled = []
    val_files_acdc = []
    val_files_dsb = []

    pid_pattern = re.compile(r"^(ACDC_patient\d+|DSB_\d+)")

    for img_path in all_nii_files:
        fname = os.path.basename(img_path)
        match = pid_pattern.match(fname)
        if not match:
            continue
        pid_str = match.group(1)
        
        entry = {"image": img_path}
        lbl_path = os.path.join(label_dir, fname)
        has_label = os.path.exists(lbl_path)
        if has_label:
            entry['label'] = lbl_path
            
        if pid_str in train_valid_pids:
            train_files_all.append(entry)
            if fname in train_target_labeled_names and has_label:
                train_files_labeled.append(entry)
        elif pid_str in val_valid_pids:
            if 'ACDC' in pid_str:
                val_files_acdc.append(entry)
            else:
                val_files_dsb.append(entry)

    random.shuffle(val_files_dsb)
    dsb_subset_size = max(0, int(len(val_files_dsb) * 0.01))
    val_files_dsb = val_files_dsb[:dsb_subset_size]
    
    # Combined validation dataset
    val_files = val_files_acdc[:10] + val_files_dsb

    # Build transforms (Assuming build_transforms and collate_fn_ignore_none are defined in your module)
    train_transforms, val_transforms = build_transforms(patch_size, num_classes=4)

    # Construct datasets using MONAI CacheDataset
    train_ds_all = CacheDataset(data=train_files_all, transform=train_transforms, cache_rate=0.25, num_workers=num_workers)
    train_ds_labeled = CacheDataset(data=train_files_labeled, transform=train_transforms, cache_rate=0.25, num_workers=num_workers)
    val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=0.25, num_workers=num_workers)

    train_loader_all = torch.utils.data.DataLoader(
        train_ds_all, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True, collate_fn=collate_fn_ignore_none
    )
    train_loader_labeled = torch.utils.data.DataLoader(
        train_ds_labeled, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True, collate_fn=collate_fn_ignore_none
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=1, shuffle=False, 
        num_workers=0, pin_memory=False, collate_fn=collate_fn_ignore_none
    )

    return train_loader_all, train_loader_labeled, val_loader


