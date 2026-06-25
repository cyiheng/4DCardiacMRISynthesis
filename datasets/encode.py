import os
import glob
import pandas as pd
import torch
from torch.utils.data import DataLoader
import monai
from monai.transforms import (
    Compose,
    LoadImaged,
    Spacingd,
    ScaleIntensityRangePercentilesd,
    EnsureTyped,
    SpatialPadd,
    CenterSpatialCropd,
)
from tqdm import tqdm

# --- IMPORT YOUR CUSTOM MODULES ---
from models.custom_AEKL import DualDecoderVAE 
from dataset_utils import collate_fn_ignore_none, get_patient_id_from_filename

# --- 1. CONFIGURATION ---
MODEL_PATH = "./results/001_VAE3D_dualdecoder/checkpoints/best_autoencoder_model.pth" # Point to your best model
data_dir = 'data'  # All inputs and CSVs are subfolders under `data/`
DATA_DIR = os.path.join(data_dir, 'images')
OUTPUT_DIR = os.path.join(data_dir, 'latents')

# Consolidated Split Path (No more manual bad case or split parsing needed here)
TRAIN_CSV_PATH = os.path.join(data_dir, 'train_split_final.csv')

# Configuration
GET_SCALING = True 
LATENT_DIM = 8
NUM_CLASSES = 4
TARGET_SPATIAL_SIZE = (192, 192, 16) 
BATCH_SIZE = 1 
NUM_WORKERS = 4

def get_train_pids():
    """
    Reads the consolidated training CSV to identify patient IDs 
    assigned to the training set.
    """
    print(f"Parsing consolidated training CSV: {TRAIN_CSV_PATH}...")
    train_pids = set()
    
    if not os.path.exists(TRAIN_CSV_PATH):
        print(f"Error: {TRAIN_CSV_PATH} was not found.")
        return train_pids

    df = pd.read_csv(TRAIN_CSV_PATH)
    for _, row in df.iterrows():
        dataset = row['dataset']
        pid_raw = str(row['pid'])
        
        # Consistent prefix naming logic matching dataset_utils / split
        prefix = f"{dataset}_{pid_raw}" if dataset == "ACDC" else f"DSB_{pid_raw}"
        
        # Double check DSB slice count filter
        if dataset == "DSB" and int(row['n_slices']) <= 4:
            continue
            
        train_pids.add(prefix)

    print(f"  - Identified {len(train_pids)} valid Training Patients.")
    return train_pids

# --- 2. MAIN SCRIPT LOGIC ---
torch.manual_seed(42)

def main():
    print("--- Starting Dataset Pre-encoding Script ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_pids_set = get_train_pids()

    # 1. Load Model
    print(f"Loading DualDecoderVAE from {MODEL_PATH}...")
    autoencoder = DualDecoderVAE(latent_dim=LATENT_DIM, num_classes=NUM_CLASSES).to(device)
    
    try:
        autoencoder.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except RuntimeError as e:
        print("\nERROR: Model size mismatch. Did you update custom_AEKL.py to use 4 layers and the correct strides?")
        raise e
        
    autoencoder.eval()

    # 2. Transforms (Exact same pre-processing as Training)
    transforms = Compose([
        LoadImaged(keys=["image"], image_only=False, ensure_channel_first=True),
        Spacingd(keys=["image"], pixdim=(1.0, 1.0, -1.0), mode="bilinear"),
        # Deterministic padding/cropping to target size
        SpatialPadd(keys=["image"], spatial_size=TARGET_SPATIAL_SIZE, method="symmetric", mode="edge"),
        CenterSpatialCropd(keys=["image"], roi_size=TARGET_SPATIAL_SIZE),
        ScaleIntensityRangePercentilesd(keys=["image"], lower=0.5, upper=99.5, b_min=-1.0, b_max=1.0, clip=True),
        EnsureTyped(keys=["image"], dtype=torch.float32),
    ])

    # 3. DataLoader
    all_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.nii.gz")))[:]
    print(f"Found {len(all_files)} files on disk.")

    file_dicts = [{"image": f, "filepath": f} for f in all_files]
    ds = monai.data.Dataset(data=file_dicts, transform=transforms)
    
    loader = DataLoader(
        ds, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS, 
        collate_fn=collate_fn_ignore_none
    )

    # 4. Encoding Loop
    train_latents_for_scaling = []
    
    print(f"Starting Encoding... Output: {OUTPUT_DIR}")
    with torch.no_grad():
        for batch in tqdm(loader, desc="Encoding"):
            if batch is None: continue
            
            images = batch["image"].to(device)
            paths = batch["filepath"] 
            if isinstance(paths, str): paths = [paths]

            # VAE Forward -> Get Mean (Mu)
            _, _, mu, _ = autoencoder(images)
            
            z_encoded = mu 
            z_cpu = z_encoded.cpu()
            
            for i in range(len(z_cpu)):
                tensor_data = z_cpu[i] # Shape: (8, 24, 24, 16)
                original_path = paths[i]
                original_name = os.path.basename(original_path)
                save_name = original_name.replace(".nii.gz", ".pt")
                
                # Save latent tensor to disk
                torch.save(tensor_data, os.path.join(OUTPUT_DIR, save_name))
                
                # Collect stats for the train set
                if GET_SCALING:
                    pid = get_patient_id_from_filename(original_name)
                    if pid in train_pids_set:
                        train_latents_for_scaling.append(tensor_data.clone())

    # 5. Calculate Scaling Factor
    if GET_SCALING and len(train_latents_for_scaling) > 0:
        print(f"\n--- Calculating Statistics on {len(train_latents_for_scaling)} Training Volumes ---")
        
        # Stack all tensors (Memory safe for 15k items at 24x24x16 size)
        full_tensor = torch.stack(train_latents_for_scaling) 
        
        global_mean = torch.mean(full_tensor)
        global_std = torch.std(full_tensor)
        
        # Multiply latents by this scale_factor during LFM/LDM training to get standard deviation ~ 1.0
        scale_factor = 1.0 / global_std.item()
        
        print(f"Global Mean: {global_mean:.6f}")
        print(f"Global Std:  {global_std:.6f}")
        print("-" * 30)
        print(f"RECOMMENDED SCALING FACTOR: {scale_factor:.6f}")
        print("-" * 30)
        
        # Save stats to a text file for reference
        with open(os.path.join(OUTPUT_DIR, "scaling_factor.txt"), "w") as f:
            f.write(f"Mean: {global_mean:.6f}\n")
            f.write(f"Std: {global_std:.6f}\n")
            f.write(f"Scale Factor: {scale_factor:.6f}\n")
            
    print("\n--- Encoding Complete ---")

if __name__ == "__main__":
    main()