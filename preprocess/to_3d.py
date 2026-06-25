import os
import csv
import nibabel as nib
import numpy as np
from tqdm import tqdm

def load_acdc_metadata(metadata_path):
    """
    Parses the ACDC metadata CSV file to map patient IDs to ED and ES frame indices.
    Returns a dict: {'patient001': {'ed': 0, 'es': 11}, ...}
    """
    meta_dict = {}
    if not os.path.exists(metadata_path):
        print(f"  [Warning] Metadata file not found: {metadata_path}")
        return meta_dict

    with open(metadata_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row['pid']
            # Convert string to int and adjust to 0-based indexing
            try:
                ed_frame = int(row['ed_frame']) - 1
                es_frame = int(row['es_frame']) - 1
                meta_dict[pid] = {'ed': ed_frame, 'es': es_frame}
            except ValueError:
                continue # Skip rows with invalid data
    return meta_dict

def unpack_4d_to_3d_nifti(nifti_path, dataset_name, patient_id, output_dir_img, output_dir_lbl, acdc_indices=None):
    """
    Loads a 4D NIfTI file and saves 3D frames. 
    If acdc_indices is provided, saves the corresponding GT segmentation for ED/ES frames.
    """
    try:
        # Load the 4D NIfTI file
        nifti_4d_img = nib.load(nifti_path)
        data_4d = nifti_4d_img.get_fdata()
        
        # Pre-load GT files if this is an ACDC patient
        img_ed_gt, img_es_gt = None, None
        if dataset_name == 'ACDC' and acdc_indices:
            parent_dir = os.path.dirname(nifti_path)
            gt_ed_path = os.path.join(parent_dir, f"{patient_id}_sax_ed_gt.nii.gz")
            gt_es_path = os.path.join(parent_dir, f"{patient_id}_sax_es_gt.nii.gz")
            
            if os.path.exists(gt_ed_path): img_ed_gt = nib.load(gt_ed_path)
            if os.path.exists(gt_es_path): img_es_gt = nib.load(gt_es_path)

        if len(data_4d.shape) != 4:
            return

        _, _, _, num_frames = data_4d.shape

        for t in range(num_frames):
            # --- 1. Save Image ---
            volume_3d_data = data_4d[:, :, :, t]
            nifti_3d_img = nib.Nifti1Image(volume_3d_data, nifti_4d_img.affine, nifti_4d_img.header)
            
            # Filename: dataset_patient_frame_XXX.nii.gz
            output_filename = f"{dataset_name}_{patient_id}_frame_{t:03d}.nii.gz"
            output_path_img = os.path.join(output_dir_img, output_filename)
            nib.save(nifti_3d_img, output_path_img)

            # --- 2. Save Label (ACDC Only) ---
            if dataset_name == 'ACDC' and acdc_indices:
                gt_to_save = None
                
                # Check if current frame matches the ED or ES index from CSV
                if t == acdc_indices['ed'] and img_ed_gt is not None:
                    gt_to_save = img_ed_gt
                elif t == acdc_indices['es'] and img_es_gt is not None:
                    gt_to_save = img_es_gt
                
                if gt_to_save is not None:
                    # Save to label dir with SAME NAME as image
                    output_path_lbl = os.path.join(output_dir_lbl, output_filename)
                    nib.save(gt_to_save, output_path_lbl)

    except Exception as e:
        print(f"  [Error] Failed to process {nifti_path}: {e}")

def process_datasets(base_data_dir, output_dir_img, output_dir_lbl):
    os.makedirs(output_dir_img, exist_ok=True)
    os.makedirs(output_dir_lbl, exist_ok=True)
    
    datasets_info = {
        'ACDC': 'ACDC_Preprocessed',
        'DSB': 'DSB_nifti'
    }

    for dataset_name, dataset_folder in datasets_info.items():
        dataset_path = os.path.join(base_data_dir, dataset_folder)
        if not os.path.isdir(dataset_path): continue

        print(f"\nProcessing dataset: {dataset_name}")

        for split in ['train', 'test', 'val']:
            split_path = os.path.join(dataset_path, split)
            if not os.path.isdir(split_path): continue

            # --- Load Metadata for ACDC ---
            current_metadata = {}
            if dataset_name == 'ACDC':
                # Assumes metadata csv is in the dataset folder, e.g., ACDC_Preprocessed/train_metadata.csv
                csv_path = os.path.join(dataset_path, f"{split}_metadata.csv")
                current_metadata = load_acdc_metadata(csv_path)

            patient_dirs = sorted([p for p in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, p))])
            
            for patient_id in tqdm(patient_dirs, desc=f'  -> {split.capitalize()}'):
                patient_path = os.path.join(split_path, patient_id)
                
                # Get indices for this patient if ACDC
                indices = current_metadata.get(patient_id) if dataset_name == 'ACDC' else None

                target_file = None
                for filename in os.listdir(patient_path):
                    if 'sax_t.nii.gz' in filename:
                        target_file = filename
                        break

                if target_file:
                    nifti_path = os.path.join(patient_path, target_file)
                    unpack_4d_to_3d_nifti(
                        nifti_path, dataset_name, patient_id, 
                        output_dir_img, output_dir_lbl, 
                        acdc_indices=indices
                    )

if __name__ == "__main__":
    BASE_DATA_FOLDER = 'path/to/data'
    OUTPUT_IMG_FOLDER = './data/images'
    OUTPUT_LBL_FOLDER = './data/labels'

    process_datasets(BASE_DATA_FOLDER, OUTPUT_IMG_FOLDER, OUTPUT_LBL_FOLDER)
    print("\nPreprocessing complete!")