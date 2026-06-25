import os
import nibabel as nib
import torch
from torch.utils.data import Dataset


class LatentFlowDataset(Dataset):
    def __init__(self, datalist, transforms, vae_scaling):
        self.datalist = datalist
        self.transforms = transforms
        self.vae_scaling = vae_scaling

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        item = self.datalist[idx]
        ref_latent = torch.load(item["ref_latent"], weights_only=False, map_location="cpu") * self.vae_scaling
        dri_latent = torch.load(item["dri_latent"], weights_only=False, map_location="cpu") * self.vae_scaling

        load_dict = {
            k: v
            for k, v in item.items()
            if k in ["dri_image", "ref_image", "dri_label", "ref_label"] and v is not None
        }
        data = self.transforms(load_dict)
        has_dri = 1.0 if "dri_label" in data else 0.0
        has_ref = 1.0 if "ref_label" in data else 0.0

        return {
            "ref_latent": ref_latent,
            "dri_latent": dri_latent,
            "ref_image": data["ref_image"],
            "dri_image": data["dri_image"],
            "dri_label": data.get("dri_label", torch.zeros_like(data["dri_image"])),
            "ref_label": data.get("ref_label", torch.zeros_like(data["ref_image"])),
            "has_dri_label": torch.tensor(has_dri),
            "has_ref_label": torch.tensor(has_ref),
        }


class LatentDataset(Dataset):
    def __init__(self, datalist, max_slices=20.0):
        self.datalist = datalist
        self.max_slices = max_slices

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        item = self.datalist[idx]
        latent = torch.load(item["latent_path"], weights_only=False)
        diag_id = torch.tensor(item["diag_id"], dtype=torch.long)
        scalars = torch.tensor(item["scalars"], dtype=torch.float32)

        _, h, w, d = latent.shape
        mask = torch.zeros((1, h, w, d), dtype=torch.float32)
        n_slices_norm = scalars[3].item()
        raw_n_slices = int(n_slices_norm * self.max_slices)
        valid_slices = min(max(raw_n_slices, 1), d)
        diff = d - valid_slices
        start_idx = diff // 2
        end_idx = start_idx + valid_slices
        mask[..., start_idx:end_idx] = 1.0

        return {
            "latent": latent,
            "diag_id": diag_id,
            "scalars": scalars,
            "mask": mask,
        }


class ConditionedLatentFlowDataset(Dataset):
    def __init__(self, datalist, vae_scaling, spatial_depth=16, max_slices=20.0):
        self.datalist = datalist
        self.vae_scaling = vae_scaling
        self.spatial_depth = spatial_depth
        self.max_slices = max_slices

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        item = self.datalist[idx]
        ref_latent = torch.load(item["ref_latent_path"], weights_only=False, map_location="cpu") * self.vae_scaling
        dri_latent = torch.load(item["dri_latent_path"], weights_only=False, map_location="cpu") * self.vae_scaling

        diag_id = torch.tensor(item["diag_id"], dtype=torch.long)
        scalars = torch.tensor(item["scalars"], dtype=torch.float32)
        time_val_norm = float(item["time_value"])

        _, h, w, d = ref_latent.shape
        mask = torch.zeros((1, h, w, d), dtype=torch.float32)
        true_slices = self._infer_true_slices(item, scalars, d)
        valid_slices = min(max(true_slices, 1), d)
        if valid_slices < d:
            diff = d - valid_slices
            start_idx = diff // 2
            end_idx = start_idx + valid_slices
            mask[..., start_idx:end_idx] = 1.0
        else:
            mask[...] = 1.0

        return {
            "ref_latent": ref_latent,
            "dri_latent": dri_latent,
            "diag_id": diag_id,
            "scalars": scalars,
            "time_val": time_val_norm,
            "mask": mask,
        }

    def _infer_true_slices(self, item, scalars, latent_depth):
        ref_image_path = item.get("ref_image_path")
        if ref_image_path and os.path.exists(ref_image_path):
            try:
                img_obj = nib.load(ref_image_path)
                return img_obj.shape[-1]
            except Exception:
                pass
        return int(scalars[3].item() * self.max_slices)
