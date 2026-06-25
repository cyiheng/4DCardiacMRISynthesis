import numpy as np
import torch
from collections import defaultdict


class AntiCheatMonitor:
    def __init__(self):
        self.metrics = defaultdict(list)

    def check(self, ref_latent, dri_latent, predicted_residual):
        with torch.no_grad():
            b_flat = dri_latent.view(dri_latent.shape[0], -1)
            r_flat = predicted_residual.view(predicted_residual.shape[0], -1)
            sim_r_b = torch.cosine_similarity(r_flat, b_flat).mean()
            mag_r = torch.mean(torch.abs(predicted_residual))
            mag_b = torch.mean(torch.abs(dri_latent))
            ratio = mag_r / (mag_b + 1e-6)
            self.metrics["sim_RB"].append(sim_r_b.item())
            self.metrics["mag_ratio"].append(ratio.item())
            return sim_r_b.item(), ratio.item()

    def log_and_reset(self):
        if not self.metrics["sim_RB"]:
            return {}
        out = {k: np.mean(v) for k, v in self.metrics.items()}
        self.metrics = defaultdict(list)
        return out


def compute_smoothness_loss(flow_map):
    """Calculates Total Variation (TV) smoothness loss for motion flow."""
    diff_h = torch.abs(flow_map[:, :, 1:, :, :] - flow_map[:, :, :-1, :, :])
    diff_w = torch.abs(flow_map[:, :, :, 1:, :] - flow_map[:, :, :, :-1, :])
    diff_d = torch.abs(flow_map[:, :, :, :, 1:] - flow_map[:, :, :, :, :-1])
    return torch.mean(diff_h) + torch.mean(diff_w) + torch.mean(diff_d)
