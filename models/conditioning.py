import math
import torch
import torch.nn as nn


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.max_period) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class ClinicalScalarEmbedding(nn.Module):
    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.max_period) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class ClinicalConditioningModel(nn.Module):
    def __init__(self, embedding_dim=512, num_diagnosis=7, num_scalars=4):
        super().__init__()
        self.diag_embed = nn.Embedding(num_diagnosis, embedding_dim)
        self.scalar_proj = nn.Sequential(
            nn.Linear(num_scalars, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, diag_ids, scalars):
        d_emb = self.diag_embed(diag_ids)
        s_emb = self.scalar_proj(scalars)
        tokens = torch.cat([d_emb, s_emb], dim=1)
        tokens = self.mlp(tokens)
        return tokens.unsqueeze(1)
