import torch
import torch.nn as nn


class UNetBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class LatentFeatureMotionPredictor(nn.Module):
    def __init__(self, in_channels, latent_channels=8, base_features=32):
        super().__init__()
        self.down1 = UNetBlock3D(in_channels, base_features)
        self.down2 = UNetBlock3D(base_features, base_features * 2)
        self.bottleneck = UNetBlock3D(base_features * 2, base_features * 4)

        self.up1 = UNetBlock3D(base_features * 4 + base_features * 2, base_features * 2)
        self.up2 = UNetBlock3D(base_features * 2 + base_features, base_features)

        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.upsample = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False)
        self.motion_head = nn.Conv3d(base_features, latent_channels, kernel_size=3, padding=1)

    def forward(self, z_ref, z_dri):
        x = torch.cat([z_ref, z_dri], dim=1)
        d1 = self.down1(x)
        p1 = self.pool(d1)
        d2 = self.down2(p1)
        p2 = self.pool(d2)
        b = self.bottleneck(p2)
        u1 = self.up1(torch.cat([self.upsample(b), d2], dim=1))
        u2 = self.up2(torch.cat([self.upsample(u1), d1], dim=1))
        return self.motion_head(u2)
