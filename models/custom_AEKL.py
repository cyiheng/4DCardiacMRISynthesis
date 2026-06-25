from typing import Sequence, List, Union
import torch
import torch.nn as nn
from monai.networks.blocks import Convolution, SpatialAttentionBlock, Upsample
from monai.networks.nets.autoencoderkl import AEKLResBlock

class SplitInput(nn.Module):
    def __init__(self, split_size, return_head=True):
        super().__init__()
        self.split_size = split_size
        self.return_head = return_head # If True return first half, else second half

    def forward(self, x):
        # x shape: (B, 8, H, W, D)
        if self.return_head:
            return x[:, :self.split_size, ...]
        else:
            return x[:, self.split_size:, ...]

def get_replicate_conv(spatial_dims, in_channels, out_channels, strides=1, kernel_size=3):
    # Padding calculation to keep size same if stride=1, or halved if stride=2
    pad_size = (kernel_size - 1) // 2
    return nn.Sequential(
        nn.ReplicationPad3d(pad_size), 
        Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            strides=strides,
            kernel_size=kernel_size,
            padding=0, 
            conv_only=True
        )
    )

class AnisotropicEncoder(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        channels: Sequence[int],
        out_channels: int,
        num_res_blocks: Sequence[int],
        norm_num_groups: int,
        norm_eps: float,
        attention_levels: Sequence[bool],
        strides: Sequence[Sequence[int]], 
        with_nonlocal_attn: bool = True,
    ) -> None:
        super().__init__()
        self.spatial_dims = spatial_dims
        
        blocks: List[nn.Module] = []
        # Initial projection
        blocks.append(get_replicate_conv(spatial_dims, in_channels, channels[0], strides=1))

        input_channel = channels[0]
        
        for i in range(len(channels)):
            output_channel = channels[i]
            
            # 1. ResBlocks
            for _ in range(num_res_blocks[i]):
                blocks.append(
                    AEKLResBlock(
                        spatial_dims=spatial_dims,
                        in_channels=input_channel,
                        norm_num_groups=norm_num_groups,
                        norm_eps=norm_eps,
                        out_channels=output_channel,
                    )
                )
                input_channel = output_channel
                
                if attention_levels[i]:
                    blocks.append(
                        SpatialAttentionBlock(
                            spatial_dims=spatial_dims,
                            num_channels=input_channel,
                            norm_num_groups=norm_num_groups,
                            norm_eps=norm_eps,
                        )
                    )

            # 2. Downsample (Stride) - APPLIED AT END OF EVERY BLOCK defined in strides
            # Use the stride corresponding to this block index
            s = strides[i]
            # Check if this stride actually downsamples anything
            if sum(s) > 3: # i.e., not [1,1,1]
                blocks.append(get_replicate_conv(spatial_dims, input_channel, input_channel, strides=s))

        # Final bottleneck blocks
        if with_nonlocal_attn:
            blocks.append(AEKLResBlock(spatial_dims, input_channel, norm_num_groups, norm_eps, input_channel))
            blocks.append(SpatialAttentionBlock(spatial_dims, input_channel, norm_num_groups, norm_eps))
            blocks.append(AEKLResBlock(spatial_dims, input_channel, norm_num_groups, norm_eps, input_channel))

        blocks.append(nn.GroupNorm(num_groups=norm_num_groups, num_channels=input_channel, eps=norm_eps, affine=True))
        blocks.append(get_replicate_conv(spatial_dims, input_channel, out_channels, strides=1))

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class AnisotropicDecoder(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        channels: Sequence[int],
        in_channels: int,
        out_channels: int,
        num_res_blocks: Sequence[int],
        norm_num_groups: int,
        norm_eps: float,
        attention_levels: Sequence[bool],
        strides: Sequence[Sequence[int]],
        with_nonlocal_attn: bool = True,
    ) -> None:
        super().__init__()
        
        # Reverse configurations for decoder
        reversed_channels = list(reversed(channels))
        reversed_strides = list(reversed(strides))
        reversed_attention = list(reversed(attention_levels))
        reversed_num_res = list(reversed(num_res_blocks))

        blocks: List[nn.Module] = []
        
        # Project latent to memory dimensions
        blocks.append(get_replicate_conv(spatial_dims, in_channels, reversed_channels[0], strides=1))

        if with_nonlocal_attn:
            blocks.append(AEKLResBlock(spatial_dims, reversed_channels[0], norm_num_groups, norm_eps, reversed_channels[0]))
            blocks.append(SpatialAttentionBlock(spatial_dims, reversed_channels[0], norm_num_groups, norm_eps))
            blocks.append(AEKLResBlock(spatial_dims, reversed_channels[0], norm_num_groups, norm_eps, reversed_channels[0]))

        block_in_ch = reversed_channels[0]
        
        for i in range(len(reversed_channels)):
            block_out_ch = reversed_channels[i]
            
            # 1. Upsample (Opposite of Encoder Downsample)
            # If the encoder had a stride here, we upsample BEFORE processing the block
            s = reversed_strides[i]
            if sum(s) > 3:
                blocks.append(
                    Upsample(
                        spatial_dims=spatial_dims,
                        mode="nontrainable",
                        in_channels=block_in_ch,
                        out_channels=block_in_ch,
                        interp_mode="trilinear",
                        scale_factor=tuple(s), 
                        align_corners=False, # Usually False for trilinear
                    )
                )
                # Post-upsample convolution to smooth artifacts
                blocks.append(get_replicate_conv(spatial_dims, block_in_ch, block_in_ch, strides=1))

            # 2. ResBlocks
            for _ in range(reversed_num_res[i]):
                blocks.append(
                    AEKLResBlock(
                        spatial_dims=spatial_dims,
                        in_channels=block_in_ch,
                        norm_num_groups=norm_num_groups,
                        norm_eps=norm_eps,
                        out_channels=block_out_ch,
                    )
                )
                block_in_ch = block_out_ch
                
                if reversed_attention[i]:
                    blocks.append(
                        SpatialAttentionBlock(
                            spatial_dims=spatial_dims,
                            num_channels=block_in_ch,
                            norm_num_groups=norm_num_groups,
                            norm_eps=norm_eps,
                        )
                    )

        blocks.append(nn.GroupNorm(num_groups=norm_num_groups, num_channels=block_in_ch, eps=norm_eps, affine=True))
        blocks.append(get_replicate_conv(spatial_dims, block_in_ch, out_channels, strides=1))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

class DualDecoderVAE(nn.Module):
    def __init__(self, latent_dim, num_classes, spatial_dims=3):
        super().__init__()
        self.latent_dim = latent_dim
        
        # --- CONFIGURATION FOR 12x12x16 LATENT ---
        # 4 Layers. X/Y Stride = 2^4 = 16. Z Stride = 1.
        # Input: 192 -> 96 -> 48 -> 24 -> 12.
        channels = (32, 64, 128) 
        strides = [
            [2, 2, 1], 
            [2, 2, 1], 
            [2, 2, 1], 
        ]
        num_res_blocks = [2, 2, 2, 2]
        attention_levels = (False, False, False, False)
        
        encoder_out_channels = 2 * latent_dim 

        self.encoder = AnisotropicEncoder(
            spatial_dims=spatial_dims,
            in_channels=1,
            channels=channels,
            out_channels=encoder_out_channels,
            num_res_blocks=num_res_blocks,
            norm_num_groups=32, norm_eps=1e-6,
            attention_levels=attention_levels,
            strides=strides, with_nonlocal_attn=False
        )
        
        self.quant_conv_mu = SplitInput(latent_dim, return_head=True)
        self.quant_conv_log_sigma = SplitInput(latent_dim, return_head=False)
        
        self.decoder_img = AnisotropicDecoder(
            spatial_dims=spatial_dims,
            channels=channels,
            in_channels=latent_dim,
            out_channels=1,
            num_res_blocks=num_res_blocks,
            norm_num_groups=32, norm_eps=1e-6,
            attention_levels=attention_levels,
            strides=strides, with_nonlocal_attn=False
        )
        
        self.decoder_seg = AnisotropicDecoder(
            spatial_dims=spatial_dims,
            channels=channels,
            in_channels=latent_dim,
            out_channels=num_classes,
            num_res_blocks=num_res_blocks,
            norm_num_groups=32, norm_eps=1e-6,
            attention_levels=attention_levels,
            strides=strides, with_nonlocal_attn=False
        )

    def reparameterize(self, mu, log_sigma):
        std = torch.exp(0.5 * log_sigma)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu = self.quant_conv_mu(h)
        log_sigma = self.quant_conv_log_sigma(h)
        z = self.reparameterize(mu, log_sigma)
        recon_img = self.decoder_img(z)
        recon_seg = self.decoder_seg(z)
        return recon_img, recon_seg, mu, log_sigma