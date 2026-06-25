import torch
from custom_AEKL import AnisotropicEncoder, AnisotropicDecoder, SplitInput
from monai.networks.nets import AutoencoderKL


def setup_vae(latent_dim=8, num_classes=4):
    # 1. Initialize Standard VAE
    autoencoder = AutoencoderKL(
        spatial_dims=3, in_channels=1, out_channels=1 + num_classes, channels=(32, 64, 128),
        latent_channels=latent_dim, num_res_blocks=[2,2,2], norm_num_groups=32, attention_levels=(False, False, False),
        norm_eps = 1e-06, with_encoder_nonlocal_attn=False, with_decoder_nonlocal_attn=False,
    )

    custom_strides = [[2, 2, 1], [2, 2, 1], [2, 2, 1]]
    # 3. Replace Encoder
    autoencoder.encoder = AnisotropicEncoder(
        spatial_dims=3,
        in_channels=1,
        channels=(64,128,256),
        out_channels=latent_dim * 2, # Latent * 2
        num_res_blocks=[2, 2, 2],
        norm_num_groups=32,
        norm_eps=1e-6,
        attention_levels=(False, False, False),
        strides=custom_strides, # Pass custom strides
        with_nonlocal_attn=False,
    )

    # 4. Replace Decoder
    autoencoder.decoder = AnisotropicDecoder(
        spatial_dims=3,
        channels=(32, 64, 128),
        in_channels=latent_dim,
        out_channels=1 + num_classes,
        num_res_blocks=[2, 2, 2],
        norm_num_groups=32,
        norm_eps=1e-6,
        attention_levels=(False, False, False),
        strides=custom_strides, # Pass custom strides
        with_nonlocal_attn=False,
    )

    # 4. Override Wrapper Internal Layers
    # The wrapper tries to use convolutions to split Mu/Sigma. 
    # We replace them with our splitting logic because our encoder output is already correct.

    # Input (8ch) -> Take first 4 -> Mu (4ch)
    autoencoder.quant_conv_mu = SplitInput(latent_dim, return_head=True)

    # Input (8ch) -> Take last 4 -> Sigma (4ch)
    autoencoder.quant_conv_log_sigma = SplitInput(latent_dim, return_head=False)

    # Latent (4ch) -> Identity -> Decoder (4ch)
    autoencoder.post_quant_conv = torch.nn.Identity()

    # Handle older MONAI versions that might use a single 'quant_conv' attribute
    if hasattr(autoencoder, 'quant_conv'):
        autoencoder.quant_conv = torch.nn.Identity()
    # --- CRITICAL FIX END ---

    print(f"  -> Anisotropic VAE configured with Slicing Layers.")

    return autoencoder
