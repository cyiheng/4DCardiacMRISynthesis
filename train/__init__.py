"""train: training entrypoints and stage scripts for multi-stage model development.

This package contains separate training scripts for each major pipeline stage:
- stage1.py: VAE dual-decoder autoencoder pretraining with segmentation guidance
- stage2.py: latent-space motion predictor training using precomputed latents
- stage3.py: first-frame diffusion generator training with clinical conditioning
- stage4.py: flow-aware diffusion training using latent motion and time conditioning
"""
