# Anatomy-Guided Residual Motion Diffusion for Controllable 4D Cardiac MRI Synthesis

Accepted for MICCAI 2026 ! See you there ! 

This repository will share the code later after cleaning it :D

<p align="center">
  <img src="imgs/overview.png"/>
</p>

<!-- TODO: Add architecture diagram here 
This repository shares source code to run [training](#training-from-scratch), [inference](#inference), and a standalone gradio [demo](#demo) to generate motion from a text prompt.


## Demo

1. Clone the repository:
```bash
git clone https://github.com/cyiheng/TextToCine2DMRI
cd TextToCine2DMRI
```

2. Create a virtual environment:
```bash
conda create -n text2cine python=3.10
conda activate text2cine
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r demo/demo_requirements.txt
```
3. Download the pre-trained weights for the demo and place them in the `results/` folder：
- Google Drive (not available yet, will update soon)
- [Baidu](https://pan.baidu.com/s/1NybeuS2JwO1jjmhhTqm6KA?pwd=jeua)

4. Run the gradio demo with the command:
```bash
python -m demo.app
```
![gradio_interface](imgs/gradio_interface.png)


## Training from scratch

### Dependencies

1. Clone the repository:
```bash
git clone https://github.com/cyiheng/TextToCine2DMRI
cd TextToCine2DMRI
```

2. Create a virtual environment:
```bash
conda create -n text2cine python=3.10
conda activate text2cine
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### Data preparation
1. Download both [ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html) and [Kaggle](https://www.kaggle.com/competitions/second-annual-data-science-bowl/overview) dataset in the folder `data/` and unzip them.
2. Use the preprocessing of [CineMA](https://github.com/mathpluscode/CineMA)
3. Make sure the preprocessed data is saved in folder `data/ACDC_preprocessed` and `data/DSB_nifti`. It will also generate the metadata.csv files for both datasets.

### Train
As the proposed method consists of 4 stages, we provide separate training scripts for each stage. You can run them sequentially to train the full model:
```bash
python -m train.train_stage1 # Finetune the VAE on the 2D data.
python -m train.train_stage2 # Train the flow predictor on the latent space.
python -m train.train_stage3 # Finetune the Stable Diffusion UNet on the first frame.
python -m train.train_stage4 # Finetune the Stable Diffusion UNet on the full sequence with flow guidance.
```

### Inference
The code for each step of inference is available to check quickly if it works per block.
```bash
python -m inference.inference_stage1.py 
...
python -m inference.inference_full.py 
```

The global inference `inference_full.py` allow to generate the full sequence from text prompt.

-->

<!-- 
If you use this code for your research, please cite our papers.

```
TODO
```
-->
