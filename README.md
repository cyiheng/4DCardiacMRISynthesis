# Anatomy-Guided Residual Motion Diffusion for Controllable 4D Cardiac MRI Synthesis

This repository contains the official PyTorch implementation of our paper:
Anatomy-Guided Residual Motion Diffusion for Controllable 4D Cardiac MRI Synthesis
Accepted at MICCAI 2026, Strasbourg, France.

##  Overview

We propose a 4D controllable generative framework for anatomically consistent data augmentation.
Unlike prior methods that focus solely on intensity synthesis or require external pseudo-labeling for downstream task training, our framework directly generates anatomically aligned image-mask pairs. By decoupling 4D synthesis into static anatomical generation and ED-referenced residual latent motion prediction, we reduce learning complexity while ensuring strict temporal coherence and subject-specific anatomical consistency ([Example](imgs/SupplementaryFile-3003.mp4)).

<p align="center">
  <img src="imgs/overview.png"/>
</p>

This repository shares source code to run [training](#training-from-scratch), [inference](#inference), and a standalone gradio [demo](#demo).


## Demo

1. Clone the repository:
```bash
git clone https://github.com/cyiheng/4DCardiacMRISynthesis
cd 4DCardiacMRISynthesis
```

2. Create a virtual environment:
```bash
conda create -n synthenv python=3.10
conda activate synthenv
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r demo/demo_requirements.txt
```
3. Download the pre-trained weights for the demo and place them in the `demo/weights/` folder：
- Google Drive (not available yet, will update soon)
- [Baidu](https://pan.baidu.com/s/11rkAlZhLc2qzhRzRRyNKyg?pwd=hpng)

4. Run the gradio demo with the command:
```bash
python -m demo.app
```
![gradio_interface](imgs/gradio_interface.png)


## Training from scratch

### Dependencies

1. Clone the repository:
```bash
git clone https://github.com/cyiheng/4DCardiacMRISynthesis
cd 4DCardiacMRISynthesis
```

2. Create a virtual environment:
```bash
conda create -n synthenv python=3.10
conda activate synthenv
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### Data preparation
1. Download both [ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html) and [Kaggle](https://www.kaggle.com/competitions/second-annual-data-science-bowl/overview) dataset in the folder `data/` and unzip them.
2. Use the preprocessing of [CineMA](https://github.com/mathpluscode/CineMA)
3. Make sure the preprocessed data is saved in folder `data/ACDC_preprocessed` and `data/DSB_nifti`. It will also generate the metadata.csv files for both datasets.
4. Prepare the training and testing split set (change path if necessary)
```
python -m preprocess.to_3d
python -m preprocess.split
```
You should obtain a dataset folder with the following structure:
```text
.4DCardiacMRISynthesis
└── data
    ├── images
    │   ├── ACDC_patient001_frame_000.nii.gz
    │   └── ...
    ├── labels
    │   ├── ACDC_patient001_frame_000.nii.gz
    │   └──  ...
    ├── bad_cases.csv
    ├── test_split_final.csv
    └── train_split_final.csv
```

### Train
As the proposed method consists of 4 stages, we provide separate training scripts for each stage. You can run them sequentially to train the full model:
```bash
# 1. Finetune the VAE.
python -m train.stage1 

# 2. Train the Deterministic Latent Motion.
python -m datasets.encode # pre encode the image into latent vectors
# Visual check (OPTIONAL): python -m datasets.decode # decode the .pt back to image space
python -m train.stage2 

# 3. Train the latent diffusion model for the anatomy generation
python -m datasets.augment # data augmentation for training stage 3
python -m train.stage3

# 4. Train the latent diffusion model for the motion generation
python -m train.stage4
```
*Note: Do not forget to adapt the scaling factor if trained from scratch*

### Inference
The code for each step of inference is available to check quickly if it works per block.
```bash
python -m inference.stage1
...
python -m inference.gen4d
```
*Note: Update the paths according to yours*

The global inference `gen4d.py` allow to generate the full sequence.

### Citation
If you find this code or our paper useful for your research, please cite:
```
TODO
```
