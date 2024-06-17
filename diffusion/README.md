# Automated Data Augmentation using Diffusion Models
This repository hosts code for content in the SRC Diffusion report. We have tested all code using Linux and an NVIDIA RTX 4090 running CUDA version 12.4.

## Requirements
Python v3.11+ and the following libraries:
- pytorch
- torchvision
- numpy
- tqdm
- pythae
- scikit-learn
- matplotlib
- seaborn
- kornia
- torchmetrics
- PIL
- diffusers
- transformers
- datasets
- pandas
- einops
- python-dotenv

**Important:** A GPU compatible with CUDA may be required to run some sections. Additionally, the Python libraries above with a CUDA version must also be downloaded (namely pytorch and torchvision).

## Butterfly Generation (`butterfly`)
1. Download [pretrained butterfly diffusion model](https://uflorida-my.sharepoint.com/:f:/g/personal/laurachang_ufl_edu/Ell4PMR4xzVBpGRsSU3AU9YBC9okm4s4uXs0rSYtEFIExw?e=rx52GQ) and place the folder directly in the `butterfly` directory:
    ```bash
    .
    ├── ddpm-butterflies-64                  # pretrained diffusion model
    │   └── ...
    ├── butterflies.ipynb
    └── ...
    ```
    Alternatively, [train your own](https://github.com/huggingface/diffusion-models-class/blob/main/unit1/01_introduction_to_diffusers.ipynb).
2. Please run the `butterfly/butterflies.ipynb` notebook.

## Face Generation (`face`)
Please run the `face/celeba.ipynb` notebook.

## Class Imbalance Reduction (`class-imbalance`)
Please run the `class-imbalance/subset-cifar10.ipynb` notebook.

## Dimensional-Aware ControlNet (`dacn`)
1. The Scene Parse dataset needs to be [downloaded](http://sceneparsing.csail.mit.edu/) first and placed in the directory `dacn/data/ADEChallengeData2016`. Then, please run `dacn/prepare_ade.sh` before running for the first time to prepare the Scene Parse dataset appropriately. 
2. Run `dacn/train.sh` to train and generate images. To customize the prompt, edit the `--validation-prompt` flag in `dacn/train.sh` to a different string.

## Model Collapse (`model-collapse`)
Optimizing GAN-generated image performance on CIFAR10 classification.

The pipeline is carried out as follows:
1. Train a variational autoencoder (`model-collapse/vqvae`) on CIFAR10 images. This model will be used for feature extraction to compare original and synthetic images.
2. Perform GAN optimization (`model-collapse/gan-optimization`) to generate optimized synthetic images.
3. Evaluate results using classifier evaluation (`model-collapse/classifier-evaluation`).

Please see the `README.md` in each folder for specific instructions on running each pipeline section.

For t-SNE visualization of real vs synthetic images, please run the `model-collapse/tsne.ipynb` notebook.