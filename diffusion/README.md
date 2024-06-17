# Automated Data Augmentation using Diffusion Models
This repository hosts code for content in the SRC Diffusion report.

## Requirements
Python v3.10+ and the following libraries:
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
- datasets
- pandas

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

## AAAI Work
See https://github.com/UFESL/unaugment:
- [distribution bar graphs and original/synthetic CIFAR-10 image comparisons](https://github.com/UFESL/unaugment/blob/main/src/debiasing/subset-cifar10.ipynb)
- [celeb-a diffusion model](https://colab.research.google.com/drive/1D56x2yyXy67vL-NsfgbA4tAKhic9abAc)

To generate classifier performance confusion matrices, the [classifier-evaluation directory](https://github.com/UFESL/gen-ai-bias/tree/main/classifier-evaluation) can be used as a reference.

## Model Collapse (`model-collapse`)
Optimizing GAN-generated image performance on CIFAR10 classification.

The pipeline is carried out as follows:
1. Train a variational autoencoder (`model-collapse/vqvae`) on CIFAR10 images. This model will be used for feature extraction to compare original and synthetic images.
2. Perform GAN optimization (`model-collapse/gan-optimization`) to generate optimized synthetic images.
3. Evaluate results using classifier evaluation (`model-collapse/classifier-evaluation`).

Run the full pipeline with:
```bash
cd model-collapse
bash run.sh
```
For t-SNE visualization of real vs synthetic images, please run the `model-collapse/tsne.ipynb` notebook.