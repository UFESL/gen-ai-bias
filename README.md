This repository hosts code for content in the SRC Diffusion report.

## Butterfly Generation (TECHCON Work)
See the [report-figures](./report-figures/) directory.

## CelebA Diffusion (Guidance towards multiple attributes)
See the [report-figures](./report-figures/) directory.

## AAAI Work
See https://github.com/UFESL/unaugment:
- [distribution bar graphs and original/synthetic CIFAR-10 image comparisons](https://github.com/UFESL/unaugment/blob/main/src/debiasing/subset-cifar10.ipynb)
- [celeb-a diffusion model](https://colab.research.google.com/drive/1D56x2yyXy67vL-NsfgbA4tAKhic9abAc)

To generate classifier performance confusion matrices, the [classifier-evaluation directory](https://github.com/UFESL/gen-ai-bias/tree/main/classifier-evaluation) can be used as a reference.

## t-SNE Visualization (real vs synthetic images)
See the [report-figures](./report-figures/) directory.

## Optimizing GAN-generated image performance on CIFAR10 classification

The pipeline is carried out as follows:
1. [Train a variational autoencoder](./vqvae/) on CIFAR10 images. This model will be used for feature extraction to compare original and synthetic images.
2. [Perform GAN optimization](./gan-optimization/) to generate optimized synthetic images.
3. Evaluate results using [classifier evaluation](./classifier-evaluation/).

**Notes** A GPU compatible with CUDA may be required to run some sections. Most of the conda environments likely include many more packages than are required to run the uploaded code (I directly exported what I experimented with).