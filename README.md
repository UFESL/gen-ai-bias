This repository hosts code for content in the SRC Diffusion report.

### Butterfly Generation (TECHCON Work)
See the [report-figures](./report-figures/) directory.

### Optimizing GAN-generated image performance on CIFAR10 classification. (TODO: add link to or discussion on performance drop issue and proposed solution).

The pipeline is carried out as follows:
1. [Train a variational autoencoder](./vqvae/) on CIFAR10 images. This model will be used for feature extraction to compare original and synthetic images.
2. [Perform GAN optimization](./gan-optimization/) to generate optimized synthetic images.
3. Evaluate results using [classifier evaluation](./classifier-evaluation/).

A GPU compatible with CUDA may be required to run some sections. Most of the conda environments likely include many more packages than are required to run the uploaded code (I directly exported what I experimented with).