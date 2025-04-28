### GANs

This repository contains code for several GAN models, including individual models and their chained versions. In the chained models, each iteration takes additional inputs from previous iterations. For example, iteration 3 uses the biased dataset, along with outputs from iterations 1 and 2. The generated images and features are saved in a convenient format, allowing the next iteration's model to easily access the previous outputs using PyTorch's built-in `ImageDataset` function.

The models implemented include:
- Standard GAN
- GAN with perceptual loss
- GAN with FID loss

Perceptual loss is computed as mean square error of real features and generated features. Whereas, the FID loss is calculated by computing FID between the real dataset (CelebA) and the biased dataset + a batch of generated images. The previous version of FID used an approximate method, but the current version uses an iterative approach (Newton-Schulz to compute matrix square root) that provides a practically exact computation of FID on the GPU. This method enables backpropagation through the FID loss.

These new loss terms are incorporated into the generator’s loss function as: Generator Loss = Adversarial Loss + λ * New Loss
