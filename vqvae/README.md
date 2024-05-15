## [Pythae](https://github.com/clementchadebec/benchmark_VAE)

This library implements some of the most common (Variational) Autoencoder models under a unified implementation. The associated paper can be found [here](https://arxiv.org/abs/2206.08309).

This repo was used to train a Vector Quantized VAE (VQVAE) to perform feature extraction on CIFAR10 images.

## Usage

### [Conda Environment Setup](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
1. Create the environment from the environment.yml file
```bash
conda env create -f environment.yml
```
2. Activate the new environment: `conda activate pytorch-vae2`

### Dataset Download
This is needed to format the CIFAR10 dataset into the required `train_data.npz` and `eval_data.npz` files
```bash
python3 data-download.py cifar10
```

### VQVAE Training
```bash
python3 training.py --dataset cifar10 --model_name vqvae --model_config 'configs/cifar10/vqvae_config2.json' --training_config 'configs/cifar10/base_training_config2.json'
```
