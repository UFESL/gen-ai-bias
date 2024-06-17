#!/bin/bash

# TSNE visualization
echo "To get the t-SNE visualization of the original and synthetic images (Figure 17), please run the tsne.ipynb notebook separately."

# VQVAE
cd vqvae
python data-download.py cifar10
python training.py --dataset cifar10 --model_name vqvae --model_config 'configs/cifar10/vqvae_config2.json' --training_config 'configs/cifar10/base_training_config2.json'