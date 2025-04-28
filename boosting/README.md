# Automated Data Augmentation using Ensemble Boosting Models
This repository hosts code for content in the SRC Ensemble Boosting Model. We have tested all code using Linux and an NVIDIA A100 80GB GPU running CUDA version 12.4.

## Requirements
Python v3.10+ and the following libraries:
- pytorch (with CUDA)
- torchvision
- numpy
- tqdm
- PIL
- pandas
- scipy
- scikit-learn
- matplotlib

This module primarily focuses on generating synthetic images using GANs, incorporating various loss functions. We used the Fréchet inception distance (FID) as the metric of bias between any two datasets.

### `Prepare_datasets`
The `prepare_datasets.ipynb` script is used to create and save subsets of the CelebA dataset, such as "Smiling", "Male", "Young", and others. Additionally, it allows for the specification of exclusions (e.g., "Not Male", "Not Smiling"), effectively creating custom subclasses of the CelebA dataset for further use in training and evaluation.
It is required to provide the location of the CelebA images and a CSV file containing the attribute information for the images.



