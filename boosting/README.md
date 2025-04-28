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

### Prepare datasets
The `prepare_datasets.ipynb` file is used to create and save subsets of the CelebA dataset, such as "Smiling", "Male", "Young", and others. Additionally, it allows for the specification of exclusions (e.g., "Not Male", "Not Smiling"), effectively creating custom subclasses of the CelebA dataset for further use in training and evaluation.
It is required to provide the location of the CelebA images and a CSV file containing the attribute information for the images.

### Biased datasets
The `biased_subsets.ipynb` file takes two folders created by the `prepare_datasets.ipynb` and generates a new folder containing a specified percentage of images from the first folder, and the remaining images from the second folder. It also saves the corresponding features and FID statistics for further analysis.

### FID custom
The `fid_custom.py` file contains a custom implementation of the FID metric. It also includes functionality to save the features of images (extracted from the last layer of the InceptionV3 model) along with the mean and covariance of these features. Saving these statistics helps avoid redundant computations in the future, significantly reducing processing time.

### FID of augmented datasets
The `fid_aug_custom.ipynb` file computes the FID score at each step given the locations of features for the biased dataset and the generated images (`folder_path`). At each step, 10 images from the generated images are successively added, and the corresponding FID scores are saved in the `fid_list_data` folder.

### FID graph
The `fid_graph.ipynb` file takes the data saved in the `fid_list_data` folder and visualizes the FID scores on a graph for various models. This allows for granular observation of how the generated images affect the FID scores over time.

### Chain FID stats
The `chain_fid_stats.ipynb` file computes the FID score at each stage of image generation when chaining multiple GANs together. It also visualizes these FID scores on a graph and performs a t-SNE of the mean feature vectors to analyze the features in a lower-dimensional space.

### GANs
The `GANs` folder contains various GAN architectures and their chained versions. Please refer to the contents of the folder for more detailed information about the specific models used.

**Note:** The `celebA_data` folder contains the one-time setup for features, including the mean and covariance of CelebA images features.


