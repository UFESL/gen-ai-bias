# Automated Data Augmentation using Ensemble Models
This repository hosts code for content in the SRC Ensemble Model report. We have tested all code using Linux and an NVIDIA A100 80GB GPU running CUDA version 12.4.

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
- accelerate

**Important:** A GPU compatible with CUDA may be required to run some sections. Additionally, the Python libraries above with a CUDA version must also be downloaded (namely pytorch and torchvision).

## ControlNet Conditioned Ensemble Model (`controlnet`)

## Masked Conditioned Ensemble Model (`masked`)
Please see the README.md in the folder for more specifics on how to run, such as custom pathing.
1. Generate face images with StyleGAN or another GAN and place into a folder, such as `stylegan_output`.

2. Create mask images for StyleGAN images by running
```bash
python mask_stylegan.py -i <input_dir> -o <output_dir> -l <label>
```
3. Generate new images from the StyleGAN images and their corresponding masks by inpainting the mask area by running
```bash
python inpaint_stylegan.py -i <input_dir> -m <mask_dir> -o <output_dir> -n <num_images> -p <prompt>
```

4. Calculate the FID score by running
```bash
python metrics.py -o <output_dir> -d <dataset_dir>
```

## Blend Ensemble Model (`blend`)