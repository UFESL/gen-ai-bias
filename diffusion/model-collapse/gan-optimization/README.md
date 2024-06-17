This repo performs the synthetic image optimization pipeline. Code for the original GAN model can be found [here](https://github.com/NVlabs/stylegan2-ada-pytorch) and code for latent vector recovery can be found [here](https://github.com/yxlao/reverse-gan.pytorch)

## Usage
### Prerequisites
1. Move the trained vqvae model files into `./pythae_cifar_vqvae2`

The folder structure should be as follows:
```bash
.
├── dnnlib # helper file directory
│   └── ...
├── out # synthetic image results are saved here
│   └── {dataset_size}
│       └── {optimization iteration}
│           ├── bird
│           ├── car
│           └── ...
├── pythae_cifar_vqvae2 # the pretrained VAE
│   ├── decoder.pkl
│   ├── encoder.pkl
│   ├── environment.json
│   ├── model_config.json
│   ├── model.pt
│   ├── training_config.json
├── dcgan_reverse_mmd_real_latent.py
├── README.md
└── run.sh
```
2. Set up conda environment as in [vqvae](../vqvae/README.md#conda-environment-setup)

### Gan Optimization Pipeline
```bash
conda activate pytorch-vae2
./run.sh
```