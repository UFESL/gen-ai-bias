# Generative Adeversarial Networks (GANs)
To run the sample code, call the run.sh script provided, for example:

```bash
bash run.sh
```

You can download the horse2zebra dataset using the following command. Please manually download CelebA dataset from https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html 

```bash
bash download_datasets.sh
```

The following Python packages are required: 

- Pytorch
- Pytorch-lightning
- torchvision
- tqdm
- click
- requests
- pyspng
- ninja 
- imageio-ffmpeg==0.4.3 

Please note that StyleGAN will only run on an NVIDIA GPU with CUDA version 11.1 or higher on Linux.