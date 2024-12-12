# ControlNet Conditioned Model
Note, for all directories mentioned in these steps, please create them initially to not encounter possible pathing issues.

1. Generate face images with StyleGAN or another GAN and place into a folder, such as `stylegan_output`.
    - For StyleGAN, please download the [repository](https://github.com/NVlabs/stylegan2-ada-pytorch) and generate images with
```bash
python generate.py --outdir=stylegan_output --trunc=1 --seeds=0-2000 --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl
```
Note that the range given in seeds is the number of images produced, and can be adjusted to produce more, e.g. 5,000 images can be produced by changing it to `--seeds=0-5000`.

2. Generate new images with the GA images as conditioning to a ControlNet for a diffusion model by running
```bash
python controlnet_stylegan.py -i <input_dir> -m <mask_dir> -o <output_dir> -n <num_images> -p <prompt>
```
where `input_dir` is the folder containing the StyleGAN images, e.g. `stylegan_output`, `output_dir` is the folder the generated images will be saved to, e.g. `output`, `num_images` the number of images to generate, and `prompt` the prompt to use during generation, e.g. `"a person with black, curly hair"`. Please include surrounding quotation marks (`""`) over the prompt string to ensure it is captured correctly by the script. With these sample values, the command would be
```bash
python controlnet_stylegan.py -i stylegan_output -o output -n 500 -p "a person with black, curly hair"
```

3. Calculate the FID score by running
```bash
python metrics.py -o <output_dir> -d <dataset_dir>
```
where `output_dir` is the folder of the produced images, e.g. `output_mask_hair`, and `dataset_dir` is the folder containing the CelebA dataset, e.g. `data`.

Note this requires the [CelebA dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) to be manually downloaded prior to running the metrics. This must be downloaded to a folder called `celeba` inside of `dataset_dir`.