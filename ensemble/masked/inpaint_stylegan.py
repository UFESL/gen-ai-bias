import torch
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image
import os
import argparse

def gen_images(num_imgs, input_dir, mask_dir, output_dir, prompt):
    pipeline = AutoPipelineForInpainting.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting", torch_dtype=torch.float16
    )
    pipeline.enable_model_cpu_offload()

    files = sorted(os.listdir(mask_dir))
    num_files = len(files)
    if num_files < num_imgs:
        num_imgs = num_files

    gen_files = files[:num_imgs]
    for file in gen_files:
        init_image = load_image(f"{input_dir}/{file}")
        mask_image = load_image(f"{mask_dir}/{file}")
        image = pipeline(prompt=prompt, image=init_image, mask_image=mask_image).images[0]
        image.save(f"{output_dir}/{file}")

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_dir", type=str, default="input")
parser.add_argument("-o", "--output_dir", type=str, default="output_mask_hair")
parser.add_argument("-m", "--mask_dir", type=str, default="mask_hair")
parser.add_argument("-n", "--num_imgs", type=int, default=500)
parser.add_argument("-p", "--prompt", type=str, default="a person with black, curly hair")
args = parser.parse_args()

gen_images(args.num_imgs, args.input_dir, args.mask_dir, args.output_dir, args.prompt)