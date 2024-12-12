from diffusers.utils import load_image, make_image_grid
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
from PIL import Image
import cv2
import numpy as np
import os
import argparse

def gen_images(num_imgs, input_dir, output_dir, prompt):
    files = sorted(os.listdir(input_dir))
    num_files = len(files)
    if num_files < num_imgs:
        num_imgs = num_files
    
    test_files = files[:num_imgs]
    for file in test_files:
        original_image = load_image(f"{input_dir}/{file}")
        image = np.array(original_image)

        low_threshold = 100
        high_threshold = 200

        image = cv2.Canny(image, low_threshold, high_threshold)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        canny_image = Image.fromarray(image)

        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16, use_safetensors=True)
        pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16, use_safetensors=True
        )
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()

        output = pipe(prompt, image=original_image, control_image=canny_image).images[0]
        output.save(f"{output_dir}/{file}")

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_dir", type=str, default="input")
parser.add_argument("-o", "--output_dir", type=str, default="output")
parser.add_argument("-n", "--num_imgs", type=int, default=500)
parser.add_argument("-p", "--prompt", type=str, default="a person with white hair")
args = parser.parse_args()

gen_images(args.num_imgs, args.input_dir, args.output_dir, args.prompt)