# Short experiment testing conditioned ensemble model of GAN outputs as inputs to ControlNet diffusion model
from diffusers.utils import load_image, make_image_grid
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
from PIL import Image
import cv2
import numpy as np

# Image produced by GAN
# You can find this in my Hipergator directory in the ensemble_test folder.
original_image = load_image("seed0301.png")
image = np.array(original_image)

low_threshold = 100
high_threshold = 200

# Get canny edges for input to ControlNet
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

# Generate sample from diffusion model with text prompt and GAN image as input
output1 = pipe(
    "a man", image=original_image, control_image=canny_image,
).images[0]
output2 = pipe("a man with black hair", image=original_image, control_image=canny_image).images[0]
output3 = pipe("a man with red hair", image=original_image, control_image=canny_image).images[0]
make_image_grid([output1, output2, output3], rows=1, cols=3).save("output6.png")