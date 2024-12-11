# Use a pipeline as a high-level helper
from transformers import pipeline
import PIL
import os
import numpy as np
import argparse

def mask(img_dir, out_dir, mask_label):
    pipe = pipeline("image-segmentation", model="jonathandinu/face-parsing", device=0)

    for file in sorted(os.listdir(img_dir)):
        file_name, file_extension = os.path.splitext(file)
        out = pipe(f"{img_dir}/{file}")
        for entry in out:
            label = entry["label"]
            if label == mask_label:
                mask = np.array(entry["mask"])
                newim = PIL.Image.fromarray(mask)
                newim.save(f"{out_dir}/{file_name}.png")

# Script arguments
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_dir", type=str, default="stylegan_output")
parser.add_argument("-o", "--output_dir", type=str, default="mask_hair")
parser.add_argument("-l", "--label", type=str, default="hair")
args = parser.parse_args()

# Perform mask of given label on given input images
mask(args.input_dir, args.output_dir, args.label)