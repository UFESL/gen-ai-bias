import os
from PIL import Image
import torch
from torchvision.datasets import VisionDataset
import numpy as np
from transformers import pipeline
from diffusers.utils import load_image
from pathlib import Path
import torchvision
import pandas


class ADE20kDataset(VisionDataset):
    def __init__(self, root="data", split="train", transform=None):
        super().__init__()
        self.split = split
        df = pandas.read_csv(f"{root}/sceneCategories.txt", header=None, sep="\s+")
        self.labels = df[1]
        self.labels = [label.replace("_", " ") for label in self.labels]
        self.root = f"{root}/{self.split}"
        self.root_depth = f"{self.root}/depth"
        self.root_anno = f"{self.root}/annotations"
        self.root_orig = f"{self.root}/images"
        self.transform = transform
        self.filenames = sorted(os.listdir(self.root_anno))

        # Only produce depth maps once, if files in folder assumed already done
        if len(os.listdir(self.root_depth)) == 0:
            self.get_depth_maps()

    def get_depth_maps(self):
        # Load depth estimator via HF
        depth_estimator = pipeline("depth-estimation")
        for file in sorted(os.listdir(self.root_orig)):
            filename = Path(file).stem
            image = load_image(f"{self.root_orig}/{file}")
            image = depth_estimator(image)["depth"]
            image = np.array(image)
            image = image[:, :, None]
            image = np.concatenate([image, image, image], axis=2)
            detected_map = torch.from_numpy(image).float() / 255.0
            depth_map = detected_map.permute(2, 0, 1).unsqueeze(0).half()
            torchvision.utils.save_image(depth_map, f"{self.root_depth}/{filename}.png")

    def __getitem__(self, idx):
        label = self.labels[idx]
        segment = Image.open(os.path.join(self.root_anno, self.filenames[idx])).convert("RGB")
        depth = Image.open(os.path.join(self.root_depth, self.filenames[idx])).convert("RGB")
        orig = Image.open(
            os.path.join(self.root_orig, f"{Path(self.filenames[idx]).stem}.jpg")
        ).convert("RGB")
        if self.transform is not None:
            segment = self.transform(segment)
            depth = self.transform(depth)
            orig = self.transform(orig)

        overlay = segment * depth

        return {"overlay": overlay, "label": label, "segment": segment, "depth": depth, "orig": orig}

    def __len__(self):
        return len(self.filenames)
