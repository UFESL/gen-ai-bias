import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import transforms
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader
from PIL import Image
import os
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output_dir", type=str, default="output_mask_hair")
parser.add_argument("-d", "--dataset", type=str, default="data")
args = parser.parse_args()

transform = transforms.Compose([transforms.Resize((299, 299)), transforms.ToTensor()])

fid = FrechetInceptionDistance(normalize=True)

ds = CelebA(args.dataset, split="test", transform=transform, download=False)
dl = DataLoader(ds, batch_size=128)

for batch_idx, batch in enumerate(tqdm(dl)):
    data, label = batch
    fid.update(data, real=True)

images = []
for filename in sorted(os.listdir(args.output_dir)):
    image = Image.open(f"{args.output_dir}/{filename}").convert('RGB')
    image = transform(image)
    images.append(image)
fake = torch.stack(images)

fid.update(fake, real=False)
result = fid.compute()
print(result.item())