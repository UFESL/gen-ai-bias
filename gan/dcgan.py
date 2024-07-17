# Adapted from https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html and https://github.com/Lightning-AI/pytorch-lightning/tree/master/examples/fabric/dcgan

import os
import time
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils
from torchvision.datasets import CelebA
from lightning.fabric import Fabric, seed_everything

# Root directory for dataset
dataroot = sys.argv[1]
# Number of workers for dataloader
workers = 2
# Batch size during training
batch_size = 128
# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64
# Number of channels in the training images. For color images this is 3
nc = 3
# Size of z latent vector (i.e. size of generator input)
nz = 100
# Size of feature maps in generator
ngf = 64
# Size of feature maps in discriminator
ndf = 64
# Number of training epochs
num_epochs = 10
# Learning rate for optimizers
lr = 0.0002
# Beta1 hyperparameter for Adam optimizers
beta1 = 0.5
# Number of GPUs available. Use 0 for CPU mode.
num_gpus = 1

def main():
    seed_everything(999)

    fabric = Fabric(accelerator="auto", devices=1)
    fabric.launch()

    # We can use an image folder dataset the way we have it setup.
    # Create the dataset
    dataset = CelebA(
        root=dataroot,
        split="all",
        transform=transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
        download=False,
    )
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=workers
    )

    output_dir = Path("outputs-fabric", time.strftime("dcgan_%Y%m%d-%H%M%S"))
    output_dir.mkdir(parents=True, exist_ok=True)

    real_batch = next(iter(dataloader))
    torchvision.utils.save_image(
        real_batch[0][:64],
        output_dir / "sample-data.png",
        padding=2,
        normalize=True,
    )

    generator = Generator()

    # Apply weights_init func to randomly init all weights
    generator.apply(weights_init)

    discriminator = Discriminator()
    discriminator.apply(weights_init)

    criterion = nn.BCELoss()
    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, nz, 1, 1, device=fabric.device)

    # Establish convention for real and fake labels during training
    real_label = 1.0
    fake_label = 0.0

    # Setup Adam optimizers for both G and D
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))

    discriminator, optimizer_d = fabric.setup(discriminator, optimizer_d)
    generator, optimizer_g = fabric.setup(generator, optimizer_g)
    dataloader = fabric.setup_dataloaders(dataloader)

    losses_g = []
    losses_d = []
    iteration = 0

    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            discriminator.zero_grad()
            # Format batch
            real = data[0]
            b_size = real.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=fabric.device)
            # Forward pass real batch through D
            output = discriminator(real).view(-1)
            # Calculate loss on all-real batch
            err_d_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            fabric.backward(err_d_real)
            d_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=fabric.device)
            # Generate fake image batch with G
            fake = generator(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = discriminator(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            err_d_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            fabric.backward(err_d_fake)
            d_g_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            err_d = err_d_real + err_d_fake
            # Update D
            optimizer_d.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            generator.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = discriminator(fake).view(-1)
            # Calculate G's loss based on this output
            err_g = criterion(output, label)
            # Calculate gradients for G
            fabric.backward(err_g)
            d_g_z2 = output.mean().item()
            # Update G
            optimizer_g.step()

            # Output training stats
            if i % 50 == 0:
                fabric.print(
                    f"[{epoch}/{num_epochs}][{i}/{len(dataloader)}]\t"
                    f"Loss_D: {err_d.item():.4f}\t"
                    f"Loss_G: {err_g.item():.4f}\t"
                    f"D(x): {d_x:.4f}\t"
                    f"D(G(z)): {d_g_z1:.4f} / {d_g_z2:.4f}"
                )

            # Save Losses for plotting later
            losses_g.append(err_g.item())
            losses_d.append(err_d.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iteration % 500 == 0) or (
                (epoch == num_epochs - 1) and (i == len(dataloader) - 1)
            ):
                start = time.time()
                with torch.no_grad():
                    fake = generator(fixed_noise).detach().cpu()
                print({time.time() - start})
                if fabric.is_global_zero:
                    torchvision.utils.save_image(
                        fake,
                        output_dir / f"fake-{iteration:04d}.png",
                        padding=2,
                        normalize=True,
                    )
                fabric.barrier()

            iteration += 1
    t = time.strftime("%Y%m%d-%H%M%S")
    print(f"End: {t}")

# custom weights initialization called on ``netG`` and ``netD``
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# Generator Code
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. ``(ngf*8) x 4 x 4``
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. ``(ngf*4) x 8 x 8``
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. ``(ngf*2) x 16 x 16``
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. ``(ngf) x 32 x 32``
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. ``(nc) x 64 x 64``
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is ``(nc) x 64 x 64``
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x 32 x 32``
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*2) x 16 x 16``
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 8 x 8``
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*8) x 4 x 4``
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, input):
        return self.main(input)
    
if __name__ == "__main__":
    main()
