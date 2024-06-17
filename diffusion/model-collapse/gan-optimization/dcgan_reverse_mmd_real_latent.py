from __future__ import print_function
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils

import dnnlib
import legacy
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import Subset

import numpy as np
import csv

import math

from pythae.models import AutoModel
# from pythae.models import VQVAE, VQVAEConfig
# vae_model_config = VQVAEConfig.from_json_file("vqvae_config2.json")
# from pythae.models.nn.benchmarks.cifar import (
#     Encoder_Conv_AE_CIFAR as Encoder_VQVAE,
# )
# from pythae.models.nn.benchmarks.cifar import (
#     Decoder_Conv_AE_CIFAR as Decoder_VQVAE,
# )

import os

import datetime

class RBF(nn.Module):

    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()
        self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.data.sum() / (n_samples ** 2 - n_samples)

        return self.bandwidth

    def forward(self, X):
        device = X.device  # Get the device of input tensor
        L2_distances = torch.cdist(X, X) ** 2
        bandwidth = self.get_bandwidth(L2_distances).to(device)  # Ensure bandwidth is on the same device as X
        bandwidth_multipliers = self.bandwidth_multipliers.to(device)
        return torch.exp(-L2_distances[None, ...] / (bandwidth * bandwidth_multipliers)[:, None, None]).sum(dim=0)


class MMDLoss(nn.Module):

    def __init__(self, kernel=RBF()):
        super().__init__()
        self.kernel = kernel

    def forward(self, X, Y):
        device = X.device  # Get the device of input tensors
        X = X.to(device)  # Ensure X is on the same device as Y
        Y = Y.to(device)  # Ensure Y is on the same device as X
        K = self.kernel(torch.vstack([X, Y]))

        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()
        return XX - 2 * XY + YY

def reverse_z(netG, g_z, opt, label, clip='disabled'):
    """
    Estimate z_approx given G and G(z).

    Args:
        netG: nn.Module, generator network.
        g_z: Variable, G(z).
        opt: argparse.Namespace, network and training options.
        z: Variable, the ground truth z, ref only here, not used in recovery.
        clip: Although clip could come from of `opt.clip`, here we keep it
              to be more explicit.
    Returns:
        Variable, z_approx, the estimated z value.
    """
    # sanity check
    assert clip in ['disabled', 'standard', 'stochastic']

    device = torch.device('cuda')

    # loss metrics
    mmd_loss = MMDLoss().to(device)
    # mse_loss_ = nn.MSELoss().to(device)

    # init tensor
    if opt.z_distribution == 'uniform':
        z_approx = torch.FloatTensor(opt.batch, netG.z_dim).uniform_(-1, 1).to(device)
    elif opt.z_distribution == 'normal':
        z_approx = torch.FloatTensor(opt.batch, netG.z_dim).normal_(0, 1).to(device)
    else:
        raise ValueError()

    z_approx.requires_grad = True

    # optimizer
    optimizer_approx = optim.Adam([z_approx], lr=opt.lr,
                                  betas=(opt.beta1, 0.999))
    # maybe also try rmsprop, original authors encouraged

    # vae
    vae = AutoModel.load_from_folder(
        'pythae_cifar_vqvae2'
    ).to(device)

    g_z = g_z.to(device)
    g_z_latent = vae.embed(g_z) # Get G(z) embeddings
    g_z_latent = g_z_latent.detach()
    # original_shape = g_z.shape
    # g_z = g_z.view(g_z.shape[0], -1)
    
    loop_count = math.ceil(opt.size/opt.batch)
    # last batch should be
    for batch_iter in range(loop_count):
        # train
        for i in range(opt.niter):

            if batch_iter == loop_count - 1:
                g_z_batch_latent = g_z_latent[batch_iter * opt.batch:]
            else:
                g_z_batch_latent = g_z_latent[batch_iter * opt.batch: (batch_iter + 1) * opt.batch]

            # print(i)
            if (i % 500 == 0):
                # save z_approx tensor
                # torch.save(z_approx, f'z_approx_{i}.pth')
                # with open(f'z_approx_{i}.csv', 'w', newline='') as csvfile:
                #     csvwriter = csv.writer(csvfile)
                #     csvwriter.writerows(z_approx.detach().cpu().numpy())

                print(f"z_approx: {z_approx}, max: {torch.max(z_approx)}, min: {torch.min(z_approx)}, median: {torch.median(z_approx)}, mean: {torch.mean(z_approx)}, std: {torch.std(z_approx)}")

            g_z_approx = netG(z_approx, label, truncation_psi=1, noise_mode='const')

            g_z_approx_latent = vae.embed(g_z_approx)

            # g_z_approx = g_z_approx.view(g_z_approx.shape[0], -1)
            # mse_g_z = mse_loss(g_z_approx, g_z)
            # mse_z = mse_loss_(z_approx, z)
            # g_z_latent = vae.embed(g_z)
            mmd_g_z = mmd_loss(g_z_approx_latent, g_z_batch_latent)

            # mse_g_z = mse_loss(F.log_softmax(input_dist, dim=1), cifar10_data)
            # if i % 100 == 0:
            #     print("[Iter {}] mse_g_z: {}, MSE_z: {}"
            #           .format(i, mse_g_z.data.item(), mse_z.data.item()))
            if i % 100 == 0:
                # Get the current time
                current_time = datetime.datetime.now()

                # Format the time as a string
                time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")

                print(f"[{time_str} Iter {i}] mmd_dist: {mmd_g_z.data.item()}")
                # print(f"[Iter {i}] mse_dist: {mse_g_z.data.item()} MSE_z: {mse_z.data.item()}")

            # bprop
            optimizer_approx.zero_grad()
            # mse_g_z.backward()
            mmd_g_z.backward()
            optimizer_approx.step()

            # clipping
            if clip == 'standard':
                z_approx.data[z_approx.data > 1] = 1
                z_approx.data[z_approx.data < -1] = -1
            if clip == 'stochastic':
                z_approx.data[z_approx.data > 1] = random.uniform(-1, 1)
                z_approx.data[z_approx.data < -1] = random.uniform(-1, 1)

            if i % 500 == 0:
                # g_z_approx_restored = torch.clone(g_z_approx)
                # g_z_approx_restored = g_z_approx_restored.view(original_shape)
                # vutils.save_image(g_z_approx.data, f'g_z_approx_{i}.png', normalize=True)
                # # save g(z_approx) image
                # vutils.save_image(g_z_approx_restored.data, 'g_z_approx_final.png', normalize=True)

                # Step 1: Find min and max
                min_val = torch.min(g_z_approx.data)
                max_val = torch.max(g_z_approx.data)

                # Step 2: Subtract min from all data points
                g_z_approx_norm = g_z_approx - min_val
                # Step 3: Divide by range
                g_z_approx_norm /= (max_val - min_val)

                # Save individual images
                cifar_classes = {
                    0: 'airplane',
                    1: 'automobile',
                    2: 'bird',
                    3: 'cat',
                    4: 'deer',
                    5: 'dog',
                    6: 'frog',
                    7: 'horse',
                    8: 'ship',
                    9: 'truck'
                }
                class_name = cifar_classes[opt.class_idx]
            
                outdir = f"out/{opt.size}/opt_iter_{i}/{class_name}"
                os.makedirs(outdir, exist_ok=True)

                save_ctr = batch_iter * opt.batch

                for idx, image in enumerate(g_z_approx_norm):
                    output_path = f"{outdir}/s_{class_name}_{save_ctr:05d}.png"  # Provide the path where 
                    # print(image.cpu().numpy().transpose(1,2,0).shape)
                    # exit(0)
                    # vutils.save_image(image, output_path, normalize=True)
                    vutils.save_image(image, output_path)

                    # PIL.Image.fromarray(image.cpu().numpy().transpose(1,2,0), 'RGB').save(output_path)
                    save_ctr += 1 # For the next image

        # Next batch has different random initialization
        if opt.z_distribution == 'uniform':
            z_approx = torch.FloatTensor(opt.batch, netG.z_dim).uniform_(-1, 1).to(device)
        elif opt.z_distribution == 'normal':
            z_approx = torch.FloatTensor(opt.batch, netG.z_dim).normal_(0, 1).to(device)


    return z_approx


def reverse_gan(opt):
    device = torch.device('cuda')

    # load original images
    transform = transforms.Compose([transforms.ToTensor()])
    cifar10_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    cifar_train_class_indices = [i for i, i_label in enumerate(cifar10_train.targets) if i_label == opt.class_idx]
    random.shuffle(cifar_train_class_indices)
    cifar_train_subset = Subset(cifar10_train, cifar_train_class_indices[:opt.size])

    g_z = torch.stack([data for data, target in cifar_train_subset])
    g_z = g_z * 2 - 1 # try to match range of gan outputs. maybe normalize before comparison
    
    # vutils.save_image(g_z.data, 'g_z.png', normalize=True)

    with dnnlib.util.open_url(opt.network) as f:
        netG = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    for param in netG.parameters():
        param.requires_grad = False

    # Labels.
    label = torch.zeros([opt.batch, netG.c_dim], device=device)
    if netG.c_dim != 0:
        assert opt.class_idx is not None, 'Must specify class label with --class when using a conditional network'
        label[:, opt.class_idx] = 1
    else:
        if opt.class_idx is not None:
            print ('warn: --class=lbl ignored when running on an unconditional network')

    # recover z_approx from standard
    z_approx = reverse_z(netG, g_z, opt, label, clip=opt.clip)
    print(z_approx.cpu().data.numpy().squeeze())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip', default='stochastic',
                        help='disabled|standard|stochastic')
    parser.add_argument('--z_distribution', default='uniform',
                        help='uniform | normal')
    # parser.add_argument('--nz', type=int, default=100,
    #                     help='size of the latent z vector')
    # parser.add_argument('--nc', type=int, default=3,
    #                     help='number of channels in the generated image')
    # parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--niter', type=int, default=5000,
                        help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='beta1 for adam. default=0.5')
    parser.add_argument('--ngpu', type=int, default=1,
                        help='number of GPUs to use')
    # parser.add_argument('--netG', default='dcgan_out/netG_epoch_10.pth',
    #                     help="path to netG (to continue training)")
    parser.add_argument('--network', type=str, 
                        default='https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/cifar10.pkl', help='path to generator')
    parser.add_argument('--outf', default='dcgan_out',
                        help='folder to output images and model checkpoints')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--profile', action='store_true',
                        help='enable cProfile')
    parser.add_argument('--batch', type=int, default=1, help='number of images to handle at a time')
    parser.add_argument('--class_idx', type=int, help='class index')
    parser.add_argument('--size', type=int, default=1, help='dataset size to generate')

    opt = parser.parse_args()
    print(opt)

    if opt.size < opt.batch:
        raise argparse.ArgumentTypeError(f"Dataset size must be at least as large as batch (batch={opt.batch}).")

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    device = torch.device('cuda')
    if device=='cuda':
        torch.cuda.manual_seed_all(opt.manualSeed)
        cudnn.benchmark = True  # turn on the cudnn autotuner
        # torch.cuda.set_device(1)

    reverse_gan(opt)
