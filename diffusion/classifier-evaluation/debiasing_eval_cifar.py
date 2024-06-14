'''Train MNIST with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar, split_dataset, plot_class_distribution, print_classification_report, augment_dataset
import sys
import random
import numpy as np
from torch.utils.data import ConcatDataset
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--seed', default=0, type=int, help='seed')
parser.add_argument('--train_size', default=0, type=int, help='training dataset size')
parser.add_argument('--synthetic', action='store_true', help='train classifier from synthetic image')
parser.add_argument('--optimized', action='store_true', help='train classifier from optimized synthetic images')
parser.add_argument('--val_split', default=0.2, type=float, help='val_size = val_split * train_size')
parser.add_argument('--architecture', default='DLA', type=str, help='classifier architecture')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Deterministic evaluation
torch.manual_seed(args.seed)
# torch.cuda.manual_seed(args.seed)
# torch.cuda.manual_seed_all(args.seed)
# np.random.seed(args.seed)
# random.seed(args.seed)

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(args.seed)

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    # torchvision.transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    # torchvision.transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Split dataset according to specified sizes and proportions
save_folder = f'output/CIFAR/{args.train_size}_{args.seed}_{args.architecture}'
redirect_progress_to_file = True # Instead of printing to std out

if (args.optimized):
    save_folder = f'{save_folder}_synthetic_optimized'
    dataset = torchvision.datasets.ImageFolder('./data/gan-opt/10000/opt_iter_5000/', transform=transform_train)
elif (args.synthetic):
    save_folder = f'{save_folder}_synthetic'
    # dataset = torchvision.datasets.ImageFolder('./data/cifar-10-synthetic/', transform=transform_train)
    # dataset = torchvision.datasets.ImageFolder('./data/cifar-10-GAN/', transform=transform_train)
    dataset = torchvision.datasets.ImageFolder('./data/gan-opt/10000/opt_iter_0/', transform=transform_train)
else:
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)

trainset, valset = train_test_split(dataset, train_size=args.train_size, test_size=int(args.train_size*args.val_split), random_state=args.seed)   

os.makedirs(save_folder, exist_ok=True)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2, worker_init_fn=seed_worker, generator=g)
valloader = torch.utils.data.DataLoader(
    valset, batch_size=100, shuffle=False, num_workers=2, worker_init_fn=seed_worker, generator=g)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
    
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2, worker_init_fn=seed_worker, generator=g)

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
plot_class_distribution(trainset, 'Training', f'{save_folder}/train_dist.png', classes, superclass_relabel=False)
plot_class_distribution(valset, 'Validation', f'{save_folder}/val_dist.png', classes, superclass_relabel=False)

# Model
print('==> Building model..')
if args.architecture == 'VGG':
    net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
else:
    net = SimpleDLA(num_classes=len(classes))
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    # cudnn.benchmark = True # Disabled for deterministic results

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total), backspace=not redirect_progress_to_file)


def val(epoch):
    global best_acc
    net.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(valloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (val_loss/(batch_idx+1), 100.*correct/total, correct, total), backspace=not redirect_progress_to_file)

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc

    if (epoch % 50 == 0):
        print_classification_report(net, testloader, classes, device, save_folder, epoch, superclass_relabel=False)

def epoch_loop():
    for epoch in range(start_epoch, start_epoch+200):
        train(epoch)
        val(epoch)
        scheduler.step()

if redirect_progress_to_file:
    # Redirect stdout to a file (redirect printing training & val accuracy for each epoch)
    # Carefule, overwrites
    with open(f'{save_folder}/training_progress.txt', 'w') as file:
        sys.stdout = file

        epoch_loop()

    # Reset stdout to the console
    sys.stdout = sys.__stdout__
else:
    epoch_loop()

print_classification_report(net, testloader, classes, device, save_folder, epoch=-1, superclass_relabel=False)