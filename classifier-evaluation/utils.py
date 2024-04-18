'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math

import torch.nn as nn
import torch.nn.init as init

import torch
from torch.utils.data import Subset
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report
import seaborn as sns
import numpy as np
from torchvision.datasets import ImageFolder

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None, backspace=True):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    if (backspace):
        for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
            sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

# Get training and validation dataloaders following specified sizes/proportions
def split_dataset(cifar_dataset, train_size, val_size, proportion_animal, superclass_relabel):
    # Define CIFAR10 classes that belong to animal or vehicle
    animal_classes = [2, 3, 4, 5, 6, 7]
    vehicle_classes = [0, 1, 8, 9]

    # Map labels to animal/vehicle superclasses
    cifar_dataset_superclass_targets = [0 if label in animal_classes else 1 for label in cifar_dataset.targets]

    # Relabel to superclasses
    if (superclass_relabel):
        cifar_dataset.targets = cifar_dataset_superclass_targets

    assert len(cifar_dataset_superclass_targets) == len(cifar_dataset), "Superclass target labels don't match dataset size"

    # Get indices of animal and vehicle superclasses
    animal_indices = [i for i, label in enumerate(cifar_dataset_superclass_targets) if label == 0]
    vehicle_indices = [i for i, label in enumerate(cifar_dataset_superclass_targets) if label == 1]
    assert len(animal_indices) + len(vehicle_indices) == len(cifar_dataset), "Animal and vehicle indices do not sum to length of full dataset"

    # Calculate the number of samples for each class in the subsets
    num_animal_train = int(train_size * proportion_animal)
    num_vehicle_train = train_size - num_animal_train
    num_animal_val = int(val_size * proportion_animal)
    num_vehicle_val = val_size - num_animal_val
    assert num_animal_train + num_animal_val == int((train_size+val_size) * proportion_animal), "Animal training and validation split do not sum to specified sizes/proportions"
    assert num_vehicle_train + num_vehicle_val == train_size+val_size-num_animal_train-num_animal_val, "CIFAR train/val splits do not sum to specified sizes/proportions"

    # Randomly select positions from animal and vehicle superclasses 
    animal_indices_selected = torch.randperm(len(animal_indices))[:num_animal_train+num_animal_val] # Shuffle animal_indices positions, select subset to cover train and val
    animal_indices_train_selected = animal_indices_selected[:num_animal_train] # Take first part for training
    animal_indices_val_selected = animal_indices_selected[num_animal_train:num_animal_train+num_animal_val] # Take second part for validation
    vehicle_indices_selected = torch.randperm(len(vehicle_indices))[:num_vehicle_train+num_vehicle_val]
    vehicle_indices_train_selected = vehicle_indices_selected[:num_vehicle_train]
    vehicle_indices_val_selected = vehicle_indices_selected[num_vehicle_train:num_vehicle_train+num_vehicle_val]

    # Use the selected indexes to take desired subsets of animal and vehicle superclasses
    train_indices = torch.cat([torch.tensor(animal_indices)[animal_indices_train_selected], torch.tensor(vehicle_indices)[vehicle_indices_train_selected]])
    val_indices = torch.cat([torch.tensor(animal_indices)[animal_indices_val_selected], torch.tensor(vehicle_indices)[vehicle_indices_val_selected]])

    # Check no overlap between the two
    assert len(set(train_indices.numpy()).intersection(set(val_indices.numpy()))) == 0, f"Overlap between training and validation indices: {set(train_indices.numpy()).intersection(set(val_indices.numpy()))}"

    # Check the labels indeed fall within animal and vehicle superclasses
    assert all(cifar_dataset_superclass_targets[i] == 0 for i in torch.tensor(animal_indices)[animal_indices_train_selected]), "Not all entries in animal_indices are animals"
    assert all(cifar_dataset_superclass_targets[i] == 1 for i in torch.tensor(vehicle_indices)[vehicle_indices_train_selected]), "Not all entries in vehicle_indices are vehicles"

    # Create subsets from indices
    trainset = Subset(cifar_dataset, train_indices)
    valset = Subset(cifar_dataset, val_indices)

    assert len(trainset) == train_size, f"Size of training subset differs from specification: {len(trainset)} != {train_size}"
    assert len(valset) == val_size, f"Size of validation subset differs from specification: {len(valset)} != {val_size}"

    split_dataset_sanity_check(trainset, animal_classes, train_size, proportion_animal, superclass_relabel)

    return trainset, valset

def split_dataset_sanity_check(dataset, animal_classes, train_size=-1, proportion_animal=-1, superclass_relabel = False):
    num_animals = 0

    if (superclass_relabel):
        for _, label in dataset:
            # print(label)
            if label == 0:
                num_animals += 1
    else:
        for _, label in dataset:
            # print(label)
            if label in animal_classes:
                num_animals += 1

    num_vehicles = len(dataset) - num_animals

    print("animal count:", num_animals)
    print("vehicle count", num_vehicles)

    if (train_size != -1 and proportion_animal != -1):
        assert num_animals == (int)(train_size * proportion_animal), f"Animal count doesn't align with specified parameters:{num_animals} != {(int)(train_size * proportion_animal)}"
        assert num_vehicles == train_size - num_animals, "Vehicle count doesn't align with specified parameters"

def plot_class_distribution(dataset, split, filename, cifar_classes, superclass_relabel):
    font_size = 20

    # Dictionary to store counts for each class
    class_counts = defaultdict(int)

    # Count occurrences of each class
    for _, label in dataset:
        class_counts[label] += 1

    # Extract counts for each class
    counts_per_class = [class_counts[i] for i in range(len(cifar_classes))]

    if (superclass_relabel):
        plt.figure(figsize=(5, 4))
    else:
        plt.figure(figsize=(10, 4))
    plt.bar(cifar_classes, counts_per_class, color='skyblue')
    # plt.xlabel('CIFAR Classes')
    plt.ylabel('Count', fontsize=font_size)

    # Rotate the x-axis labels for better visibility if needed
    if (superclass_relabel):
        plt.xticks(fontsize=font_size)
    else:
        plt.xticks(rotation=30, fontsize=font_size)
    plt.yticks(fontsize=font_size)

    plt.tight_layout()
    plt.savefig(filename)

def print_classification_report(model, testloader, class_names, device, filepath, epoch, superclass_relabel):
    if superclass_relabel:
        font_size = 40
    else:
        font_size = 20

    # Lists to store ground truth and predicted labels
    all_ground_truth = []
    all_predicted_labels = []

    # Iterate through the entire test set
    for images, labels in testloader:
        with torch.no_grad():
            outputs = model(images.to(device))
            _, predicted = torch.max(outputs, 1)

        # Append the ground truth and predicted labels to the lists
        all_ground_truth.extend(labels.cpu().numpy())
        all_predicted_labels.extend(predicted.cpu().numpy())

    # Convert lists to numpy arrays
    ground_truth = np.array(all_ground_truth)
    predicted_labels = np.array(all_predicted_labels)

    # Calculate accuracy
    accuracy = accuracy_score(ground_truth, predicted_labels)
    print(f'Accuracy: {accuracy:.2f}')

    # Calculate F1 score
    f1 = f1_score(ground_truth, predicted_labels, average='weighted')
    print(f'F1 Score: {f1:.2f}')

    # Save the confusion matrix
    cm = confusion_matrix(ground_truth, predicted_labels)
    plt.figure(figsize=(10, 9))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, cbar=False, annot_kws={"fontsize":font_size})
    # plt.title(f'Confusion Matrix')
    plt.xlabel('Predicted', fontsize=font_size)
    plt.ylabel('Ground Truth', fontsize=font_size)
    # plt.show()

    # Rotate the x-axis labels for better visibility if needed
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)

    plt.tight_layout()
    plt.savefig(f'{filepath}/confusion_matrix_{epoch}.png')

    classification_report_path = f'{filepath}/classification_report_{epoch}.txt'
    with open(classification_report_path, 'w') as file:
        file.write("Classification Report:\n" + classification_report(ground_truth, predicted_labels, target_names=class_names))

    # Print a message to indicate where the classification report is saved
    print(f"Classification report saved to {classification_report_path}")

def augment_dataset(synthetic_dataset, proportion_animal, train_size, superclass_relabel):
    # Calculate number of images to augment
    num_animals = int(train_size * proportion_animal)
    num_vehicles = train_size - num_animals
    num_augment = abs(num_animals - num_vehicles)

    if proportion_animal <= 0.5: # Had more vehicles in training
        synth_proportion_animal = 1 # So augment with animal images
    else: # Had more animals in training
        synth_proportion_animal = 0 # So augment with vehicle images

    # Add synthetic images to dataloaders
    synth_trainset, synth_valset = split_dataset(synthetic_dataset, num_augment, int(num_augment*0.2), synth_proportion_animal, superclass_relabel)

    return synth_trainset, synth_valset

# Edit ImageFolder class so labels return as superclasses
class CustomImageFolder(ImageFolder):
    def __init__(self, root, transform=None, target_transform=None):
        super(CustomImageFolder, self).__init__(root, transform=transform, target_transform=target_transform)

    def __getitem__(self, index):
        # Call the parent class's __getitem__ method to get the original behavior
        original_result = super(CustomImageFolder, self).__getitem__(index)

        # Modify the target (label) to return superclass instead of individual class index
        image, original_label = original_result
        animal_classes = [2, 3, 4, 5, 6, 7]
        superclass_label = 0 if original_label in animal_classes else 1

        # Return the modified result
        return image, superclass_label