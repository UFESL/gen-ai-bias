import random
import matplotlib.pyplot as plt
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm

def get_folder_size(folder_path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for file in filenames:
            file_path = os.path.join(dirpath, file)
            total_size += os.path.getsize(file_path)
    return total_size


def plot_random_image_from_dir(directory):
    # List all image files in the directory
    image_files = [f for f in os.listdir(directory) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    random_index = random.randint(0, len(image_files) - 1)
    
    # Get the corresponding image file
    selected_image = image_files[random_index]
    
    # Open and plot the image
    image_path = os.path.join(directory, selected_image)
    image = Image.open(image_path)
    
    # Plot the image
    plt.imshow(image)
    plt.axis('off')  # Hide axis
    plt.show()

def count_files_in_folder(folder_path):
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    return len(files)

def compute_fid(real_dir, gen_dir, img_size=(128, 128), batch_size=16, feature_dim=2048, device=torch.device('cuda')):
    def load_images_from_folder(folder):
        images = []
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
        ])
        for filename in tqdm(os.listdir(folder), desc=f"Loading images from {folder}"):
            img_path = os.path.join(folder, filename)
            if os.path.splitext(filename)[-1].lower() not in valid_extensions:
                continue  # Skip non-image files
            try:
                img = Image.open(img_path).convert('RGB')
                img = transform(img)
                images.append(img)
            except Exception as e:
                print(f"Error loading image {filename}: {e}")
        print(f"{len(images)} Images")
        return images

    def process_batches(images, is_real):
        for i in tqdm(range(0, len(images), batch_size), desc=f"Processing {'real' if is_real else 'generated'} images"):
            batch = images[i:i + batch_size]
            batch = torch.stack(batch).to(device)  # Move batch to GPU
            batch = (batch * 255).clamp(0, 255).to(torch.uint8)  # Convert to uint8 and clamp pixel values
            fid_metric.update(batch, real=is_real)
            del batch  # Free up memory
            torch.cuda.empty_cache()

    # Initialize FID metric
    fid_metric = FrechetInceptionDistance(feature=feature_dim).to(device)

    # Load real and generated images
    real_images = load_images_from_folder(real_dir)
    generated_images = load_images_from_folder(gen_dir)

    # Process images and update FID metric
    process_batches(real_images, is_real=True)
    process_batches(generated_images, is_real=False)

    # Compute FID score
    fid_score = fid_metric.compute()
    print(f"FID Score: {fid_score}")
    return fid_score