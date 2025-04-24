import os
import torch
import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import Inception_V3_Weights
from scipy.linalg import sqrtm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
inception = models.inception_v3(weights=Inception_V3_Weights.DEFAULT)
inception.fc = torch.nn.Identity()  # Remove classification head
inception.eval().to(device)

transform = transforms.Compose([
    transforms.Resize((299, 299)),  # Resize to InceptionV3 input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])

def get_image_paths(folder, extensions=(".png", ".jpg", ".jpeg")):
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(extensions)]

def extract_and_save_features(image_folder, save_folder):
    os.makedirs(save_folder, exist_ok=True)  # Create folder if it doesn't exist
    image_paths = get_image_paths(image_folder)

    for path in image_paths:
        img_name = os.path.splitext(os.path.basename(path))[0]  # Get filename without extension
        save_path = os.path.join(save_folder, f"{img_name}.npy")
        image = Image.open(path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device

        with torch.no_grad():
            feature = inception(image)  # Shape: [1, 2048]
        
        np.save(save_path, feature.cpu().numpy().squeeze())

    print(f"Features saved in {save_folder}")
    
def features_to_stat(*feature_folders):
    feature_files = []
    
    # Collect all .npy files from the given folders
    for folder in feature_folders:
        feature_files.extend([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".npy")])
    
    features = []

    for file in feature_files:
        feature = np.load(file)
        features.append(feature)
        
    N = len(features)
    features = np.array(features)
        
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    
    return {'mean': mu, 'covariance': sigma, 'num_samples': N}

def save_stats(stats, filename):
    np.savez(filename, mean=stats['mean'], covariance=stats['covariance'], num_samples=stats['num_samples'])
    
def load_stats(filename):
    data = np.load(filename)
    return {'mean': data['mean'], 'covariance': data['covariance'], 'num_samples': data['num_samples'].item()}
    
    
def calculate_fid(mu_r, sigma_r, mu_g, sigma_g):
    mean_diff = np.sum((mu_r - mu_g) ** 2)
    
    sqrt_sigma_r_sigma_g = sqrtm(sigma_r @ sigma_g)
    cov_diff = np.trace(sigma_r + sigma_g - 2 * sqrt_sigma_r_sigma_g)
    
    fid_score = mean_diff + np.real(cov_diff)
    return fid_score

def calculate_fid_svd(mu_r, sigma_r, mu_g, sigma_g):
    mean_diff = torch.sum((torch.tensor(mu_r) - torch.tensor(mu_g)) ** 2)
    
    # SVD of covariance matrices
    U_r, S_r, V_r = torch.svd(torch.tensor(sigma_r))
    U_g, S_g, V_g = torch.svd(torch.tensor(sigma_g))
    
    # Square root of covariance matrices
    sqrt_sigma_r = U_r @ torch.diag(torch.sqrt(S_r)) @ V_r.t()
    sqrt_sigma_g = U_g @ torch.diag(torch.sqrt(S_g)) @ V_g.t()
    
    cov_diff = torch.trace(torch.tensor(sigma_r) + torch.tensor(sigma_g) - 2 * (sqrt_sigma_r @ sqrt_sigma_g))
    
    fid_score = mean_diff + cov_diff
    return fid_score