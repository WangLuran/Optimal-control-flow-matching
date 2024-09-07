# coding=utf-8

from utils import run_lib_flowgrad_oc
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import logging
import os
import torchvision.transforms as transforms
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader, SubsetRandomSampler
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision import models

# Path to your CelebA images folder
image_dir = 'path_to_your_downloaded_celeba_images'

# Transform to resize and convert to tensor
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images to 128x128 pixels
    transforms.ToTensor(),          # Convert to tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
])


class CelebADataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.image_files = os.listdir(img_dir)  # Get list of image filenames

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")  # Load image
        
        if self.transform:
            image = self.transform(image)
        
        return image

# Create the dataset and DataLoader
celeba_dataset = CelebADataset(image_dir, transform=transform)

print('data finish')

# Set random seed for reproducibility
np.random.seed(42)

# Generate random indices for sampling 1,000 images
indices = np.random.choice(len(celeb_a_dataset), 1000, replace=False)

# Create a subset sampler
sampler = SubsetRandomSampler(indices)

# Load the sampled subset with a DataLoader
data_loader = DataLoader(celeb_a_dataset, sampler=sampler, batch_size=64)

print('data loader finish')

config = RectifiedFlow/configs/celeba_hq_pytorch_rf_gaussian.py
text_prompt = 'A photo of a smiling face.'
alpha = 0.7
model_path = '../checkpoint_10.pth'

def main(argv):
  run_lib_flowgrad_oc.flowgrad_edit(config, text_prompt, alpha, model_path, data_loader, output_folder)


if __name__ == "__main__":
  app.run(main)
