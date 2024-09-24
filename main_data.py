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
import numpy as np
from torchvision import datasets, transforms
import imageio
import json

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file("config", None, "Rectified Flow Model configuration.", lock_config=True)
flags.DEFINE_integer("batch_size", 1, "batch size")
flags.DEFINE_integer("index", 0, "position of samples")
text_prompts = ['A photo of an old face.','A photo of a sad face.','A photo of a smiling face.','A photo of an angry face.','A photo of a face with curly hair.']
alpha = 0.7
model_path = '../checkpoint_10.pth'

# # Create an empty dictionary
# empty_dict = {}

# # Define the path for the new empty JSON file
# file_path = 'output/results.json'

# # Write the empty dictionary to the file
# with open(file_path, 'w') as json_file:
#     json.dump(empty_dict, json_file)

# print(f"Empty JSON file created at {file_path}")

def get_img(path=None):
    img = imageio.imread(path) ### 4-no expression
    print(img.shape)
    img = img / 255.
    img = img[np.newaxis, :, :, :]
    img = img.transpose(0, 3, 1, 2)
    print('read image from:', path, 'img range:', img.min(), img.max())
    img = torch.tensor(img).float()
    img = torch.nn.functional.interpolate(img, size=256)

    return img

# Path to your CelebA images folder
image_dir = 'data_celeba_hq_1024'

class CelebADataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.image_files = os.listdir(img_dir)  # Get list of image filenames

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_files[idx])
        image = get_img(img_path)  # Load image
        
        return image

# Create the dataset and DataLoader
celeba_dataset = CelebADataset(image_dir)

print('data finish')

# Set random seed for reproducibility
np.random.seed(42)


def main(argv):
  # Generate random indices for sampling 1,000 images
  indices = [i+FLAGS.index for i in range(100)]

  # Create a subset sampler
  sampler = SubsetRandomSampler(indices)
  # Load the sampled subset with a DataLoader
  data_loader = DataLoader(celeba_dataset, sampler=sampler, batch_size=FLAGS.batch_size)

  print('data loader finish')

  #c, l, i = run_lib_flowgrad_oc.flowgrad_edit(FLAGS.config, text_prompts, alpha, model_path, data_loader)
  _ = run_lib_flowgrad_oc.dflow_edit(FLAGS.config, text_prompts, alpha, model_path, data_loader)




if __name__ == "__main__":
  app.run(main)

