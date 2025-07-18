import os
from PIL import Image
import torch
import numpy as np
import random as rd 
import torchvision
import configparser
import cv2
import torchvision.transforms.functional
#for reading the config file 
from utility.settings import *
import utility.file_manager as fm
# Creating a custom dataset class 

INVERT = False
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir:str, label_dir:str, vertical_flip = False, horizontal_flip = False, other_transforms = False, **kwargs):
        
        
        self.image_dir = image_dir
        self.label_dir = label_dir

        self.image_files = sorted(os.listdir((self.image_dir)))
        self.label_files = sorted(os.listdir(self.label_dir))

        if '.Ds_store' in self.label_files:
            self.label_files.remove('.Ds_Store')
    
        if '.Ds_store' in self.image_files:
            self.image_files.remove('.Ds_Store')

        self.vertical_flip = vertical_flip
        self.horizontal_flip = horizontal_flip
        self.other_transforms = other_transforms
        self.to_tensor = torchvision.transforms.functional.to_tensor
        self.resize = torchvision.transforms.functional.resize
    def __len__(self):
        if len(self.image_files) != len(self.label_files):
            print(len(self.image_files), len(self.label_files))

        return len(self.image_files)
        

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.image_files[index])
        label_path = os.path.join(self.label_dir, self.label_files[index])
    
        image_tensor = fm.load_image_tensor(image_path)
        label_tensor = fm.load_image_tensor(label_path)
        
        # Apply random transformations with probability 0.5
        if (self.horizontal_flip and bool(AUGMENT_ACTIVE) and np.random.rand() > 0.5):
          image_tensor = self.vertical_flip(image_tensor)
          label_tensor = self.vertical_flip(label_tensor)
        if(self.vertical_flip and bool(AUGMENT_ACTIVE) and np.random.rand() > 0.5):
           image_tensor = self.vertical_flip(image_tensor)
           label_tensor = self.vertical_flip(label_tensor)
        if self.other_transforms:
            for other_transform in self.other_transforms:
                image_tensor = self.other_transform(image_tensor)
                label_tensor = self.other_transform(label_tensor)
                
            image_tensor = self.resize((TARGET_RESOLUTION, TARGET_RESOLUTION))(image_tensor)
            label_tensor = self.resize((TARGET_RESOLUTION, TARGET_RESOLUTION))(label_tensor)

        if INVERT:
            label_tensor = 1 - label_tensor

        return image_tensor, label_tensor, self.image_files[index]
    
    