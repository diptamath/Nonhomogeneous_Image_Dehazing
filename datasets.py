import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from PIL import Image
from skimage import io, transform
import os
import numpy as np
import random

class NH_HazeDataset(Dataset):
    def __init__(self, hazed_image_files, dehazed_image_files, root_dir, crop=False, crop_size=256, multi_scale=False, rotation=False, color_augment=False, transform=None):
        """
        Args:
             split_file: Path to the split file
             root_dir: Directory with all the images
             transform: Optional transform to be appeared on a sample
        """
        hazed_file = open(hazed_image_files, 'r')
        self.hazed_image_files = hazed_file.readlines()
        dehazed_file = open(dehazed_image_files, 'r')
        self.dehazed_image_files = dehazed_file.readlines()
        self.root_dir = root_dir
        self.transform = transform        
        self.crop = crop
        self.crop_size = crop_size
        self.multi_scale = multi_scale
        self.rotation = rotation
        self.color_augment = color_augment
        self.rotate90 = transforms.RandomRotation(90)  
        self.rotate45 = transforms.RandomRotation(45)    

    def __len__(self):
        return len(self.hazed_image_files)

    def __getitem__(self, idx):
        image_name = self.hazed_image_files[idx][0:-1].split('/')
        hazed_image = Image.open(os.path.join(self.root_dir, image_name[0], image_name[1], image_name[2])).convert('RGB')
        dehazed_image = Image.open(os.path.join(self.root_dir, image_name[0], 'GT', image_name[2])).convert('RGB')  # change the filename
        
        if self.rotation:
            degree = random.choice([90, 180, 270])
            hazed_image = transforms.functional.rotate(hazed_image, degree) 
            dehazed_image = transforms.functional.rotate(dehazed_image, degree)

        if self.color_augment:
            #contrast_factor = 1 + (0.2 - 0.4*np.random.rand())
            #hazed_image = transforms.functional.adjust_contrast(hazed_image, contrast_factor)
            #dehazed_image = transforms.functional.adjust_contrast(dehazed_image, contrast_factor)
            hazed_image = transforms.functional.adjust_gamma(hazed_image, 1)
            dehazed_image = transforms.functional.adjust_gamma(dehazed_image, 1)                           
            sat_factor = 1 + (0.2 - 0.4*np.random.rand())
            hazed_image = transforms.functional.adjust_saturation(hazed_image, sat_factor)
            dehazed_image = transforms.functional.adjust_saturation(dehazed_image, sat_factor)
            
        if self.transform:
            hazed_image = self.transform(hazed_image)
            dehazed_image = self.transform(dehazed_image)

        if self.crop:
            W = hazed_image.size()[1]
            H = hazed_image.size()[2] 

            Ws = np.random.randint(0, W-self.crop_size-1, 1)[0]
            Hs = np.random.randint(0, H-self.crop_size-1, 1)[0]
            
            hazed_image = hazed_image[:,Ws:Ws+self.crop_size,Hs:Hs+self.crop_size]
            dehazed_image = dehazed_image[:,Ws:Ws+self.crop_size,Hs:Hs+self.crop_size]
                       
        if self.multi_scale:
            H = dehazed_image.size()[1]
            W = dehazed_image.size()[2]
            hazed_image_s1 = transforms.ToPILImage()(hazed_image)
            dehazed_image_s1 = transforms.ToPILImage()(dehazed_image)
            hazed_image_s2 = transforms.ToTensor()(transforms.Resize([H/2, W/2])(hazed_image_s1))
            dehazed_image_s2 = transforms.ToTensor()(transforms.Resize([H/2, W/2])(dehazed_image_s1))
            hazed_image_s3 = transforms.ToTensor()(transforms.Resize([H/4, W/4])(hazed_image_s1))
            dehazed_image_s3 = transforms.ToTensor()(transforms.Resize([H/4, W/4])(dehazed_image_s1))
            hazed_image_s1 = transforms.ToTensor()(hazed_image_s1)
            dehazed_image_s1 = transforms.ToTensor()(dehazed_image_s1)
            return {'hazed_image_s1': hazed_image_s1, 'hazed_image_s2': hazed_image_s2, 'hazed_image_s3': hazed_image_s3, 'dehazed_image_s1': dehazed_image_s1, 'dehazed_image_s2': dehazed_image_s2, 'dehazed_image_s3': dehazed_image_s3}
        else:
            return {'hazed_image': hazed_image, 'dehazed_image': dehazed_image}
        
