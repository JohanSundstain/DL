import os

import torch
import cv2
from torch.utils.data import Dataset, DataLoader
import albumentations as A 
import numpy as np
from albumentations.pytorch import ToTensorV2

class Dataset(Dataset):
	def __init__(self, config_path, transform=None):
		self.config_path = config_path
		self.transform = transform
		self.samples = []
		self.mask_shape = (388, 388)
		self.image_shape = (572, 572)
		with open(config_path, "r") as f:
			self.pathes = f.readlines()
        
		for path in self.pathes:
			if path == '\n':
				continue
			
			image_path, mask_path = path.split(" ")
			self.samples.append((image_path, mask_path[:-1]))

	def __len__(self):
		return len(self.samples)

	def __getitem__(self, idx):
		image_path, mask_path = self.samples[idx]
		if os.name == 'posix':
			image_path = image_path.replace('\\', '/')
			mask_path = mask_path.replace('\\', '/')

		img = cv2.imread(image_path)
		img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
		 
		mask_np = np.array(mask)
		img_np = np.array(img_rgb)

		if self.transform != None:
			aug = self.transform(image=img_np, mask=mask_np)
			tens_img = aug["image"]
			tens_mask = aug["mask"]
			tens_mask = tens_mask.unsqueeze(0)
		else:
			tens_img = torch.tensor(img_np)
			tens_mask = torch.tensor(mask_np)

		tens_mask[tens_mask > 1] = 1
		tens_img = tens_img.float()
		tens_mask = tens_mask.float()

		return tens_img, tens_mask 
    
transform = A.Compose([
	A.Resize(640, 640),
	A.HorizontalFlip(p=0.5),  
	A.VerticalFlip(p=0.5),
    A.Rotate(limit=40, p=0.5),  
	A.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
    ToTensorV2() 
])

transform_hsv = A.Compose([
	A.Resize(640, 640),
    A.HueSaturationValue(sat_shift=30, val_shift=20, p=1.0),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
    A.Rotate(limit=30, p=0.5),
    A.HorizontalFlip(p=0.5),
	ToTensorV2()
])
