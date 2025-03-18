import os

import torch
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

class HotDogDataset(Dataset):
	def __init__(self, root, transform=None):
		self.root = root
		self.transform = transform
		self.classes = sorted(os.listdir(root))  
		self.class_index = { class_name: i for i, class_name in enumerate(self.classes) }
		self.samples = []
        
		for class_name in self.classes:
			dir_path = os.path.join(self.root, class_name)
			for image_name in os.listdir(dir_path):
				# (image_path, label)
				self.samples.append((os.path.join(dir_path, image_name), float(self.class_index[class_name])))

	def __len__(self):
		return len(self.samples)

	def __getitem__(self, idx):
		img_path, label = self.samples[idx]
		img = cv2.imread(img_path)
		img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img_np = np.array(img_rgb)

		if self.transform != None:
			aug = self.transform(image=img_np)
			tens_img = aug["image"]
		else:
			tens_img = torch.tensor(img_np)

		tens_img = tens_img.float()

		return tens_img, torch.tensor(label)
    

transform = A.Compose([
    A.Resize(224, 224),  # Изменяем размер
	A.HorizontalFlip(p=0.5),  # Горизонтальный флип с вероятностью 50%
	A.VerticalFlip(p=0.5),
    A.Rotate(limit=40, p=0.5),  # Поворот на случайный угол до 40 градусов с вероятностью 50%
	A.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
    ToTensorV2()  # Преобразование в тензоры
])