import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import random

class TrainWithValidDataset(Dataset):
	valid_samples = []
	train_samples = []
	samples = []
	def __init__(self, ann_path, image_path, transforms=None, k=0.1):
		self.k = k
		self.image_names = {}
		self.transforms = transforms

		self._fill(ann_path, image_path)
		
		valid_files, train_files = self._separate()

		for sample in valid_files:
			TrainWithValidDataset.valid_samples.append(sample)

		for sample in train_files:
			TrainWithValidDataset.train_samples.append(sample)

		TrainWithValidDataset.train()
		
	@staticmethod	
	def train():
		TrainWithValidDataset.samples = TrainWithValidDataset.train_samples
	
	@staticmethod
	def valid():
		TrainWithValidDataset.samples = TrainWithValidDataset.valid_samples

	@staticmethod
	def clear():
		TrainWithValidDataset.samples.clear()
		TrainWithValidDataset.train_samples.clear()
		TrainWithValidDataset.valid_samples.clear()
	
	def _fill(self, ann_path, image_path):

		with open(ann_path, "r") as f:
			ann = f.readlines()
			ann = ann[1::]

		for line in ann:
			if line == "\n":
				continue
			splitted_line = line.split(",")
			file_name = os.path.join(image_path, splitted_line[0])
			if file_name not in self.image_names:
				self.image_names[file_name] = []

			self.image_names[file_name].append(splitted_line[1::])


	def _separate(self):
		n = int(len(self.image_names) * self.k)
		shuffled = list(self.image_names.items())
		random.shuffle(shuffled)

		return shuffled[:n], shuffled[n:] 

	def __len__(self):
		return len(TrainWithValidDataset.samples)

	def __getitem__(self, idx):
		img_path, labels = TrainWithValidDataset.samples[idx]

		img = Image.open(img_path).convert("RGB")
		w, h = img.size

		boxes = []
		classes = []

		for label in labels:
			w, h = label[0], label[1]
			xmin, ymin, xmax, ymax = int(label[3]), int(label[4]), int(label[5]), int(label[6])
			boxes.append([xmin, ymin, xmax, ymax])
			classes.append(1)
          

		boxes = torch.tensor(boxes, dtype=torch.float32)
		classes = torch.tensor(classes, dtype=torch.int64)

		target = {
            'boxes': boxes,
            'labels': classes,
            'image_id': torch.tensor([idx]),
            'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64)
        }

		if self.transforms:
			img = self.transforms(img)

		return img, target, img_path
	