import torch
from torch.utils.data import DataLoader

from mydataloader import Dataset, transform, transform_hsv
from model import Unet
from train import train


if __name__ == "__main__":
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')	

	train_dataset = Dataset(config_path='train.txt', transform=transform_hsv)
	valid_dataset = Dataset(config_path='valid.txt', transform=transform_hsv)

	tloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
	vloader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

	model = Unet(1)

	train(model=model, device=device, tloader=tloader, vloader=vloader, epoches=100, valid=5)


