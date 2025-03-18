import torch
import timm
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from mydataloader import HotDogDataset, transform
from train import train, test
from model import MobNet

def main():
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	num_classes = 1

	model = MobNet(num_classes=1)
	model = model.to(device=device)

	model2 = timm.create_model("mobilenetv2_100", pretrained=False)
	model2.classifier = nn.Linear(model2.classifier.in_features, num_classes)
	model2.to(device=device)

	train_dataset = HotDogDataset("../dataset/train", transform=transform)
	valid_dataset = HotDogDataset("../dataset/valid", transform=transform)	
	test_dataset = HotDogDataset("../dataset/test", transform=transform)

	train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
	valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
	test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

	train(model=model, device=device, tloader=train_loader, vloader=valid_loader, epoch=10, validation=5)
	#test(model=model, path=".\\weights\\_100.pth", device=device, tloader=test_loader)

if __name__ == "__main__":
	main()