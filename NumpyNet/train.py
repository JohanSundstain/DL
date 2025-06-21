import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from lgbt import lgbt
import logging
import pickle

from model import NumNet
from pychort import cross_entropy_loss, softmax

logging.basicConfig(level=logging.INFO, format='%(message)s', filename='train.log', filemode='w')
LR = 0.01
EPOCH = 10
MIN_LOSS = float('inf')

if __name__ == "__main__":
	transform = transforms.Compose([
		transforms.ToTensor(),  
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  
	])

	trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
											download=True, transform=transform)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,
											shuffle=True)

	testset = torchvision.datasets.CIFAR10(root='./data', train=False,
										download=True, transform=transform)
	testloader = torch.utils.data.DataLoader(testset, batch_size=1,
											shuffle=False)

	class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
				'dog', 'frog', 'horse', 'ship', 'truck']


	model = NumNet(num_classes=10)
	
	for epoch in range(EPOCH):
		running_loss = 0.0

		for image, label in lgbt(trainloader,desc=f'epoch {epoch}', mode="chn"):
			image = image.squeeze().numpy()
			one_hot_label = np.zeros(10)
			one_hot_label[label.item()] = 1

			logits = model.forward(image)
			loss = cross_entropy_loss(logits=logits, labels=one_hot_label)
			running_loss += loss

			prop = softmax(logits)
			grad_logits = prop - one_hot_label
			
			model.backward(grad_logits)
			model.update_weights(LR)
		print(f'loss {running_loss/len(trainloader)}')
		logging.info(f'train loss {running_loss/len(trainloader)}')

		running_loss = 0.0
		for image, label in lgbt(testloader, mode="usa"):
			image = image.squeeze().numpy()
			one_hot_label = np.zeros(10)
			one_hot_label[label.item()] = 1

			logits = model.forward(image)
			loss = cross_entropy_loss(logits=logits, labels=one_hot_label)
			running_loss += loss
		
		if running_loss < MIN_LOSS:
			MIN_LOSS = running_loss
			with open("model.pkl", "wb") as f:
				pickle.dump(model, f)
			print('file was saved')
	
		print(f'loss {running_loss/len(testloader)}')
		logging.info(f'valid loss {running_loss/len(testloader)}')
			


            
              
 