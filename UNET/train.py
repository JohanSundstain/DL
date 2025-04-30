import torch
import cv2
from model import Unet
import numpy as np
from mydataloader import transform, Dataset
from lgbt import lgbt
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='train.log', filemode='w')

def compute_binary_iou(preds, targets):
	preds = torch.sigmoid(preds)
	preds = (preds > 0.5).float()

	intersection = (preds * targets).sum((1, 2, 3))
	union = (preds + targets - preds * targets).sum((1, 2, 3))
	iou = (intersection + 1e-6) / (union + 1e-6)
	
	return iou.mean()


def train(model, device, tloader, vloader, epoches, valid):

	optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
	criterion = torch.nn.BCEWithLogitsLoss()
	model = model.to(device=device)
	best_loss = float('inf')
	for epoch in range(epoches):
		model.train()
		running_loss = 0.0
		running_iou = 0.0
		for images, masks in lgbt(tloader, desc=f"epoch {epoch}", mode='ita', hero='tralalero'):
			images = images.to(device)
			masks = masks.to(device)

			optimizer.zero_grad()
			outputs = model(images)
			loss = criterion(outputs, masks)
			loss.backward()
			optimizer.step()
			iou = compute_binary_iou(outputs, masks)

			running_iou += iou.item()
			running_loss += loss.item()
		logging.info(f"train epoch {epoch} loss {running_loss/len(tloader)} iou {running_iou/len(tloader)}")
		print(f'train loss {running_loss/len(tloader)} train iou {running_iou/len(tloader)}')

		if epoch % valid == 0 and epoch != 0:
			running_loss = 0.0
			running_iou = 0.0
			model.eval()
			with torch.no_grad():
				for images, masks in lgbt(vloader, desc=f'valid {epoch//valid}', mode='ussr', hero='shimpanzini'):
					images = images.to(device)
					masks = masks.to(device)
					outputs = model(images)
					loss = criterion(outputs, masks)
					iou = compute_binary_iou(outputs, masks)

					running_loss += loss.item()
					running_iou += iou.item()
				
				logging.info(f"valid epoch {epoch} loss {running_loss/len(vloader)} iou {running_iou/len(vloader)}")
				if best_loss > running_loss/len(vloader):
					best_loss = running_loss/len(vloader)
					torch.save(model.state_dict(), 'best_model.pth')
					print(f"saved model with loss {running_loss/len(vloader)}")

				print(f"valid loss {running_loss/len(vloader)} valid iou {running_iou/len(vloader)}")