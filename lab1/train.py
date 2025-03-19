import torch
import torch.nn as nn
from tqdm import tqdm
import torch
import torchmetrics.classification
from model import MobNetNew
import matplotlib.pyplot as plt

def set_in_plot(ax, x, y, title):
	ax.plot(x,y)
	ax.set_title(title)
	ax.set_xlabel("epoch")
	ax.legend()
	ax.grid(True)


def draw_metrics(dict, desc):
	fig, axes = plt.subplots(1, 4, figsize=(14,5))
	epoch = range(len(dict['loss']))

	set_in_plot(axes[0], epoch, dict['loss'], "Loss")
	set_in_plot(axes[1], epoch, dict['accuracy'], "Accuracy")
	set_in_plot(axes[2], epoch, dict['recall'], "Recall")
	set_in_plot(axes[3], epoch, dict['precision'], "Precision")

	fig.suptitle(desc)
	plt.savefig(f"experements/{desc}.png", dpi=300)
	plt.show()



def train(model, device, tloader, vloader, epoch=10, validation=1):

	train_results = {'loss': [], 'accuracy': [], 'precision':[], 'recall':[] }
	valid_results = {'loss': [], 'accuracy': [], 'precision':[], 'recall':[] }
	
	train_metrics = { 'accuracy': torchmetrics.classification.BinaryAccuracy(threshold=0.5).to(device=device),
 					  'precision':  torchmetrics.classification.BinaryPrecision(threshold=0.5).to(device=device),
					  'recall': torchmetrics.classification.BinaryRecall(threshold=0.5).to(device=device)}
	valid_metrics = { 'accuracy': torchmetrics.classification.BinaryAccuracy(threshold=0.5).to(device=device),
 					  'precision':  torchmetrics.classification.BinaryPrecision(threshold=0.5).to(device=device),
					  'recall': torchmetrics.classification.BinaryRecall(threshold=0.5).to(device=device)}
	
	loss_func = nn.BCEWithLogitsLoss()
	opt = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-8) 

	current_epoch = 0
	min_loss = 1e+6
	iterations = int(epoch/validation)
	for _ in range(iterations):
		model.train()
		for _ in range(validation):
			running_loss = 0.0
			current_epoch += 1
			for data, label in tqdm(tloader, desc=f'train {current_epoch}'):
				data = data.to(device)
				label = label.to(device)

				opt.zero_grad()
				output = model(data)
				loss = loss_func(output, label[:,None])
				loss.backward()
				opt.step()

				pred = torch.sigmoid(output)
				train_metrics['accuracy'].update(pred, label[:,None])
				train_metrics['recall'].update(pred, label[:,None])
				train_metrics['precision'].update(pred, label[:,None])
				running_loss += loss.item()

			train_results['loss'].append(running_loss / len(tloader))
			train_results['accuracy'].append(train_metrics['accuracy'].compute().item())
			train_results['recall'].append(train_metrics['recall'].compute().item())
			train_results['precision'].append(train_metrics['precision'].compute().item())

			train_metrics['accuracy'].reset()
			train_metrics['recall'].reset()
			train_metrics['precision'].reset()

			print(f"loss: {train_results['loss'][-1]:.2f}" +
		 	f" accuracy: {train_results['accuracy'][-1]:.2f}" +
			f" precision: {train_results['precision'][-1]:.2f}" +
			f" recall: {train_results['recall'][-1]:.2f}")
		
		model.eval()
		running_loss = 0.0
		with torch.no_grad():
			for data, label in tqdm(vloader, desc=f'validation '):
				data = data.to(device)
				label = label.to(device)

				output = model(data)
				loss = loss_func(output, label[:,None])
				
				pred = torch.sigmoid(output)
				valid_metrics['accuracy'].update(pred, label[:,None])
				valid_metrics['recall'].update(pred, label[:,None])
				valid_metrics['precision'].update(pred, label[:,None])
				
				running_loss += loss.item()

		avg_loss = running_loss / len(vloader) 
		if avg_loss < min_loss:
			min_loss = avg_loss
			torch.save(model.state_dict(), f"weights\\best_{epoch}.pth")

		valid_results['loss'].append(running_loss / len(vloader))
		valid_results['accuracy'].append(valid_metrics['accuracy'].compute().item())
		valid_results['recall'].append(valid_metrics['recall'].compute().item())
		valid_results['precision'].append(valid_metrics['precision'].compute().item())

		valid_metrics['accuracy'].reset()
		valid_metrics['recall'].reset()
		valid_metrics['precision'].reset()

		print(f"loss: {valid_results['loss'][-1]:.2f}" +
		 	f" accuracy: {valid_results['accuracy'][-1]:.2f}" +
			f" precision: {valid_results['precision'][-1]:.2f}" +
			f" recall: {valid_results['recall'][-1]:.2f}")
	
  
	torch.save(model.state_dict(), f"weights\\epoch_{epoch}.pth")
	draw_metrics(train_results, "Train")
	draw_metrics(valid_results, "Validation")

def test(model, path,  device, tloader):
	model.load_state_dict(torch.load(path))
	model.to(device=device)
	loss_func = nn.BCEWithLogitsLoss()
	
	test_metrics = {  'accuracy': torchmetrics.classification.BinaryAccuracy(threshold=0.5).to(device=device),
 					  'precision':  torchmetrics.classification.BinaryPrecision(threshold=0.5).to(device=device),
					  'recall': torchmetrics.classification.BinaryRecall(threshold=0.5).to(device=device)}

	model.eval()
	running_loss = 0.0
	with torch.no_grad():
		for data, label in tqdm(tloader, desc=f'test '):
			data = data.to(device)
			label = label.to(device)

			output = model(data)
			pred = torch.sigmoid(output)
			loss = loss_func(output, label[:,None])
			test_metrics['accuracy'].update(pred, label[:,None])
			test_metrics['recall'].update(pred, label[:,None])
			test_metrics['precision'].update(pred, label[:,None])
			
			running_loss += loss.item()

	avg_loss = running_loss / len(tloader) 
	print(f"loss: {avg_loss:.2f}" +
		 	f" accuracy: {test_metrics['accuracy'].compute().item():.2f}" +
			f" precision: {test_metrics['recall'].compute().item():.2f}" +
			f" recall: {test_metrics['precision'].compute().item():.2f}")
	