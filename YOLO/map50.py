import torch
import numpy as np
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.utils import draw_bounding_boxes
import matplotlib.pyplot as plt
from lgbt import lgbt
from PIL import Image
from torchmetrics.detection import MeanAveragePrecision

from dataset import TrainWithValidDataset  


if __name__ == "__main__":
	num_classes = 2  
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

	transforms = T.Compose([
		T.ToTensor(),
		T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])
	dataset = TrainWithValidDataset('dataset/train/_annotations.csv', 'dataset/train', transforms, k=0.0)
	data_loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

	anchor_sizes = ((16,),(32,), (64,), (128,), (256,))  
	aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes) 
	anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

	model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
		pretrained=False,
		rpn_anchor_generator=anchor_generator 
	)

	in_features = model.roi_heads.box_predictor.cls_score.in_features
	
	model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
	model.load_state_dict(torch.load("fasterrcnn_model_best_bs6.pth"))
	model.to(device)
	model.eval()

	metric = MeanAveragePrecision(
		iou_type="bbox",
		iou_thresholds=[0.5],  
		box_format="xyxy",
		class_metrics=True
	)
	with torch.no_grad():
		for images, targets, _ in lgbt(data_loader):
			images = list(img.to(device) for img in images)
			targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
              
			preds = model(images)
			
			formatted_preds = []
			for pred in preds:
				formatted_preds.append({
					"boxes": pred["boxes"].cpu(),
					"scores": pred["scores"].cpu(),
					"labels": pred["labels"].cpu()
				})
			
			formatted_targets = []
			for target in targets:
				formatted_targets.append({
					"boxes": target["boxes"].cpu(),
					"labels": target["labels"].cpu()
				})
			
			metric.update(formatted_preds, formatted_targets)

	result = metric.compute()
	print(f"mAP@50: {result['map_50'].item():.4f}")
