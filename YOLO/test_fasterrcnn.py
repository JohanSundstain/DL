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

from dataset import TrainWithValidDataset  

num_classes = 2  
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

transforms = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
dataset = TrainWithValidDataset('dataset/train/_annotations.csv', 'dataset/train', transforms)
data_loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)), num_workers=4)

anchor_sizes = ((16,),(32,), (64,), (128,), (256,))  
aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes) 
anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
    pretrained=False,
    rpn_anchor_generator=anchor_generator 
)

in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model.to(device)

model.load_state_dict(torch.load("fasterrcnn_model_best_bs6.pth"))
model.to(device)
model.eval()
with torch.no_grad():
	image, _, path = dataset[0]
	image = image.to(device)
	output = model([image])
	boxes = output[0]['boxes'][output[0]['scores'] > 0.5]
	print(len(boxes))

	img = Image.open(path).convert("RGB")

	img_np = np.array(img)  

	img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)  

	result_img = draw_bounding_boxes(img_tensor, boxes, width=6)
	plt.imshow(result_img.permute(1, 2, 0))
	plt.show()

