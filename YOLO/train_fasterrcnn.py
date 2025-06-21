import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from lgbt import lgbt
from dataset import TrainWithValidDataset
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s', filename='train.log', filemode='w')

num_classes = 2  
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

transforms = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
dataset = TrainWithValidDataset('train/_annotations.csv', 'train', transforms)
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

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

num_epochs = 70
min_val_loss = float('inf')

for epoch in range(num_epochs):
    # ======== ТРЕНИРОВКА ========
    model.train()
    dataset.train()
    train_loss = 0.0
    num_train_batches = 0
    
    for images, targets in lgbt(data_loader, desc=f'train {epoch}'):
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
              
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        train_loss += losses.item()
        num_train_batches += 1

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
    
    avg_train_loss = train_loss / num_train_batches
    logging.info(f'Epoch {epoch} | Train loss: {avg_train_loss:.4f}')
    print(f"Epoch {epoch} | Train loss: {avg_train_loss:.4f}")

    lr_scheduler.step()
    
    # ======== ВАЛИДАЦИЯ ========
    dataset.valid()
    val_loss = 0.0
    num_val_batches = 0
    
    with torch.no_grad():
        for images, targets in lgbt(data_loader, desc=f'valid {epoch}'):
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            val_loss += losses.item()
            num_val_batches += 1
    
    avg_val_loss = val_loss / num_val_batches
    logging.info(f'Epoch {epoch} | Val loss: {avg_val_loss:.4f}')
    print(f"Epoch {epoch} | Val loss: {avg_val_loss:.4f}")
    
    if avg_val_loss < min_val_loss:
        min_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'fasterrcnn_model_best.pth')
        print(f"Best model saved with val loss: {avg_val_loss:.4f}")

torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': avg_val_loss,
}, 'fasterrcnn_model_final.pth')
print("Training complete! Final model saved.")