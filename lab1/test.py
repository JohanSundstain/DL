import cv2
import albumentations as A
import matplotlib.pyplot as plt

img_path = r'dataset/test/hotdog/1501.jpg'

img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

transform = A.Compose(
	[
		A.Resize(256, 256),        # Изменение размера
		A.HorizontalFlip(p=0.5),   # Горизонтальный флип (50% вероятности)
		A.RandomBrightnessContrast(p=1),
		A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])
aug = transform(image=img_rgb)
aug_img = aug["image"]
print(aug_img)
plt.imshow(aug_img)
plt.show()
