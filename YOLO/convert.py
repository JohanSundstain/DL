import pandas as pd
import os
import shutil

def save(ann, src_img, dst_img, dst_ann):
	for key, value in ann:
		name, ext = os.path.splitext(key)
		ann_line = ""
		for obj in value:
			class_idx = 0
			w,h = int(obj[0]), int(obj[1])
			xmin,ymin = int(obj[3]), int(obj[4])
			xmax,ymax = int(obj[5]), int(obj[6])

			x_center = (xmin + xmax) / 2 / w
			y_center = (ymin + ymax) / 2 / h
			print(xmax-xmin)
			print(ymax-ymin)
			bbox_w = float((xmax - xmin) / w)
			bbox_h = float((ymax - ymin) / h)

			ann_line += f'{class_idx} {x_center} {y_center} {bbox_w} {bbox_h}\n'

		path_ann = os.path.join(dst_ann, f'{name}.txt')
		path_img = os.path.join(dst_img, f'{name}.jpg')
		path_src_img = os.path.join(src_img, f'{name}.jpg')
		shutil.copy(path_src_img, path_img)
		with open(path_ann, "w") as f:
			f.write(ann_line)


files_with_ann = {}

with open("dataset/train/_annotations.csv") as f:
	ann = f.readlines()
	ann = ann[1::]

for line in ann:
	if line == "\n":
		continue
	splitted_line = line.split(",")
	file_name = splitted_line[0]
	if file_name not in files_with_ann:
		files_with_ann[file_name] = []

	files_with_ann[file_name].append(splitted_line[1::])

target_images_train = "yolo_dataset/images/train"
target_images_val = "yolo_dataset/images/val"
target_labels_train = "yolo_dataset/labels/train"
target_labels_val = "yolo_dataset/labels/val"

source_images_path = 'dataset/train'
image_list = os.listdir(source_images_path)

os.makedirs(target_images_train,exist_ok=True)
os.makedirs(target_images_val,exist_ok=True)
os.makedirs(target_labels_train,exist_ok=True)
os.makedirs(target_labels_val,exist_ok=True)

ann_list = list(files_with_ann.items())
n = int(0.9 * len(ann_list))
train = ann_list[:n]
val = ann_list[n:]

save(val, source_images_path, target_images_val, target_labels_val)
save(train, source_images_path, target_images_train, target_labels_train)
