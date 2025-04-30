import os
import shutil

PATH_DIR = r"dataset\water_v2\water_v2"
TRAIN_DIRS = r"train.txt"
VALID_DIRS = r"val.txt"
PATH_ANN = r"Annotations"
PATH_IMG = r"JPEGImages"

def extract_image_mask(dirs):
	masks_list = []
	images_list = []
	ann = os.path.join(PATH_DIR, PATH_ANN)
	img = os.path.join(PATH_DIR, PATH_IMG)
	for dir in dirs:
		if dir == "\n":
			continue
		
		path_files_ann = os.path.join(ann, dir[:-1:])
		path_files_img = os.path.join(img, dir[:-1:])
		list_of_ann = os.listdir(path_files_ann)
		list_of_img = os.listdir(path_files_img)

		for file in list_of_img:
			file_with_ext = os.path.splitext(file)[0]+'.png'
			if file_with_ext in list_of_ann:
				path_to_image = os.path.join(path_files_img, file)
				images_list.append(path_to_image)

		for file in list_of_ann:
			if os.path.splitext(file)[1] == '.png':
				path_to_image = os.path.join(path_files_ann, file)
				masks_list.append(path_to_image)


	return images_list, masks_list

def main():
	train_dirs = os.path.join(PATH_DIR, TRAIN_DIRS)
	valid_dirs = os.path.join(PATH_DIR, VALID_DIRS)

	ann = os.path.join(PATH_DIR, PATH_ANN)
	img = os.path.join(PATH_DIR, PATH_IMG)

	with open(train_dirs, "r") as f:
		train_dirs = f.readlines()
	
	with open(valid_dirs, "r") as f:
		valid_dirs = f.readlines()

	images_train, masks_train = extract_image_mask(train_dirs)
	images_valid, masks_valid = extract_image_mask(valid_dirs)

	with open("train.txt", "w") as f:
		for image, mask in zip(images_train, masks_train):
			f.write(f'{image} {mask}\n')

	with open("valid.txt", "w") as f:
		for image, mask in zip(images_valid, masks_valid):
			f.write(f'{image} {mask}\n')

	

if __name__ == "__main__":
	main()

