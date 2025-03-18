# separate dataset to train, valid and test part
import os
import shutil
import numpy as np

# train + valid + test = 100% of dataset
train_dir = 'dataset/train'
valid_dir = 'dataset/valid'


def separate():
	valid_hd = f'{valid_dir}/hotdog'
	valid_nhd = f'{valid_dir}/nothotdog'
	train_hd = f'{train_dir}/hotdog'
	train_nhd = f'{train_dir}/nothotdog'

	try:
		os.makedirs(valid_nhd)
		os.makedirs(valid_hd)
	except FileExistsError:
		print("The valid dir already exists")
		return

	hd = np.array(os.listdir(f'{train_dir}/hotdog')) 
	nhd = np.array(os.listdir(f'{train_dir}/nothotdog'))	

	np.random.shuffle(hd)
	np.random.shuffle(nhd)

	for i in range(int(hd.size * 0.1)):
		src = os.path.join(train_hd,hd[i])
		dst = os.path.join(valid_hd,hd[i])
		shutil.move(src, dst)
		src = os.path.join(train_nhd,nhd[i])
		dst = os.path.join(valid_nhd,nhd[i])
		shutil.move(src,dst)


separate()