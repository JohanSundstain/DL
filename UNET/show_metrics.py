import matplotlib.pyplot as plt


def parse_log(log_file):
	with open(log_file, "r") as f:
		logs = f.readlines()

	train_loss = []
	valid_loss = []
	valid_iou = []
	train_iou = []

	for line in logs:
		_, _, message = line.split(" - ")
		words = message.split(" ")

		if words[0] == "train":
			train_loss.append(float(words[4]))
			train_iou.append(float(words[6]))
		else:
			valid_loss.append(float(words[4]))
			valid_iou.append(float(words[6]))
	return train_loss, train_iou, valid_loss, valid_iou


def show_metrics(log_file):
	tl, ti, vl, vi = parse_log(log_file=log_file)
	fig, axes = plt.subplots(1, 4, figsize=(14,5))

	# len of train epoch and freq of valid
	te = len(tl)
	diff = te // len(vl) 
	ve = [diff*i for i in range(1,len(vl)+1)]
	te = range(te)
	print(len(ve), len(te))
	

	set_in_plot(axes[0], te, tl, "train loss")
	set_in_plot(axes[1], te, ti, "train IoU")
	set_in_plot(axes[2], ve, vl, "valid loss")
	set_in_plot(axes[3], ve, vi, "valid IoU")

	fig.suptitle("UNet")
	plt.savefig("test_images/metrics.png", dpi=300)
	plt.show()


def set_in_plot(ax, x, y, title):
	ax.plot(x,y)
	ax.set_title(title)
	ax.set_xlabel("epoch")
	ax.legend()
	ax.grid(True)


show_metrics("train.log")