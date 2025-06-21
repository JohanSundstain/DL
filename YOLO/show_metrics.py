import matplotlib.pyplot as plt


def parse_log(log_file):
	with open(log_file, "r") as f:
		logs = f.readlines()

	train_loss = []
	valid_loss = []

	for line in logs:
		splitted = line.split(" ")

		if splitted[0] == "Train":
			train_loss.append(float(splitted[-1]))
		else:
			valid_loss.append(float(splitted[-1]))
	return train_loss, valid_loss


def show_metrics(log_file):
	tl, vl, = parse_log(log_file=log_file)
	fig, axes = plt.subplots(1, 2, figsize=(14,5))

	# len of train epoch and freq of valid
	te = len(tl)
	diff = te // len(vl) 
	ve = [diff*i for i in range(1,len(vl)+1)]
	te = range(te)
	print(len(ve), len(te))
	

	set_in_plot(axes[0], te, tl, "train loss")
	set_in_plot(axes[1], ve, vl, "valid loss")

	fig.suptitle("UNet")
	plt.savefig("metrics.png", dpi=300)
	plt.show()


def set_in_plot(ax, x, y, title):
	ax.plot(x,y)
	ax.set_title(title)
	ax.set_xlabel("epoch")
	ax.legend()
	ax.grid(True)


show_metrics("train.log")