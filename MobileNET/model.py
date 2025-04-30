import torch
import torch.nn as nn
import torch.nn.init as init

class Depthwise(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride):
		super(Depthwise, self).__init__()

		self.conv = nn.Sequential(
			nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride, bias=False, groups=in_channels, padding=1),
			nn.BatchNorm2d(in_channels),
			nn.ReLU(),
			nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU()
		)

	def forward(self, x):
		return self.conv(x)


class MobNetNew(nn.Module):
	def __init__(self, num_classes):
		super(MobNetNew, self).__init__()

		self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, bias=False)
		self.bn = nn.BatchNorm2d(32)
		self.relu = nn.ReLU()
		# in_channels, out_channels, kernel_size, stride
		# confif for depthwise block
		config = [
			(32, 64, 3, 1),
			(64, 128, 3, 2),
			(128, 128, 3, 1),
			(128, 256, 3, 2),
			(256, 256, 3, 1),
			(256, 512, 3, 2),

			(512, 512, 3, 1),
			(512, 512, 3, 1),
			(512, 512, 3, 1),
			(512, 512, 3, 1),
			(512, 512, 3, 1),
			(512 ,1024, 3, 2),
			(1024, 1024, 3, 1)
		]

		self.depthwise_block = self._make_layer(config)
		self.avg_pool = nn.AvgPool2d(kernel_size=7)
		self.fc = nn.Linear(in_features=1024, out_features=num_classes)

	def _make_layer(self, config):
		layers = []

		for in_ch, out_ch, k_s, s in config:
			layers.append(
				Depthwise(in_channels=in_ch, out_channels=out_ch, kernel_size=k_s, stride=s)
			)

		return nn.Sequential(*layers)

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn(x)
		x = self.relu(x)

		x = self.depthwise_block(x)

		x = self.avg_pool(x)
		x = torch.flatten(x, 1)
		x = self.fc(x)

		return x