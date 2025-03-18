import torch
import torch.nn as nn
import torch.nn.init as init


class MobNet(nn.Module):
	def __init__(self, num_classes=1):
		super(MobNet, self).__init__()
		self.conv_3x3x3x32_s2 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1)

		self.conv_1x1x64x128 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=1)
		self.conv_1x1x32x64 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1)
		self.conv_1x1x128x128 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1,stride=1)
		self.conv_1x1x128x256 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=1)
		self.conv_1x1x256x256 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1)
		self.conv_1x1x256x512 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=1)
		self.conv_1x1x512x512 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1)
		self.conv_1x1x512x1024 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1, stride=1)
		self.conv_1x1x1024x1024 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1, stride=1)

		self.conv_d3x3x32 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, groups=32, padding=1)
		self.conv_d3x3x64_s2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, groups=64, padding=1)
		self.conv_d3x3x128 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, groups=128, padding=1)
		self.conv_d3x3x128_s2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, groups=128, padding=1)
		self.conv_d3x3x256 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, groups=256,padding=1)
		self.conv_d3x3x256_s2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, groups=256, padding=1)
		self.conv_d3x3x512 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, groups=512, padding=1)
		self.conv_d3x3x512_s2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, groups=512, padding=1)
		self.conv_d3x3x1024 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, groups=1024,padding=1)

		self.avg_pool = nn.AvgPool2d(kernel_size=7)		
		self.fc = nn.Linear(in_features=1024, out_features=1000)
		self.final_fc = nn.Linear(in_features=1000, out_features=num_classes)
		# functions
		self.relu = nn.ReLU()
		self.sigm = nn.Sigmoid()
		self.bn_32 = nn.BatchNorm2d(num_features=32)
		self.bn_64 = nn.BatchNorm2d(num_features=64)
		self.bn_128 = nn.BatchNorm2d(num_features=128)
		self.bn_256 = nn.BatchNorm2d(num_features=256)
		self.bn_512 = nn.BatchNorm2d(num_features=512)
		self.bn_1024 = nn.BatchNorm2d(num_features=1024)

	def conv(self, x, conv, bn):
		x = conv(x)
		x = bn(x)
		x = self.relu(x)
		return x
	
	def depthwise(self, x, conv_d, conv_1x1, bn_d, bn_1x1):
		x = conv_d(x)
		x = bn_d(x)
		x = self.relu(x)
		x = conv_1x1(x)
		x = bn_1x1(x)
		x = self.relu(x)
		return x 

	def forward(self, x):
		x = self.conv(x, self.conv_3x3x3x32_s2, self.bn_32) #1
		x = self.depthwise(x, self.conv_d3x3x32, self.conv_1x1x32x64, self.bn_32, self.bn_64)  #2
		x = self.depthwise(x, self.conv_d3x3x64_s2, self.conv_1x1x64x128, self.bn_64, self.bn_128) #3
		x = self.depthwise(x, self.conv_d3x3x128, self.conv_1x1x128x128, self.bn_128, self.bn_128) #4
		x = self.depthwise(x, self.conv_d3x3x128_s2, self.conv_1x1x128x256, self.bn_128, self.bn_256) #5
		x = self.depthwise(x, self.conv_d3x3x256, self.conv_1x1x256x256, self.bn_256, self.bn_256) #6 
		x = self.depthwise(x, self.conv_d3x3x256_s2, self.conv_1x1x256x512, self.bn_256, self.bn_512) #7 

		x = self.depthwise(x, self.conv_d3x3x512, self.conv_1x1x512x512, self.bn_512, self.bn_512) #8 
		x = self.depthwise(x, self.conv_d3x3x512, self.conv_1x1x512x512, self.bn_512, self.bn_512) #9 
		x = self.depthwise(x, self.conv_d3x3x512, self.conv_1x1x512x512, self.bn_512, self.bn_512) #10 
		x = self.depthwise(x, self.conv_d3x3x512, self.conv_1x1x512x512, self.bn_512, self.bn_512) #11 
		x = self.depthwise(x, self.conv_d3x3x512, self.conv_1x1x512x512, self.bn_512, self.bn_512) #12 

		x = self.depthwise(x, self.conv_d3x3x512_s2, self.conv_1x1x512x1024, self.bn_512, self.bn_1024) #13 
		x = self.depthwise(x, self.conv_d3x3x1024, self.conv_1x1x1024x1024, self.bn_1024, self.bn_1024) #14 

		x = self.avg_pool(x) #15
		x = torch.flatten(x,1)
		x = self.fc(x) 
		x = self.relu(x)
		x = self.final_fc(x)
		x = self.sigm(x)
		return x
