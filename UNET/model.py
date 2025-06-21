import torch
import torch.nn as nn
import torchvision.transforms as transorms


class EncoderBlock(nn.Module):
	def __init__(self, in_1, out_1, in_2, out_2):
		super(EncoderBlock, self).__init__()
		self.layer = nn.Sequential(
		nn.Conv2d(in_channels=in_1, out_channels=out_1, kernel_size=3, padding=1, bias=False),
		nn.BatchNorm2d(out_1),
		nn.ReLU(inplace=True),
		nn.Conv2d(in_channels=in_2, out_channels=out_2, kernel_size=3, padding=1, bias=False),
		nn.BatchNorm2d(out_2),
		nn.ReLU(inplace=True))
		self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

	def forward(self, x): 
		skip = self.layer(x)
		x = self.max_pool(skip)
		return skip, x

class DecoderBlock(nn.Module):
	def __init__(self, in_1, out_1, in_2, out_2):
		super(DecoderBlock, self).__init__()
		self.prev = None
		self.upconv = nn.ConvTranspose2d(in_channels=in_1, out_channels=in_1//2, kernel_size=4, stride=2, padding=1, bias=False)	
		self.layer = nn.Sequential(
			nn.Conv2d(in_channels=in_1, out_channels=in_1//2, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(in_1//2),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_channels=in_2, out_channels=out_2, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_2),
			nn.ReLU(inplace=True))

	def _center_crop(self, tensor, target_size):
		_, _, h, w = tensor.size()
		th, tw = target_size
		dh = (h - th) // 2
		dw = (w - tw) // 2
		return tensor[:, :, dh:dh+th, dw:dw+tw]		

	def skip(self, prev):
		self.prev = prev
	
	def forward(self, x):
		x = self.upconv(x)
		concat = torch.cat([x, self.prev], dim=1)
		x = self.layer(concat)
		return x

class BottleNeck(nn.Module):
	def __init__(self, in_1, out_1, in_2, out_2):
		super(BottleNeck, self).__init__()
		self.layer = nn.Sequential(
			nn.Conv2d(in_channels=in_1,out_channels=out_1,kernel_size=3,padding=1,bias=False),
			nn.BatchNorm2d(out_1),
			nn.ReLU(inplace=True),
			nn.Dropout2d(p=0.3),
			nn.Conv2d(in_channels=in_2,out_channels=out_2,kernel_size=3,padding=1,bias=False),
			nn.BatchNorm2d(out_2),
			nn.ReLU(inplace=True),
			nn.Dropout2d(p=0.3))
	
	def forward(self, x):
		x = self.layer(x)
		return x

class Unet(nn.Module):
	def __init__(self, num_classes):
		super(Unet, self).__init__()
		# (in conv1, out conv1, in conv2, out conv2)
		self.num_classes = num_classes
		config_encoder =[
			(3, 64, 64, 64),
			(64,128,128,128),
			(128,256,256,256),
			(256,512,512,512)]
		
		config_decoder = [
			(1024, 512, 512, 512),
			(512, 256, 256, 256),
			(256, 128, 128, 128),
			(128, 64, 64, 64)]
		
		self.encoder_layers = self._make_layers(config_encoder, EncoderBlock)
		self.botleneck = BottleNeck(512, 1024, 1024, 1024)
		self.decoder_layers = self._make_layers(config_decoder, DecoderBlock)
		self.last_layer = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1, padding=0, bias=False)
		
			
	def _make_layers(self, config, type):
		layers = []
		for in_1, out_1, in_2, out_2 in config:
			layers.append(type(in_1, out_1, in_2, out_2))
		return nn.ModuleList(layers)
		
	
	def forward(self, x):
		first_layer, x = self.encoder_layers[0](x)
		second_layer, x = self.encoder_layers[1](x)
		third_layer, x = self.encoder_layers[2](x)
		forth_layer, x = self.encoder_layers[3](x)
		bottleneck = self.botleneck(x)

		self.decoder_layers[0].skip(forth_layer)
		x = self.decoder_layers[0](bottleneck)

		self.decoder_layers[1].skip(third_layer)
		x = self.decoder_layers[1](x)

		self.decoder_layers[2].skip(second_layer)
		x = self.decoder_layers[2](x)

		self.decoder_layers[3].skip(first_layer)
		x = self.decoder_layers[3](x)

		x = self.last_layer(x)

		return x
