import torch
import torch.nn as nn
import torch.nn.init as init


import torch
import torch.nn as nn
import torch.nn.functional as F

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        hidden_dim = in_channels * expand_ratio
        self.use_residual = stride == 1 and in_channels == out_channels
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),

            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
    
    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobNet(nn.Module):
    def __init__(self, num_classes=1):
        super(MobNet, self).__init__()
        
        # Конфигурация блоков: (in_channels, out_channels, stride, expand_ratio, repeat)
        config = [
            (32, 16, 1, 1, 1),
            (16, 24, 2, 6, 2),
            (24, 32, 2, 6, 3),
            (32, 64, 2, 6, 4),
            (64, 96, 1, 6, 3),
            (96, 160, 2, 6, 3),
            (160, 320, 1, 6, 1),
        ]
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )
        
        self.blocks = self._make_layers(config)
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(320, 1280, kernel_size=1, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True)
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1280, num_classes)

    def _make_layers(self, config):
        layers = []
        for in_ch, out_ch, stride, expand_ratio, repeat in config:
            for i in range(repeat):
                layers.append(InvertedResidual(
                    in_channels=in_ch if i == 0 else out_ch, 
                    out_channels=out_ch, 
                    stride=stride if i == 0 else 1, 
                    expand_ratio=expand_ratio
                ))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.blocks(x)
        x = self.conv2(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
