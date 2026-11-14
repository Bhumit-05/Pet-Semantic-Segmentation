import torch
import torch.nn as nn

from config import NUM_CLASSES


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dilation_1: int = 1, dilation_2: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=dilation_1, dilation=dilation_1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=dilation_2, dilation=dilation_2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ControlledUNet(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, use_dilated_bottleneck: bool = False):
        super().__init__()
        self.enc1 = ConvBlock(3, 32)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        if use_dilated_bottleneck:
            self.bottleneck = ConvBlock(64, 128, dilation_1=2, dilation_2=4)
        else:
            self.bottleneck = ConvBlock(64, 128, dilation_1=1, dilation_2=1)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(64, 32)

        self.head = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))

        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)