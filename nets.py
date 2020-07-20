import os
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class UNet(nn.Module):
    def __init__(self, inC, midC, kernel_size=3, padding=1):
        super(UNet, self).__init__()

        self.d1 = UNetDown(inC, midC, kernel_size, padding)
        self.d2 = UNetDown(midC, midC*2, kernel_size, padding)
        self.m1 = UNetMiddle(midC*2, kernel_size, padding)

        self.u1 = UNetUp(midC*4, midC, kernel_size, padding)
        self.u2 = UNetUp(midC*2, midC, kernel_size, padding)

        self.c1 = UNetConv(midC, midC, kernel_size, padding)
        self.c2 = nn.Conv2d(midC, inC, kernel_size=1, padding=0)

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(F.max_pool2d(d1, kernel_size=2))
        m1 = self.m1(F.max_pool2d(d2, kernel_size=2))

        u1 = self.u1(d2, F.upsample(m1,scale_factor=2, mode='nearest'))
        u2 = self.u2(d1, F.upsample(u1,scale_factor=2, mode='nearest'))

        x = self.c1(u2)
        x = self.c2(x)
        return x

class UNetDown(nn.Module):
    def __init__(self, inC, outC, kernel_size, padding):
        super(UNetDown, self).__init__()

        self.conv1 = nn.Conv2d(inC, outC, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(outC)
        self.conv2 = nn.Conv2d(outC, outC, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(outC)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x

class UNetMiddle(nn.Module):
    def __init__(self, inC, kernel_size, padding):
        super(UNetMiddle, self).__init__()

        self.conv1 = nn.Conv2d(inC, inC*2, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(inC*2)
        self.conv2 = nn.Conv2d(inC*2, inC, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(inC)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x

class UNetUp(nn.Module):
    def __init__(self, inC, outC, kernel_size, padding):
        super(UNetUp, self).__init__()

        self.conv1 = nn.Conv2d(inC, outC, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(outC)
        self.conv2 = nn.Conv2d(outC, outC, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(outC)

    def forward(self, x1, x2):
        x1, x2 = self.adjust(x1, x2)
        x = torch.cat([x1, x2], 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x

    def adjust(self, x1, x2):
        if (x1.size(2) != x2.size(2)) or (x1.size(3) != x2.size(3)):
            min2 = min(x2.size(2), x1.size(2))
            min3 = min(x2.size(3), x1.size(3))
            x1 = x1[:, :, :min2, :min3]
            x2 = x2[:, :, :min2, :min3]
        return x1, x2

class UNetConv(nn.Module):
    def __init__(self, inC, outC, kernel_size, padding):
        super(UNetConv, self).__init__()

        self.conv1 = nn.Conv2d(inC, outC, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(outC)
        self.conv2 = nn.Conv2d(outC, outC, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(outC)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x
