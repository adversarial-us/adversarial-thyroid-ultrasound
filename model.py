"""
U-Net segmentation model for thyroid nodule segmentation.

Architecture: 4-stage encoder (32, 64, 128, 256 features), 512-feature bottleneck,
symmetric decoder with skip connections. Batch normalization and ReLU throughout.
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, ci, co, drop=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ci, co, 3, padding=1, bias=False),
            nn.BatchNorm2d(co),
            nn.ReLU(True),
            nn.Dropout2d(drop) if drop > 0 else nn.Identity(),
            nn.Conv2d(co, co, 3, padding=1, bias=False),
            nn.BatchNorm2d(co),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, features=(32, 64, 128, 256), drop=0.1):
        super().__init__()
        self.encs = nn.ModuleList()
        self.decs = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)
        c = 1
        for f in features:
            self.encs.append(DoubleConv(c, f, drop))
            c = f
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2, drop)
        for f in reversed(features):
            self.decs.append(nn.ConvTranspose2d(f * 2, f, 2, stride=2))
            self.decs.append(DoubleConv(f * 2, f))
        self.head = nn.Conv2d(features[0], 1, 1)

    def forward(self, x):
        skips = []
        for enc in self.encs:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skips = skips[::-1]
        for i in range(0, len(self.decs), 2):
            x = self.decs[i](x)
            s = skips[i // 2]
            if x.shape != s.shape:
                x = nn.functional.interpolate(x, size=s.shape[2:])
            x = self.decs[i + 1](torch.cat([s, x], 1))
        return self.head(x)
