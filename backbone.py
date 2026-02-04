import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

class FPNBackbone(nn.Module):
    def __init__(self, out_channels=256):
        super().__init__()
        backbone = resnet50(weights="IMAGENET1K_V1")

        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )

        self.c2 = backbone.layer1
        self.c3 = backbone.layer2
        self.c4 = backbone.layer3
        self.c5 = backbone.layer4

        self.lateral = nn.ModuleList([
            nn.Conv2d(256, out_channels, 1),
            nn.Conv2d(512, out_channels, 1),
            nn.Conv2d(1024, out_channels, 1),
            nn.Conv2d(2048, out_channels, 1),
        ])

        self.output = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        ])

    def forward(self, x):
        x = self.stem(x)
        c2 = self.c2(x)
        c3 = self.c3(c2)
        c4 = self.c4(c3)
        c5 = self.c5(c4)

        p5 = self.lateral[3](c5)
        p4 = self.lateral[2](c4) + F.interpolate(p5, scale_factor=2)
        p3 = self.lateral[1](c3) + F.interpolate(p4, scale_factor=2)
        p2 = self.lateral[0](c2) + F.interpolate(p3, scale_factor=2)

        feats = [
            self.output[0](p2),
            self.output[1](p3),
            self.output[2](p4),
            self.output[3](p5),
        ]
        return feats

