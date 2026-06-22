import torch
import torch.nn as nn


# -----------------------------
# Basic Conv Block
# -----------------------------
class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


# -----------------------------
# Residual Block
# -----------------------------
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.conv1 = ConvBNAct(channels, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)

        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.bn(out)

        out += identity
        return self.act(out)


# -----------------------------
# Backbone Network
# -----------------------------
class Backbone(nn.Module):
    """
    Lightweight CNN backbone for SGSM-OD.

    Output:
        feature map (B, C, H/8, W/8)
    """

    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()

        self.stem = nn.Sequential(
            ConvBNAct(in_channels, base_channels, stride=2),
            ConvBNAct(base_channels, base_channels, stride=2),
        )

        self.stage2 = nn.Sequential(
            ResidualBlock(base_channels),
            ResidualBlock(base_channels),
        )

        self.stage3 = nn.Sequential(
            ConvBNAct(base_channels, base_channels * 2, stride=2),
            ResidualBlock(base_channels * 2),
        )

        self.stage4 = nn.Sequential(
            ConvBNAct(base_channels * 2, base_channels * 2, stride=2),
            ResidualBlock(base_channels * 2),
        )

        self.out_channels = base_channels * 2

    def forward(self, x):
        x = self.stem(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        return x
