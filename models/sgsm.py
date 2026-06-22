
#Self-supervised geometric consistency ✔ Multi-scale receptive fusion (1×1, 3×3, 5×5) ✔ Plug-and-play into any backbone ✔ Returns both: enhanced features, auxiliary self-supervised loss


import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU()
        )

    def forward(self, x):
        return self.block(x)


class RotationConsistencyLoss(nn.Module):
    """
    Self-supervised rotation consistency:
    feature(x) ≈ rotate(feature(rot(x)))
    """
    def forward(self, f1, f2):
        return F.mse_loss(f1, f2)


class SGSM(nn.Module):
    """
    Self-supervised Geometric Structure Module (SGSM)

    Inputs:
        feature map from backbone

    Outputs:
        enhanced feature map + self-supervised loss
    """

    def __init__(self, channels):
        super().__init__()

        self.project = ConvBNAct(channels, channels)

        # multi-scale context
        self.branch1 = ConvBNAct(channels, channels, k=1, p=0)
        self.branch3 = ConvBNAct(channels, channels, k=3, p=1)
        self.branch5 = ConvBNAct(channels, channels, k=5, p=2)

        self.fuse = nn.Conv2d(channels * 3, channels, 1)

        self.rot_loss = RotationConsistencyLoss()

    def forward(self, x, rotated_x=None):
        """
        rotated_x: augmented version of input feature (rotated image branch)
        """

        x_proj = self.project(x)

        b1 = self.branch1(x_proj)
        b3 = self.branch3(x_proj)
        b5 = self.branch5(x_proj)

        fused = torch.cat([b1, b3, b5], dim=1)
        out = self.fuse(fused)

        loss = 0.0
        if rotated_x is not None:
            loss = self.rot_loss(out, rotated_x)

        return out, loss
