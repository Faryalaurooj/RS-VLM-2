import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Classification Head
# -----------------------------
class ClassificationHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, num_classes, 1)
        )

    def forward(self, x):
        return self.net(x)


# -----------------------------
# Regression Head
# -----------------------------
class RegressionHead(nn.Module):
    """
    Predicts generic box regression.
    For oriented detection: (cx, cy, w, h, theta)
    """

    def __init__(self, in_channels, out_dim=5):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, out_dim, 1)
        )

    def forward(self, x):
        return self.net(x)


# -----------------------------
# Base Detection Head
# -----------------------------
class DetectionHead(nn.Module):
    """
    Shared detection head used by oriented detector.

    Output:
        cls_logits: (B, num_classes, H, W)
        bbox_reg:   (B, 5, H, W)
    """

    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.cls_head = ClassificationHead(in_channels, num_classes)
        self.reg_head = RegressionHead(in_channels, out_dim=5)

    def forward(self, x):
        cls_logits = self.cls_head(x)
        bbox_reg = self.reg_head(x)

        return {
            "cls_logits": cls_logits,
            "bbox_reg": bbox_reg
        }
