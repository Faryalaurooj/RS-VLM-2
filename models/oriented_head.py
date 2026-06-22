#(Oriented Detection Head for DOTA-style rotated object detection) This head is designed for: rotated bounding boxes (x, y, w, h, θ), DOTA dataset compatibility. angle-aware regression stability, small-object sensitivity

This head is designed for:

rotated bounding boxes (x, y, w, h, θ)
DOTA dataset compatibility
angle-aware regression stability
small-object sensitivity


import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Utility: Smooth L1 Loss
# -----------------------------
class SmoothL1(nn.Module):
    def forward(self, pred, target):
        return F.smooth_l1_loss(pred, target)


# -----------------------------
# Angle-aware loss (cyclic)
# -----------------------------
class AngleLoss(nn.Module):
    """
    Handles periodicity of angle (theta).
    Ensures stability for rotated boxes.
    """
    def forward(self, pred_theta, target_theta):
        diff = torch.sin(pred_theta - target_theta)
        return torch.mean(diff ** 2)


# -----------------------------
# Oriented Bounding Box Head
# -----------------------------
class OrientedDetectionHead(nn.Module):
    """
    Predicts:
        - class scores
        - rotated bbox: (cx, cy, w, h, theta)
    """

    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.num_classes = num_classes

        # shared conv trunk
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

        # classification branch
        self.cls = nn.Conv2d(in_channels, num_classes, 1)

        # regression branch (5 params for rotated box)
        self.reg = nn.Conv2d(in_channels, 5, 1)

        self.cls_loss_fn = nn.CrossEntropyLoss()
        self.box_loss_fn = SmoothL1()
        self.angle_loss_fn = AngleLoss()

    def forward(self, x, targets=None):
        """
        Args:
            x: feature map (B, C, H, W)
            targets: optional dict for supervised loss

        Returns:
            predictions + optional losses
        """

        feat = self.conv(x)

        cls_logits = self.cls(feat)
        bbox_reg = self.reg(feat)

        outputs = {
            "cls_logits": cls_logits,
            "bbox_reg": bbox_reg
        }

        losses = {}

        if targets is not None:
            # classification loss
            losses["cls_loss"] = self.cls_loss_fn(
                cls_logits,
                targets["labels"]
            )

            # bbox regression loss
            pred_box = bbox_reg[..., :4]
            pred_theta = bbox_reg[..., 4]

            tgt_box = targets["boxes"][..., :4]
            tgt_theta = targets["boxes"][..., 4]

            losses["box_loss"] = self.box_loss_fn(pred_box, tgt_box)
            losses["angle_loss"] = self.angle_loss_fn(pred_theta, tgt_theta)

            losses["total_loss"] = (
                losses["cls_loss"] +
                losses["box_loss"] +
                0.5 * losses["angle_loss"]
            )

        return outputs, losses
