#(Full SGSM + CLIP + Oriented Detection integration model)

#This is the central model file that wires everything together:

#Backbone feature extractor (placeholder CNN)
#SGSM (self-supervised geometric module)
#CLIPFusion (semantic alignment)
#OrientedDetectionHead (final predictions)

import torch
import torch.nn as nn

from models.sgsm import SGSM
from models.oriented_head import OrientedDetectionHead
from models.clip_fusion import CLIPFusion


# -----------------------------
# Simple Backbone (placeholder)
# Replace with ResNet / Swin / ConvNeXt
# -----------------------------
class SimpleBackbone(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 7, stride=2, padding=3),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(base_channels, base_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(base_channels, base_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )

        self.out_channels = base_channels

    def forward(self, x):
        return self.net(x)


# -----------------------------
# Full SGSM Detector
# -----------------------------
class SGSMDetector(nn.Module):
    """
    End-to-end model:

    Image → Backbone → SGSM → CLIPFusion → Oriented Head
    """

    def __init__(self, num_classes=15, use_clip=True):
        super().__init__()

        self.backbone = SimpleBackbone()
        self.sgsm = SGSM(self.backbone.out_channels)

        self.use_clip = use_clip
        if use_clip:
            self.clip_fusion = CLIPFusion(self.backbone.out_channels)

        self.head = OrientedDetectionHead(
            in_channels=self.backbone.out_channels,
            num_classes=num_classes
        )

    def forward(self, x, targets=None, rotated_x=None, text_emb=None):
        """
        Args:
            x: input image
            targets: supervision dict (optional)
            rotated_x: augmented image for SGSM self-supervision
            text_emb: CLIP text embeddings (optional)
        """

        # 1. Backbone features
        feat = self.backbone(x)

        # 2. SGSM (self-supervised geometric refinement)
        feat, sgsm_loss = self.sgsm(feat, rotated_x)

        # 3. CLIP fusion (semantic alignment)
        clip_loss = 0.0
        if self.use_clip and text_emb is not None:
            feat, clip_loss = self.clip_fusion(feat, text_emb)

        # 4. Detection head
        outputs, det_losses = self.head(feat, targets)

        # 5. Combine losses
        total_loss = 0.0

        if isinstance(det_losses, dict) and len(det_losses) > 0:
            total_loss += det_losses.get("total_loss", 0.0)

        total_loss += sgsm_loss + clip_loss

        losses = {
            "sgsm_loss": sgsm_loss,
            "clip_loss": clip_loss,
            "det_losses": det_losses,
            "total_loss": total_loss
        }

        return outputs, losses
