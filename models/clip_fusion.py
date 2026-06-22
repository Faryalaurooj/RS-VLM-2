#(CLIP-guided semantic alignment module for RS-VLM-style enhancement), This module injects language-aware semantic guidance into visual features to:reduce domain gap in remote sensing, improve small-object semantics,align features with CLIP embedding space, stabilize detection in cluttered scenes,This module injects language-aware semantic guidance into visual features to:reduce domain gap in remote sensing,improve small-object semantics,align features with CLIP embedding space stabilize detection in cluttered scenes

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPProjector(nn.Module):
    """
    Projects visual features into CLIP embedding space
    """
    def __init__(self, in_dim, out_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)


class CLIPAlignmentLoss(nn.Module):
    """
    Contrastive alignment loss between vision and text embeddings
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, vision_emb, text_emb):
        vision_emb = F.normalize(vision_emb, dim=-1)
        text_emb = F.normalize(text_emb, dim=-1)

        logits = vision_emb @ text_emb.T / self.temperature

        labels = torch.arange(vision_emb.size(0)).to(vision_emb.device)

        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.T, labels)

        return (loss_i + loss_t) / 2


class CLIPFusion(nn.Module):
    """
    Fuses CLIP semantic embeddings with RS visual features

    Input:
        feature map (B, C, H, W)
        optional text embeddings (B, D)

    Output:
        enhanced feature map
        optional alignment loss
    """

    def __init__(self, in_channels, clip_dim=512):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.proj = MLPProjector(in_channels, clip_dim)

        self.fuse = nn.Sequential(
            nn.Linear(in_channels + clip_dim, in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, in_channels)
        )

        self.loss_fn = CLIPAlignmentLoss()

    def forward(self, x, text_emb=None):
        """
        Args:
            x: feature map (B, C, H, W)
            text_emb: (B, D) optional CLIP text embedding
        """

        B, C, H, W = x.shape

        pooled = self.pool(x).view(B, C)
        vision_emb = self.proj(pooled)

        loss = 0.0

        if text_emb is not None:
            loss = self.loss_fn(vision_emb, text_emb)

        # broadcast fusion
        fused = torch.cat([pooled, vision_emb], dim=1)
        fused = self.fuse(fused).view(B, C, 1, 1)

        out = x + fused

        return out, loss
