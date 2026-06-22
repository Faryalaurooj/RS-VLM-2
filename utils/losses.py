import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np


# -----------------------------
# Focal Loss (classification)
# -----------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        loss = self.alpha * (1 - pt) ** self.gamma * ce
        return loss.mean()


# -----------------------------
# IoU helper (polygon-based)
# -----------------------------
def rbox_to_poly(rbox):
    """
    Convert (cx, cy, w, h, theta) → 4-point polygon
    """
    cx, cy, w, h, theta = rbox

    c = np.cos(theta)
    s = np.sin(theta)

    dx = w / 2
    dy = h / 2

    pts = np.array([
        [-dx, -dy],
        [ dx, -dy],
        [ dx,  dy],
        [-dx,  dy]
    ])

    rot = np.array([
        [c, -s],
        [s,  c]
    ])

    pts = np.dot(pts, rot.T)
    pts[:, 0] += cx
    pts[:, 1] += cy

    return pts.astype(np.float32)


# -----------------------------
# Rotated IoU (approx via OpenCV)
# -----------------------------
def rotated_iou(box1, box2):
    poly1 = rbox_to_poly(box1)
    poly2 = rbox_to_poly(box2)

    poly1 = poly1.reshape(-1, 1, 2)
    poly2 = poly2.reshape(-1, 1, 2)

    inter_poly, _ = cv2.intersectConvexConvex(poly1, poly2)

    if inter_poly is None:
        return 0.0

    inter_area = cv2.contourArea(inter_poly)
    area1 = cv2.contourArea(poly1)
    area2 = cv2.contourArea(poly2)

    union = area1 + area2 - inter_area + 1e-6

    return inter_area / union


# -----------------------------
# IoU Loss
# -----------------------------
class RotatedIoULoss(nn.Module):
    def forward(self, pred, target):
        """
        pred/target: (B,5) -> cx,cy,w,h,theta
        """
        loss = 0.0

        for i in range(pred.shape[0]):
            iou = rotated_iou(pred[i].detach().cpu().numpy(),
                              target[i].detach().cpu().numpy())
            loss += (1 - iou)

        return torch.tensor(loss / pred.shape[0], requires_grad=True)


# -----------------------------
# Smooth L1 Box Loss
# -----------------------------
class BoxLoss(nn.Module):
    def forward(self, pred, target):
        return F.smooth_l1_loss(pred, target)


# -----------------------------
# Combined Detection Loss
# -----------------------------
class DetectionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cls_loss = FocalLoss()
        self.box_loss = BoxLoss()
        self.iou_loss = RotatedIoULoss()

    def forward(self, cls_logits, cls_targets, boxes_pred, boxes_target):
        return (
            self.cls_loss(cls_logits, cls_targets) +
            self.box_loss(boxes_pred, boxes_target) +
            self.iou_loss(boxes_pred, boxes_target)
        )
