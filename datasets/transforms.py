import torch
import random
import numpy as np
import cv2


# -----------------------------
# Normalize
# -----------------------------
class Normalize:
    def __init__(self):
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __call__(self, sample):
        img = sample["image"].numpy().transpose(1, 2, 0)
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)

        sample["image"] = torch.tensor(img, dtype=torch.float32)
        return sample


# -----------------------------
# Horizontal Flip (bbox-safe)
# -----------------------------
class HorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if random.random() > self.p:
            return sample

        img = sample["image"]
        boxes = sample["boxes"]

        _, h, w = img.shape

        img = torch.flip(img, dims=[2])

        if len(boxes) > 0:
            boxes = boxes.clone()

            # cx flip
            boxes[:, 0] = w - boxes[:, 0]

            # theta flip (important for rotation consistency)
            boxes[:, 4] = -boxes[:, 4]

        sample["image"] = img
        sample["boxes"] = boxes

        return sample


# -----------------------------
# Random Rotation (key for SGSM idea)
# -----------------------------
class RandomRotate:
    def __init__(self, max_angle=15):
        self.max_angle = max_angle

    def __call__(self, sample):
        angle = random.uniform(-self.max_angle, self.max_angle)

        img = sample["image"]
        boxes = sample["boxes"]

        _, h, w = img.shape

        img_np = img.permute(1, 2, 0).numpy()

        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        rotated = cv2.warpAffine(img_np, M, (w, h))

        # NOTE:
        # Full box rotation is complex; simplified approximation:
        if len(boxes) > 0:
            boxes = boxes.clone()
            boxes[:, 4] += np.deg2rad(angle)

        sample["image"] = torch.tensor(rotated).permute(2, 0, 1)
        sample["boxes"] = boxes

        return sample


# -----------------------------
# Compose
# -----------------------------
class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample
