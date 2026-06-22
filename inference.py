import torch
import cv2
import numpy as np

from models.detector import SGSMDetector


# -----------------------------
# RBOX → Polygon
# -----------------------------
def rbox_to_poly(box):
    cx, cy, w, h, theta = box

    c = np.cos(theta)
    s = np.sin(theta)

    dx = w / 2
    dy = h / 2

    pts = np.array([
        [-dx, -dy],
        [ dx, -dy],
        [ dx,  dy],
        [-dx,  dy]
    ], dtype=np.float32)

    rot = np.array([[c, -s],
                    [s,  c]])

    pts = pts @ rot.T
    pts[:, 0] += cx
    pts[:, 1] += cy

    return pts.astype(np.int32)


# -----------------------------
# Draw boxes
# -----------------------------
def draw_boxes(img, boxes, scores=None, threshold=0.3):
    for i, box in enumerate(boxes):
        if scores is not None and scores[i] < threshold:
            continue

        poly = rbox_to_poly(box)
        cv2.polylines(img, [poly], True, (0, 255, 0), 2)

    return img


# -----------------------------
# Decode model output properly
# -----------------------------
def decode(outputs, conf_thresh=0.3, topk=50):
    """
    Converts dense feature map output → final detections
    """

    cls = outputs["cls_logits"][0]   # (C, H, W)
    bbox = outputs["bbox_reg"][0]    # (5, H, W)

    scores = cls.softmax(dim=0).max(dim=0).values  # (H, W)

    H, W = scores.shape

    flat_scores = scores.flatten()
    topk_vals, topk_idx = torch.topk(flat_scores, k=min(topk, flat_scores.numel()))

    boxes = []
    final_scores = []

    for score, idx in zip(topk_vals, topk_idx):
        if score.item() < conf_thresh:
            continue

        y = idx // W
        x = idx % W

        box = bbox[:, y, x].detach().cpu().numpy()

        boxes.append(box)
        final_scores.append(score.item())

    return boxes, final_scores


# -----------------------------
# Inference function
# -----------------------------
def inference(image_path, model_path, device="cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # load model
    model = SGSMDetector(num_classes=15).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # load image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    orig = img.copy()

    img = cv2.resize(img, (1024, 1024))

    tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
    tensor = tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs, _ = model(tensor)

        boxes, scores = decode(outputs, conf_thresh=0.3, topk=100)

    result = draw_boxes(img.copy(), boxes, scores)

    out_path = "result.png"
    cv2.imwrite(out_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

    print(f"[INFO] Saved output to {out_path}")


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    inference(
        image_path="test.jpg",
        model_path="sgsm_detector.pth",
        device="cuda"
    )
