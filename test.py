import torch
from torch.utils.data import DataLoader

from datasets.dota import DOTADataset
from models.detector import SGSMDetector
from utils.metrics import DetectionEvaluator


# -----------------------------
# Collate function
# -----------------------------
def collate_fn(batch):
    images = torch.stack([b["image"] for b in batch])
    boxes = [b["boxes"] for b in batch]
    labels = [b["labels"] for b in batch]

    return images, boxes, labels


# -----------------------------
# Dummy decoder (same logic as inference)
# NOTE: Replace with proper NMS/decoder later
# -----------------------------
def decode_outputs(outputs, topk=50):
    cls = outputs["cls_logits"]
    bbox = outputs["bbox_reg"]

    B = cls.shape[0]
    results = []

    for i in range(B):
        scores = cls[i].softmax(dim=0).flatten()
        top = torch.topk(scores, k=min(topk, scores.numel()))

        boxes = []

        for idx in top.indices:
            idx = idx.item()

            _, h, w = bbox.shape
            y = idx // w
            x = idx % w

            box = bbox[i, :, y, x].detach().cpu().numpy()
            boxes.append(box)

        # dummy: all same class 0
        labels = [0] * len(boxes)
        scores = top.values.detach().cpu().numpy()

        results.append((boxes, scores, labels))

    return results


# -----------------------------
# Evaluation loop
# -----------------------------
def evaluate(
    img_dir,
    ann_dir,
    model_path,
    num_classes=15,
    batch_size=2,
    device="cuda"
):

    device = torch.device(device if torch.cuda.is_available() else "cpu")

    dataset = DOTADataset(img_dir, ann_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = SGSMDetector(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    evaluator = DetectionEvaluator(iou_threshold=0.5)

    with torch.no_grad():
        for images, gt_boxes, gt_labels in loader:
            images = images.to(device)

            outputs, _ = model(images)

            preds = decode_outputs(outputs)

            gts = []
            for b, l in zip(gt_boxes, gt_labels):
                gts.append((b.numpy(), l.numpy()))

            evaluator.add_batch(preds, gts)

    mAP = evaluator.evaluate()

    print(f"\n========================")
    print(f"mAP@0.5: {mAP:.4f}")
    print(f"========================\n")


# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    evaluate(
        img_dir="datasets/DOTA/images",
        ann_dir="datasets/DOTA/labels",
        model_path="sgsm_detector.pth",
        device="cuda"
    )
