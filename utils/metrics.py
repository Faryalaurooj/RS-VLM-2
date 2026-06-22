import numpy as np
from collections import defaultdict

from utils.losses import rotated_iou


# -----------------------------
# IoU matching (simple greedy)
# -----------------------------
def match_detections(pred_boxes, gt_boxes, iou_threshold=0.5):
    """
    Greedy matching for evaluation.
    Returns TP / FP counts.
    """

    matched_gt = set()
    tp, fp = 0, 0

    for pb in pred_boxes:
        best_iou = 0
        best_j = -1

        for j, gb in enumerate(gt_boxes):
            if j in matched_gt:
                continue

            iou = rotated_iou(pb, gb)
            if iou > best_iou:
                best_iou = iou
                best_j = j

        if best_iou >= iou_threshold:
            tp += 1
            matched_gt.add(best_j)
        else:
            fp += 1

    fn = len(gt_boxes) - len(matched_gt)

    return tp, fp, fn


# -----------------------------
# AP computation (VOC-style)
# -----------------------------
def compute_ap(recalls, precisions):
    recalls = np.array(recalls)
    precisions = np.array(precisions)

    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))

    for i in range(len(precisions) - 1, 0, -1):
        precisions[i - 1] = max(precisions[i - 1], precisions[i])

    idx = np.where(recalls[1:] != recalls[:-1])[0]

    ap = np.sum((recalls[idx + 1] - recalls[idx]) * precisions[idx + 1])

    return ap


# -----------------------------
# mAP evaluator
# -----------------------------
class DetectionEvaluator:
    def __init__(self, iou_threshold=0.5):
        self.iou_threshold = iou_threshold

        self.predictions = defaultdict(list)
        self.ground_truths = defaultdict(list)

    def add_batch(self, preds, gts):
        """
        preds: list of (boxes, scores, labels)
        gts: list of (boxes, labels)
        """

        for p, g in zip(preds, gts):
            pb, ps, pl = p
            gb, gl = g

            for b, s, l in zip(pb, ps, pl):
                self.predictions[l].append((b, s))

            for b, l in zip(gb, gl):
                self.ground_truths[l].append(b)

    def evaluate(self):
        ap_list = []

        for cls in self.ground_truths.keys():
            preds = sorted(self.predictions[cls], key=lambda x: -x[1])
            gts = self.ground_truths[cls]

            tp_list, fp_list = [], []
            matched = []

            for pb, score in preds:
                best_iou = 0
                best_j = -1

                for j, gb in enumerate(gts):
                    if j in matched:
                        continue

                    iou = rotated_iou(pb, gb)
                    if iou > best_iou:
                        best_iou = iou
                        best_j = j

                if best_iou >= self.iou_threshold:
                    tp_list.append(1)
                    fp_list.append(0)
                    matched.append(best_j)
                else:
                    tp_list.append(0)
                    fp_list.append(1)

            tp_cum = np.cumsum(tp_list)
            fp_cum = np.cumsum(fp_list)

            recalls = tp_cum / (len(gts) + 1e-6)
            precisions = tp_cum / (tp_cum + fp_cum + 1e-6)

            ap = compute_ap(recalls, precisions)
            ap_list.append(ap)

        return np.mean(ap_list) if len(ap_list) > 0 else 0.0
