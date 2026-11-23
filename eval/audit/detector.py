"""Box-matching metrics for detector audits (no model code here).

Greedy one-to-one matching at IoU >= 0.5 per class, highest-confidence predictions
first, as in the standard PASCAL-style protocol. Reports recall (how many real
objects were found) and precision (how many predictions were right).
"""

from __future__ import annotations

from collections import defaultdict

Box = tuple[float, float, float, float]  # x1, y1, x2, y2


def iou(a: Box, b: Box) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (area_a + area_b - inter)


def match_image(gts: list[Box], preds: list[tuple[Box, float]], thr: float = 0.5):
    """Return (true_positives, false_positives, false_negatives) for one class/image."""
    order = sorted(range(len(preds)), key=lambda i: -preds[i][1])
    used: set[int] = set()
    tp = 0
    for i in order:
        best, best_j = 0.0, -1
        for j, g in enumerate(gts):
            if j in used:
                continue
            o = iou(preds[i][0], g)
            if o > best:
                best, best_j = o, j
        if best >= thr:
            used.add(best_j)
            tp += 1
    return tp, len(preds) - tp, len(gts) - tp


def per_class_report(records: list[dict], thr: float = 0.5) -> dict[str, dict]:
    """records: one dict per (image, class): {"cls", "gts": [Box], "preds": [(Box, conf)]}."""
    tot = defaultdict(lambda: [0, 0, 0])
    for r in records:
        tp, fp, fn = match_image(r["gts"], r["preds"], thr)
        t = tot[r["cls"]]
        t[0] += tp
        t[1] += fp
        t[2] += fn
    out = {}
    for cls, (tp, fp, fn) in tot.items():
        out[cls] = {
            "gt_objects": tp + fn,
            "recall": tp / (tp + fn) if tp + fn else None,
            "precision": tp / (tp + fp) if tp + fp else None,
            "tp": tp, "fp": fp, "fn": fn,
        }
    return out
