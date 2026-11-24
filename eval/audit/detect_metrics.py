"""Image-level furniture presence metrics shared by the detector comparison."""

from __future__ import annotations

import numpy as np

# Canonical items scored (one entry per real object type, duplicates in v1's lists merged).
CANONICAL = {
    "sofa": ["sofa"],
    "chair": ["chair", "armchair", "swivel chair"],
    "bed": ["bed"],
    "plant": ["plant"],
    "clock": ["clock"],
    "vase": ["vase"],
    "table": ["table"],
    "refrigerator": ["refrigerator"],
    "oven": ["oven", "stove"],
    "lamp": ["lamp"],
    "bookshelf": ["bookcase", "shelf"],
    "wardrobe": ["wardrobe"],
    "rug": ["rug"],
    "tv": ["television receiver"],
    "painting": ["painting"],
    "cabinet": ["cabinet"],
    "ottoman": ["ottoman"],
    "dishwasher": ["dishwasher"],
}

# COCO class -> canonical item: the best case for a COCO detector (more generous than v1,
# which also needed exact string matches against its checklist).
COCO_TO_CANONICAL = {
    "couch": "sofa", "chair": "chair", "bed": "bed", "potted plant": "plant", "clock": "clock",
    "vase": "vase", "dining table": "table", "refrigerator": "refrigerator", "oven": "oven",
    "tv": "tv",
}


def present_items(class_fractions: dict[str, float], min_fraction: float = 0.005) -> set[str]:
    return {item for item, names in CANONICAL.items()
            if any(class_fractions.get(n, 0.0) >= min_fraction for n in names)}


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f, "tp": tp, "fp": fp, "fn": fn}


def score_sets(truth: list[set[str]], pred: list[set[str]]) -> dict:
    """Micro and per-item precision/recall/F1 over images."""
    tp = fp = fn = 0
    per = {i: [0, 0, 0] for i in CANONICAL}
    for t, p in zip(truth, pred):
        for item in CANONICAL:
            a, b = item in t, item in p
            if a and b:
                tp += 1; per[item][0] += 1
            elif b:
                fp += 1; per[item][1] += 1
            elif a:
                fn += 1; per[item][2] += 1
    return {"micro": prf(tp, fp, fn), "per_item": {k: prf(*v) for k, v in per.items()}}


def sets_at_threshold(scores: list[dict[str, float]], thr: float) -> list[set[str]]:
    return [{i for i, s in d.items() if s >= thr} for d in scores]


def best_threshold(truth, scores, grid=None) -> float:
    grid = grid if grid is not None else np.arange(0.05, 0.55, 0.025)
    best, best_f = grid[0], -1.0
    for g in grid:
        f = score_sets(truth, sets_at_threshold(scores, float(g)))["micro"]["f1"]
        if f > best_f:
            best, best_f = float(g), f
    return best
