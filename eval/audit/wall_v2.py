"""v2 wall colour: take the segmented wall pixels, estimate the colour in CIELAB.

Unlike v1 (K-Means over the whole frame, then a nearest CSS name), the estimate only
sees pixels the segmentation model calls wall, and uses a trimmed median in Lab so
shadows, light patches and stray edge pixels do not drag it.
"""

from __future__ import annotations

import numpy as np
from skimage.color import lab2rgb, rgb2lab


def v2_wall_rgb(image_rgb: np.ndarray, wall_mask: np.ndarray, trim: float = 0.2) -> np.ndarray:
    """Trimmed median of the wall pixels' Lab values, returned as RGB (0-255)."""
    mask = wall_mask.astype(bool)
    if mask.sum() == 0:
        raise ValueError("empty wall mask")
    lab = rgb2lab(image_rgb[mask].reshape(-1, 1, 3) / 255.0).reshape(-1, 3)
    # drop the darkest and brightest `trim/2` of pixels by lightness, then take the median
    lo, hi = np.quantile(lab[:, 0], [trim / 2, 1 - trim / 2])
    keep = lab[(lab[:, 0] >= lo) & (lab[:, 0] <= hi)]
    med = np.median(keep if len(keep) else lab, axis=0)
    rgb = lab2rgb(med.reshape(1, 1, 3))[0, 0] * 255.0
    return np.clip(rgb, 0, 255)


def mask_iou(pred: np.ndarray, truth: np.ndarray) -> float:
    p, t = pred.astype(bool), truth.astype(bool)
    union = (p | t).sum()
    return float((p & t).sum() / union) if union else 1.0
