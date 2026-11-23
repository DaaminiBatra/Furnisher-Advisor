"""Audit of v1's 'most dominant colour = wall colour' idea.

v1 (flask-server/api/color_analysis.py) resizes the whole photo to 600x400, runs
K-Means with k=5 on every pixel, takes the biggest cluster as the 'wall colour' and
names it with the nearest CSS colour in plain RGB.  Nothing identifies which pixels
are wall, so a large sofa, floor or bedspread can win.

Ground truth here comes from images whose wall pixels are labelled (ADE20K), so the
question is measurable: how far is v1's answer from the real wall colour, in a
perceptual colour space (CIEDE2000), and how much of the winning cluster is wall?
"""

from __future__ import annotations

import cv2
import numpy as np
from skimage.color import deltaE_ciede2000, rgb2lab
from sklearn.cluster import KMeans

from .v1_colors import CSS3_COLORS


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


_CSS_NAMES = list(CSS3_COLORS)
_CSS_RGB = np.array([_hex_to_rgb(CSS3_COLORS[n]) for n in _CSS_NAMES], dtype=float)


def v1_closest_css_name(rgb) -> str:
    """v1's naming: nearest CSS colour by squared RGB distance."""
    d = ((_CSS_RGB - np.asarray(rgb, dtype=float)) ** 2).sum(axis=1)
    return _CSS_NAMES[int(np.argmin(d))]


def css_name_to_rgb(name: str) -> tuple[int, int, int]:
    return _hex_to_rgb(CSS3_COLORS[name])


def v1_dominant(image_rgb: np.ndarray, k: int = 5):
    """Re-implementation of v1's detect_dominant_wall_colors on an RGB array.

    Returns (centre_rgb, label_map) where label_map has the 600x400 shape v1 used
    (rows=400, cols=600) and holds the cluster index of the winning cluster as a
    boolean mask in `winner_mask`.
    """
    small = cv2.resize(image_rgb, (600, 400))  # same call as v1 (INTER_LINEAR)
    pixels = small.reshape(-1, 3)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(pixels)
    centres = km.cluster_centers_.astype(int)
    counts = np.bincount(labels, minlength=k)
    winner = int(np.argmax(counts))
    winner_mask = (labels == winner).reshape(400, 600)
    return centres[winner], winner_mask


def true_wall_rgb(image_rgb: np.ndarray, wall_mask: np.ndarray) -> np.ndarray:
    """Median RGB of the labelled wall pixels (median is robust to shadows/pictures)."""
    return np.median(image_rgb[wall_mask.astype(bool)], axis=0)


def delta_e(rgb_a, rgb_b) -> float:
    a = rgb2lab(np.asarray(rgb_a, dtype=float).reshape(1, 1, 3) / 255.0)
    b = rgb2lab(np.asarray(rgb_b, dtype=float).reshape(1, 1, 3) / 255.0)
    return float(deltaE_ciede2000(a, b)[0, 0])


def evaluate_image(image_rgb: np.ndarray, wall_mask: np.ndarray, k: int = 5) -> dict:
    """One image: v1's answer vs the labelled wall."""
    truth = true_wall_rgb(image_rgb, wall_mask)
    centre, winner_mask = v1_dominant(image_rgb, k)
    name = v1_closest_css_name(centre)
    named = css_name_to_rgb(name)

    # How much of v1's winning cluster is actually wall? Compare at v1's 600x400 size.
    wall_small = cv2.resize(wall_mask.astype(np.uint8), (600, 400),
                            interpolation=cv2.INTER_NEAREST).astype(bool)
    purity = float((winner_mask & wall_small).sum() / max(winner_mask.sum(), 1))
    return {
        "wall_fraction": float(wall_mask.mean()),
        "de_cluster_vs_truth": delta_e(centre, truth),   # error before naming
        "de_named_vs_truth": delta_e(named, truth),      # error of what v1 shows users
        "winner_wall_purity": purity,
        "v1_name": name,
    }


def summarise(rows: list[dict]) -> dict:
    de = np.array([r["de_named_vs_truth"] for r in rows])
    pure = np.array([r["winner_wall_purity"] for r in rows])
    return {
        "n_images": len(rows),
        "median_dE_named": float(np.median(de)),
        "mean_dE_named": float(de.mean()),
        # CIEDE2000 around 2.3 is a just-noticeable difference; >10 is a clearly
        # different colour to most people.
        "share_dE_gt_10": float((de > 10).mean()),
        "share_winner_mostly_not_wall": float((pure < 0.5).mean()),
        "median_winner_wall_purity": float(np.median(pure)),
    }
