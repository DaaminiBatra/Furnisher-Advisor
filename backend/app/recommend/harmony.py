"""How well does an item's colour go with the wall?

Interior-design rules of thumb, made explicit and testable:
  * neutrals (low chroma: white, grey, beige, black, wood tones) go with almost any wall;
  * otherwise an item reads as harmonious when its hue is analogous to the wall (within
    ~40 degrees) or complementary (about 180 +/- 25 degrees);
  * and it must stand out from the wall: a sofa the same colour as the wall disappears,
    so items closer than CIEDE2000 = 12 to the wall are penalised.
Score is in [0, 1].  The rules are a transparent baseline; the feedback-learning ranker
(later phase) re-weights them from users' saves.
"""

from __future__ import annotations

from .colour import delta_e2000, lch, rgb_to_lab

NEUTRAL_CHROMA = 14.0
MIN_CONTRAST = 12.0


def _hue_gap(h1: float, h2: float) -> float:
    d = abs(h1 - h2) % 360
    return min(d, 360 - d)


def harmony_score(item_rgb, wall_rgb) -> float:
    item_lab, wall_lab = rgb_to_lab(item_rgb), rgb_to_lab(wall_rgb)
    contrast = delta_e2000(item_lab, wall_lab)
    _, ic, ih = lch(item_lab)
    _, wc, wh = lch(wall_lab)

    if ic < NEUTRAL_CHROMA:
        base = 0.85
    elif wc < NEUTRAL_CHROMA:
        base = 0.8  # neutral wall: any colour reads as an accent
    else:
        gap = _hue_gap(ih, wh)
        analogous = max(0.0, 1 - gap / 40) if gap <= 40 else 0.0
        complementary = max(0.0, 1 - abs(gap - 180) / 25) if abs(gap - 180) <= 25 else 0.0
        base = 0.35 + 0.55 * max(analogous, complementary)
    if contrast < MIN_CONTRAST:
        base *= 0.4 + 0.6 * contrast / MIN_CONTRAST
    return round(max(0.0, min(1.0, base)), 4)
