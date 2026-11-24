"""Listing colour words -> RGB.

Colour measured from product photos proved unreliable (off-white backgrounds, shadows and
lifestyle shots pull it towards grey: e.g. 'blue' sofas measured grey), so the listing's own
colour word is the primary colour and the photo measurement is the fallback.
"""

from __future__ import annotations

import re

NAMES = {
    "white": (245, 245, 242), "off white": (240, 235, 222), "ivory": (250, 245, 225), "cream": (245, 236, 215),
    "beige": (222, 205, 178), "tan": (205, 170, 125), "sand": (214, 195, 160), "taupe": (160, 140, 125),
    "natural": (200, 165, 120), "khaki": (195, 176, 130), "camel": (193, 154, 107), "oatmeal": (215, 200, 175),
    "grey": (130, 130, 132), "gray": (130, 130, 132), "light grey": (200, 200, 202), "light gray": (200, 200, 202),
    "dark grey": (80, 80, 84), "dark gray": (80, 80, 84), "charcoal": (60, 62, 66), "silver": (192, 192, 195),
    "nickel": (200, 200, 198), "brushed nickel": (190, 190, 188), "chrome": (210, 212, 215), "pewter": (140, 145, 150),
    "black": (25, 25, 27), "espresso": (60, 40, 32), "java": (70, 48, 36), "chocolate": (85, 55, 40),
    "brown": (120, 80, 55), "walnut": (110, 80, 60), "oak": (190, 150, 100), "mahogany": (100, 45, 35),
    "cherry": (130, 55, 45), "teak": (165, 120, 75), "wood": (165, 120, 80), "saddle": (140, 90, 55),
    "cognac": (150, 80, 45), "rust": (165, 80, 45), "terracotta": (195, 105, 75), "copper": (184, 115, 51),
    "bronze": (140, 110, 60), "gold": (212, 175, 55), "brass": (190, 160, 70), "mustard": (210, 170, 40),
    "yellow": (240, 210, 70), "orange": (235, 130, 40), "coral": (240, 125, 105), "red": (190, 40, 40),
    "burgundy": (110, 30, 45), "maroon": (110, 30, 40), "wine": (110, 30, 55), "pink": (235, 160, 180),
    "blush": (235, 195, 190), "rose": (210, 120, 140), "magenta": (190, 40, 130), "purple": (120, 70, 150),
    "lavender": (190, 170, 215), "plum": (100, 50, 90), "violet": (130, 80, 160),
    "blue": (50, 100, 180), "navy": (30, 40, 85), "light blue": (150, 190, 225), "sky blue": (135, 190, 230),
    "teal": (30, 130, 135), "turquoise": (60, 190, 190), "aqua": (110, 205, 205), "cyan": (60, 190, 210),
    "green": (60, 140, 80), "olive": (110, 120, 55), "sage": (160, 180, 150), "mint": (170, 220, 190),
    "emerald": (30, 120, 80), "forest": (35, 85, 55), "lime": (150, 200, 60),
    "multicolor": None, "multi": None, "clear": None, "transparent": None,
}
_PHRASES = sorted((k for k, v in NAMES.items() if v is not None), key=lambda k: -len(k))


def colour_from_words(words: str | None):
    """First recognisable colour in the listing's colour text, or None."""
    if not words:
        return None
    text = re.sub(r"[^a-z]+", " ", words.lower()).strip()
    best = None
    for phrase in _PHRASES:
        m = re.search(rf"\b{phrase}\b", text)
        if m and (best is None or m.start() < best[0]):
            best = (m.start(), phrase)
    return list(NAMES[best[1]]) if best else None
