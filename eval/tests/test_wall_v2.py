import numpy as np
import pytest

from eval.audit.wall_v2 import mask_iou, v2_wall_rgb


def test_recovers_flat_colour():
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    img[:] = (200, 180, 150)
    out = v2_wall_rgb(img, np.ones((20, 20), bool))
    assert np.allclose(out, (200, 180, 150), atol=2)


def test_ignores_pixels_outside_mask():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    img[:, :5] = (230, 230, 220)
    img[:, 5:] = (20, 20, 20)
    mask = np.zeros((10, 10), bool)
    mask[:, :5] = True
    assert np.allclose(v2_wall_rgb(img, mask), (230, 230, 220), atol=2)


def test_robust_to_shadow_and_highlight_outliers():
    img = np.full((10, 10, 3), (180, 170, 160), dtype=np.uint8)
    img[0, :] = (10, 10, 10)       # 10% very dark
    img[9, :] = (255, 255, 255)    # 10% blown out
    out = v2_wall_rgb(img, np.ones((10, 10), bool))
    assert np.allclose(out, (180, 170, 160), atol=3)


def test_empty_mask_raises():
    with pytest.raises(ValueError):
        v2_wall_rgb(np.zeros((4, 4, 3), np.uint8), np.zeros((4, 4), bool))


def test_mask_iou():
    a = np.array([[1, 1], [0, 0]], bool)
    b = np.array([[1, 0], [0, 0]], bool)
    assert mask_iou(a, b) == 0.5
    assert mask_iou(np.zeros((2, 2)), np.zeros((2, 2))) == 1.0
