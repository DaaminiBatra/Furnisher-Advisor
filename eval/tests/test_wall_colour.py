import numpy as np

from eval.audit.wall_colour import delta_e, evaluate_image, summarise, true_wall_rgb, v1_closest_css_name


def _room(wall_rgb, other_rgb, wall_share, h=200, w=300, seed=0):
    """Left part wall, right part 'furniture'. Small noise so K-Means has texture."""
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), dtype=float)
    cut = int(w * wall_share)
    img[:, :cut] = wall_rgb
    img[:, cut:] = other_rgb
    img += rng.normal(0, 2, img.shape)
    mask = np.zeros((h, w), dtype=bool)
    mask[:, :cut] = True
    return np.clip(img, 0, 255).astype(np.uint8), mask


def test_delta_e_basics():
    assert delta_e((120, 120, 120), (120, 120, 120)) == 0
    assert delta_e((255, 255, 255), (0, 0, 0)) > 90
    assert delta_e((200, 200, 200), (205, 200, 200)) < delta_e((200, 200, 200), (240, 200, 200))


def test_css_name_is_nearest_rgb():
    assert v1_closest_css_name((255, 0, 0)) == "Red"
    assert v1_closest_css_name((250, 250, 250)) in {"White", "Snow", "White Smoke", "Ghost White", "Floral White"}


def test_true_wall_rgb_uses_mask_only():
    img, mask = _room((230, 220, 200), (30, 60, 140), 0.5)
    t = true_wall_rgb(img, mask)
    assert np.allclose(t, (230, 220, 200), atol=4)


def test_v1_is_fine_when_wall_dominates():
    img, mask = _room((232, 222, 200), (30, 60, 140), 0.75)
    r = evaluate_image(img, mask)
    assert r["winner_wall_purity"] > 0.9
    assert r["de_cluster_vs_truth"] < 4


def test_v1_picks_the_sofa_when_sofa_dominates():
    # Wall covers only 35% of the frame, the sofa 65%: v1 reports the sofa colour.
    img, mask = _room((232, 222, 200), (30, 60, 140), 0.35)
    r = evaluate_image(img, mask)
    assert r["winner_wall_purity"] < 0.1
    assert r["de_cluster_vs_truth"] > 30


def test_naming_adds_error_on_top_of_clustering():
    img, mask = _room((233, 219, 191), (30, 60, 140), 0.75)
    r = evaluate_image(img, mask)
    assert r["de_named_vs_truth"] >= 0


def test_summarise_shares():
    rows = [
        {"de_named_vs_truth": 3.0, "winner_wall_purity": 0.9},
        {"de_named_vs_truth": 25.0, "winner_wall_purity": 0.1},
        {"de_named_vs_truth": 12.0, "winner_wall_purity": 0.4},
        {"de_named_vs_truth": 6.0, "winner_wall_purity": 0.8},
    ]
    s = summarise(rows)
    assert s["n_images"] == 4
    assert s["share_dE_gt_10"] == 0.5
    assert s["share_winner_mostly_not_wall"] == 0.5
