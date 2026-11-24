import json

import pytest

from app.recommend.colour import delta_e2000, rgb_to_lab
from app.recommend.harmony import harmony_score
from app.recommend.plan import PRICE_BANDS, ROOMS, best_set, missing_items
from app.recommend.rank import load_catalog, top_products


def test_ciede2000_reference_pairs():
    assert delta_e2000((50, 2.6772, -79.7751), (50, 0, -82.7485)) == pytest.approx(2.0425, abs=1e-4)
    assert delta_e2000((50, 2.5, 0), (73, 25, -18)) == pytest.approx(27.1492, abs=1e-4)
    assert delta_e2000(rgb_to_lab((10, 20, 30)), rgb_to_lab((10, 20, 30))) == 0


def test_neutral_beats_clashing_and_same_colour_is_penalised():
    blue_wall = (60, 100, 170)
    assert harmony_score((235, 230, 220), blue_wall) > harmony_score((200, 60, 150), blue_wall)
    assert harmony_score(blue_wall, blue_wall) < harmony_score((235, 230, 220), blue_wall)
    assert 0 <= harmony_score((0, 0, 0), (255, 255, 255)) <= 1


def test_analogous_and_complementary_beat_a_random_hue():
    wall = (200, 90, 60)  # terracotta
    teal = (50, 150, 160)  # roughly complementary
    clash = (170, 60, 200)
    assert harmony_score(teal, wall) > harmony_score(clash, wall)


def test_missing_items_excludes_present_and_is_ordered():
    m = missing_items("living room", {"sofa", "tv"})
    assert "sofa" not in m and "tv" not in m
    assert m[0] == "table"
    with pytest.raises(KeyError):
        missing_items("garage", set())


def test_best_set_respects_budget_and_prefers_important_items():
    plan = best_set("bedroom", set(), 30000)
    assert sum(c.price for c in plan) <= 30000
    assert {c.item for c in plan} >= {"bed"}  # most important item is bought first
    assert best_set("bedroom", set(), 100) == []
    big = best_set("bedroom", set(), 10**7)
    assert {c.item for c in big} == set(ROOMS["bedroom"])
    assert all(c.tier == "high" for c in big)


def test_best_set_skips_present_items():
    assert "bed" not in {c.item for c in best_set("bedroom", {"bed"}, 50000)}
    assert set(PRICE_BANDS) >= {i for r in ROOMS.values() for i in r}


def test_top_products_filters_type_and_dedupes(tmp_path):
    rows = [
        {"asin": "a", "type": "sofa", "title": "Grey sofa", "brand": "X", "rgb": [200, 200, 200]},
        {"asin": "b", "type": "sofa", "title": "Grey sofa", "brand": "X", "rgb": [201, 200, 200]},
        {"asin": "c", "type": "sofa", "title": "Pink sofa", "brand": "Y", "rgb": [230, 80, 160]},
        {"asin": "d", "type": "lamp", "title": "Lamp", "brand": "Z", "rgb": [255, 255, 255]},
    ]
    p = tmp_path / "c.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    cat = load_catalog(str(p))
    out = top_products(cat, "sofa", (60, 100, 170), k=3)
    assert [o["asin"] for o in out][0] == "a"
    assert len(out) == 2 and all(o["type"] == "sofa" for o in out)


def test_recommend_endpoint(client):
    from app.main import app
    from app.routers.recommend import get_catalog

    app.dependency_overrides[get_catalog] = lambda: (
        {"asin": "a", "type": "bed", "title": "Oak bed", "brand": "X", "rgb": [200, 190, 170]},
    )
    try:
        r = client.post("/api/recommend", json={
            "room_type": "Bedroom", "wall_rgb": [60, 100, 170], "present_items": ["wardrobe"],
            "budget_inr": 40000,
        })
        assert r.status_code == 200
        body = r.json()
        assert "wardrobe" not in body["missing"]
        assert body["estimated_total_inr"] <= 40000
        bed = next(p for p in body["plan"] if p["item"] == "bed")
        assert bed["products"][0]["asin"] == "a"
        assert "not live" in body["price_note"]
        assert client.post("/api/recommend", json={
            "room_type": "garage", "wall_rgb": [1, 2, 3], "budget_inr": 10}).status_code == 404
        assert client.post("/api/recommend", json={
            "room_type": "bedroom", "wall_rgb": [1, 2, 999], "budget_inr": 10}).status_code == 422
    finally:
        app.dependency_overrides.pop(get_catalog, None)
        assert client.get("/api/recommend/rooms").json()["rooms"]
