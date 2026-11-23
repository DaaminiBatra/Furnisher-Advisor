from eval.audit.v1_coverage import (
    COCO80,
    V1_ROOM_FURNITURE,
    audit,
    possible_labels,
    rename_only_items,
)


def test_possible_labels_apply_renames():
    labels = possible_labels()
    assert "sofa" in labels and "couch" not in labels
    assert "table" in labels and "bench" not in labels
    assert "dining table" in labels  # a different string from "table"


def test_known_blind_spots():
    res = audit()["per_room"]
    assert "lamp" in res["Bed room"]["never_detected"]
    assert "wardrobe" in res["Bed room"]["never_detected"]
    assert "bookshelf" in res["Library"]["never_detected"]
    assert set(res["Kitchen"]["can_be_detected"]) == {"refrigerator", "oven"}


def test_case_sensitive_exact_match_like_v1():
    # v1 compared exact strings, so 'Cabinets' can never equal a COCO label.
    assert audit({"R": ["Cabinets", "chair"]})["per_room"]["R"]["can_be_detected"] == ["chair"]


def test_totals_are_consistent():
    res = audit()
    n = sum(len(v) for v in V1_ROOM_FURNITURE.values())
    assert res["overall"]["checklist_slots"] == n == 52
    detectable = sum(len(r["can_be_detected"]) for r in res["per_room"].values())
    assert detectable + res["overall"]["never_detected_slots"] == n


def test_table_and_sofa_only_reachable_through_renames():
    assert rename_only_items() == {"table": "bench", "sofa": "couch"}
    assert "table" not in COCO80 and "dining table" in COCO80
