from eval.audit.room_items import items_present, scene_to_room, score, v1_recommendations


def test_scene_mapping():
    assert scene_to_room("bedroom") == "Bed room"
    assert scene_to_room("home_office") == "Office"
    assert scene_to_room("airport_terminal") is None


def test_items_present_threshold():
    fr = {"bed": 0.2, "lamp": 0.001, "wardrobe": 0.05}
    p = items_present(fr)
    assert "bed" in p and "wardrobe" in p and "lamp" not in p


def test_v1_rule_is_set_difference_and_case_sensitive():
    rec = v1_recommendations("Bed room", {"bed", "chair"})
    assert "bed" not in rec and "chair" not in rec
    assert "wardrobe" in rec  # v1's detector can never report it


def test_bedroom_with_wardrobe_still_gets_wardrobe_recommended():
    recs = [{"room": "Bed room", "present": {"bed", "wardrobe", "lamp"}, "detected": {"bed"}}]
    r = score(recs)["per_room"]["Bed room"]
    # v1 recommends wardrobe and lamp (undetectable) although both are in the room.
    assert r["recommended_but_present"] >= 2
    assert r["share_recommended_already_present"] > 0


def test_perfect_detector_would_score_zero_false_recommendations():
    present = {"bed", "chair"}
    recs = [{"room": "Bed room", "present": present, "detected": set(present)}]
    r = score(recs)["per_room"]["Bed room"]
    assert r["recommended_but_present"] == 0


def test_unscored_items_are_skipped():
    # 'dressing table' and 'side table' have no ADE ground truth: never counted.
    recs = [{"room": "Bed room", "present": set(), "detected": set()}]
    r = score(recs)["per_room"]["Bed room"]
    from eval.audit.room_items import ITEM_TO_ADE
    from eval.audit.v1_coverage import V1_ROOM_FURNITURE
    scored = [i for i in V1_ROOM_FURNITURE["Bed room"] if i in ITEM_TO_ADE]
    assert r["item_checks"] == len(scored)
    assert "dressing table" not in scored
