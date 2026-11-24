from eval.audit.detect_metrics import (CANONICAL, best_threshold, present_items, prf,
                                       score_sets, sets_at_threshold)


def test_present_items_uses_threshold_and_synonyms():
    fr = {"armchair": 0.02, "plant": 0.001, "bed": 0.5}
    assert present_items(fr) == {"chair", "bed"}


def test_prf_edge_cases():
    assert prf(0, 0, 0)["f1"] == 0.0
    r = prf(2, 2, 0)
    assert r["precision"] == 0.5 and r["recall"] == 1.0


def test_score_sets_counts_per_item_and_micro():
    truth = [{"sofa", "bed"}, {"sofa"}]
    pred = [{"sofa"}, {"sofa", "lamp"}]
    out = score_sets(truth, pred)
    assert out["micro"]["tp"] == 2 and out["micro"]["fp"] == 1 and out["micro"]["fn"] == 1
    assert out["per_item"]["lamp"]["fp"] == 1
    assert out["per_item"]["bed"]["fn"] == 1
    assert set(out["per_item"]) == set(CANONICAL)


def test_threshold_selection_prefers_separating_value():
    truth = [{"sofa"}, set(), {"sofa"}, set()]
    scores = [{"sofa": 0.6}, {"sofa": 0.1}, {"sofa": 0.5}, {"sofa": 0.15}]
    thr = best_threshold(truth, scores)
    assert 0.15 <= thr <= 0.5
    assert sets_at_threshold(scores, thr) == truth
