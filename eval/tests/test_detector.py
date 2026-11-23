from eval.audit.detector import iou, match_image, per_class_report


def test_iou():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0
    assert abs(iou((0, 0, 10, 10), (5, 0, 15, 10)) - 1 / 3) < 1e-9


def test_match_is_one_to_one():
    gt = [(0, 0, 10, 10)]
    preds = [((0, 0, 10, 10), 0.9), ((0, 0, 10, 10), 0.8)]  # duplicate detection
    assert match_image(gt, preds) == (1, 1, 0)


def test_missed_and_spurious():
    assert match_image([(0, 0, 10, 10)], []) == (0, 0, 1)
    assert match_image([], [((0, 0, 5, 5), 0.7)]) == (0, 1, 0)


def test_low_iou_is_not_a_match():
    assert match_image([(0, 0, 10, 10)], [((8, 8, 20, 20), 0.9)]) == (0, 1, 1)


def test_report_never_detected_class_has_zero_recall():
    recs = [
        {"cls": "Wardrobe", "gts": [(0, 0, 10, 10), (20, 20, 30, 30)], "preds": []},
        {"cls": "Chair", "gts": [(0, 0, 10, 10)], "preds": [((0, 0, 10, 10), 0.9)]},
    ]
    rep = per_class_report(recs)
    assert rep["Wardrobe"]["recall"] == 0 and rep["Wardrobe"]["gt_objects"] == 2
    assert rep["Wardrobe"]["precision"] is None
    assert rep["Chair"]["recall"] == 1 and rep["Chair"]["precision"] == 1
