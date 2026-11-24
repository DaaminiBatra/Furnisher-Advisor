# Evaluation

Furnisher Advisor is the enhanced successor to the Personalized Furnishing Advisor, a
2025 BTech group project (the baseline below). Every design choice here is measured
against that baseline on labelled public photos.

**Baseline method.** 253 indoor room photos from the ADE20K validation split (bedroom, living
room, dining room, kitchen, home office, study, library), with pixel-level ground-truth
labels. v1's exact pipeline was reproduced: YOLOv5x (COCO, conf 0.25), K-Means (k=5) on
the whole image, nearest CSS colour name, and its hand-written per-room item checklists.
Code: `eval/audit/`, runner: `eval/kaggle/phase1_audit.py`, 23 unit tests.

## Results

| Question | Result |
|---|---|
| v1 checklist items the detector can never see (no COCO class) | 27 of 52 (52%) |
| Wall colour error vs real wall pixels, median (CIEDE2000) | 12.1 (about 2 is just noticeable) |
| Photos where the colour name is off by more than 10 | 61% (of 210) |
| Photos where the "dominant colour" cluster is mostly not wall | 63% |
| Median wall purity of the chosen cluster | 38% |
| v1 recommendations for items already present in the photo | 29.7% (415 of 1,396) |

## v2 wall colour on the same photos (2026-11-24)

Same 210 photos, same ground truth (median colour of human-labelled wall pixels,
CIEDE2000). v2 = SegFormer-B2 (ADE20K) wall mask, then a trimmed median in CIELAB.

| | v1 (K-Means + CSS name) | v2 (segmentation + Lab) |
|---|---|---|
| Median error | 12.1 | 1.07 |
| Mean error | 17.4 | 2.19 |
| Photos with error > 10 | 61% | 3.3% |
| Photos with error > 5 | not recorded | 8.1% |

v2 was closer to the labelled wall colour than v1 on 99.0% of photos. v1 figures come from
the phase 1 run on the identical photo set. The segmentation model was trained on ADE20K,
the same dataset used for scoring, so this is an optimistic estimate; see the independent NYUv2 check below.
Wall-mask IoU was not captured in this run.

### Independent check on NYUv2 (not ADE20K)

300 test photos from NYUv2 (wall = label 1, confirmed visually), which the segmenter was
not trained on. Same scoring.

| | Baseline (K-Means + CSS name) | Furnisher Advisor (segmentation + Lab) |
|---|---|---|
| Median error (CIEDE2000) | 17.2 | 1.92 |
| Mean error | 21.9 | 3.32 |
| 90th percentile | 43.0 | 7.2 |
| Photos with error > 10 | 77.7% | 5.0% |
| Photos with error > 5 | 95.7% | 16.3% |

The new method was closer on 97.7% of photos; median wall-mask IoU 0.67. Error is higher
than on ADE20K, as expected away from the training distribution, and still far below the
baseline. NYUv2 scenes (homes and offices photographed with a Kinect) overlap in type with
ADE20K's indoor classes, so this is independent photos, not an entirely different domain.

## Furniture detection: COCO detector vs open-vocabulary detector

Image-level presence of 18 furniture types on ADE20K photos (ground truth: ADE20K labels,
type present when it covers at least 0.5% of the image). Scored on a held-out half of the
253 photos; the open-vocabulary threshold (0.25) was tuned on the other half. The COCO
detector (YOLOv5x, confidence 0.25) is given its best case: its class names are mapped to
our types directly, which is more generous than the baseline's exact-string matching.
Open-vocabulary detector: OWLv2 (base), prompted with plain item names, no training.

| | COCO detector (best case) | Open-vocabulary (OWLv2) |
|---|---|---|
| Recall | 0.45 | 0.81 |
| Precision | 0.67 | 0.61 |
| F1 | 0.54 | 0.69 |

Per-type recall, COCO vs open-vocabulary: sofa 0.97/1.0, chair 0.90/0.90, bed 0.89/0.96,
refrigerator 0.88/1.0, oven 0.76/0.97, table 0.48/0.56. Types COCO cannot detect at all
(recall 0.0): lamp, bookshelf, wardrobe, rug, painting, cabinet, ottoman, dishwasher;
the open-vocabulary detector recovers most of them (lamp 1.0, rug 1.0, ottoman 1.0,
painting 0.86, dishwasher 0.89, cabinet 0.72, bookshelf 0.44, wardrobe 0.29).

Trade-offs, stated plainly: the open-vocabulary detector is worse on potted plants
(0.56 vs 0.96) and has slightly lower precision (more false alarms), and it is far
heavier to run than YOLO. That is why the plan is to use it as a teacher and distil it
into a fast furniture-specific detector rather than ship it directly. Television shows
0.0 recall for both detectors; it is not being claimed either way until the cause
(very few positives in this sample, or a labelling mismatch) is checked.

## Distilled furniture detector (student)

A small YOLOv8s detector trained only on boxes produced by the open-vocabulary detector
(1,500 ADE20K training photos of home interiors; no human box annotations), then scored on
the same held-out validation photos and with the same presence metric and tuning protocol
as above (confidence threshold tuned on one half, scored on the other).

| | COCO detector (best case) | Open-vocabulary teacher | Distilled student |
|---|---|---|---|
| Recall | 0.45 | 0.81 | 0.71 |
| Precision | 0.67 | 0.61 | 0.68 |
| F1 | 0.54 | 0.69 | 0.70 |
| Model inference | n/a | slow (about 0.5 s per photo with the COCO detector in the same run) | 5.5 ms per photo on a T4 GPU (24 ms including Python overhead) |

The student matches the teacher's F1 with higher precision and lower recall, at a small
fraction of the cost, and clearly beats the COCO detector. Per-type recall for the student:
sofa 0.93, chair 0.82, bed 0.93, clock 1.0, vase 0.91, lamp 0.95, rug 0.85, painting 0.80,
ottoman 0.80, oven 0.83, refrigerator 0.81, dishwasher 0.67, plant 0.63, cabinet 0.57,
table 0.44, bookshelf 0.28, wardrobe 0.0, television 0.0.

Weak spots, stated plainly: wardrobes are not detected at all, bookshelves and tables are
poor, and television shows 0.0 (for every detector, so the cause, few positives or a label
mismatch, is still unchecked and nothing is claimed about it). The student can only learn
what the teacher labelled, so these gaps are inherited and are the next thing to improve,
for example with better prompts for wardrobe and bookshelf or a small set of human-checked
labels. Scores are on ADE20K photos; the student's own validation mAP on teacher labels is
not reported as accuracy because it measures agreement with the teacher, not with people.

## Design consequences

- Colour comes from a wall segmentation model, in CIELAB, instead of K-Means over the whole frame.
- Detection uses an open-vocabulary teacher distilled into a furniture-specific
  detector, since half of v1's checklist has no class at all.
- Recommendations are conditioned on what is detected, with a budget solver.

## Caveats

Benchmark photos are not the same as user uploads, so these are benchmark numbers. Item
presence uses a 0.5% pixel-area threshold. Labels have not yet been spot-checked by eye.
