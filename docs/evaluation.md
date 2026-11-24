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

## Design consequences

- Colour comes from a wall segmentation model, in CIELAB, instead of K-Means over the whole frame.
- Detection uses an open-vocabulary teacher distilled into a furniture-specific
  detector, since half of v1's checklist has no class at all.
- Recommendations are conditioned on what is detected, with a budget solver.

## Caveats

Benchmark photos are not the same as user uploads, so these are benchmark numbers. Item
presence uses a 0.5% pixel-area threshold. Labels have not yet been spot-checked by eye.
