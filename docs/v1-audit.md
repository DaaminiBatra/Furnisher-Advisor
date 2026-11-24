# Audit of v1 (Personalized Furnishing Advisor)

v1 was a 2025 BTech group project. This audit, run on 2026-11-24, measures where it
falls short, so that v2 is built against evidence rather than intuition.

**Method.** 253 indoor room photos from the ADE20K validation split (bedroom, living
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

## What this means for v2

- Colour must come from a wall segmentation model, in CIELAB, not a K-Means over the
  whole frame.
- Detection needs an open-vocabulary teacher distilled into a furniture-specific
  detector, since half of v1's checklist has no class at all.
- Recommendations must be conditioned on what is detected, with a budget solver.

## Caveats

ADE20K photos are not the same as user uploads, so these are benchmark numbers. Item
presence uses a 0.5% pixel-area threshold. Labels have not yet been spot-checked by eye.
