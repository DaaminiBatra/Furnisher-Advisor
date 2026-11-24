# Research log

Dated entries, newest last. Dates are when the work actually happened.

- **2025** v1 built as a BTech major project (Flask, YOLOv5, K-Means colours, fixed budgets).
- **2026-11-23** Reviewed v1 critically: COCO classes cannot cover furniture, whole-image
  K-Means conflates wall with objects, budgets and links hard-coded. Designed v2.
- **2026-11-23** Phase 0: new FastAPI backend with auth, favourites and tests on SQLite
  and Postgres (branch `phase-0-backend`).
- **2026-11-24** Phase 1: built an evaluation harness and audited v1 on 253 labelled
  ADE20K photos. Findings in `docs/v1-audit.md`.
- **2026-11-24** Phase 2a: v2 wall colour (SegFormer + CIELAB). Median error fell from
  12.1 to 1.07 (CIEDE2000) on the same 210 photos. Caveat: scored on the dataset the
  segmenter was trained on; out-of-distribution check still to do.
- Next: out-of-distribution check of wall colour, then wall/floor segmentation and an open-vocabulary teacher for a furniture detector.
