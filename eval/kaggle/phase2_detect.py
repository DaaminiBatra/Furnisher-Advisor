"""Phase 2c: furniture presence, COCO detector (v1's) vs open-vocabulary detector (OWLv2).

    python -m eval.kaggle.phase2_detect --per-room 60

Same ADE20K photos as phase 1; ground truth = ADE20K semantic labels (>= 0.5% of pixels).
COCO side gets its best case (class names mapped to our items directly, conf 0.25).
OWLv2 threshold is tuned on half the photos and scored on the other half, so the headline
numbers are held-out. Open-vocabulary prompts need no furniture-specific training.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from eval.audit.detect_metrics import (CANONICAL, COCO_TO_CANONICAL, best_threshold, present_items,
                                       score_sets, sets_at_threshold)
from eval.audit.room_items import scene_to_room
from eval.audit.v1_coverage import V1_ROOM_FURNITURE
from eval.kaggle.phase1_audit import class_fractions, download_ade, load_ade_meta

OWL = "google/owlv2-base-patch16-ensemble"
PROMPTS = {
    "sofa": "a sofa", "chair": "a chair", "bed": "a bed", "plant": "a potted plant",
    "clock": "a clock", "vase": "a vase", "table": "a table", "refrigerator": "a refrigerator",
    "oven": "an oven or stove", "lamp": "a lamp", "bookshelf": "a bookshelf",
    "wardrobe": "a wardrobe", "rug": "a rug or carpet", "tv": "a television",
    "painting": "a framed picture or painting", "cabinet": "a cabinet", "ottoman": "an ottoman",
    "dishwasher": "a dishwasher",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-room", type=int, default=60)
    ap.add_argument("--root", default="/kaggle/working/data")
    ap.add_argument("--out", default="/kaggle/working/phase2_detect.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ade = download_ade(Path(args.root))
    idx_names, scenes = load_ade_meta(ade)
    by_room = defaultdict(list)
    for stem, scene in scenes.items():
        if "_val_" in stem and (room := scene_to_room(scene)):
            by_room[room].append(stem)
    rng = random.Random(args.seed)
    chosen = []
    for room in V1_ROOM_FURNITURE:
        stems = sorted(by_room.get(room, []))
        rng.shuffle(stems)
        chosen += stems[: args.per_room]

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", dev, "| images:", len(chosen), flush=True)
    from transformers import Owlv2ForObjectDetection, Owlv2Processor

    proc = Owlv2Processor.from_pretrained(OWL)
    owl = Owlv2ForObjectDetection.from_pretrained(OWL).to(dev).eval()
    names = list(CANONICAL)
    queries = [[PROMPTS[n] for n in names]]
    yolo = torch.hub.load("ultralytics/yolov5", "yolov5x", pretrained=True, trust_repo=True)

    truth, coco_sets, owl_scores = [], [], []
    t0 = time.time()
    for n, stem in enumerate(chosen, 1):
        img = np.array(Image.open(ade / "images/validation" / f"{stem}.jpg").convert("RGB"))
        ann = np.array(Image.open(ade / "annotations/validation" / f"{stem}.png"))
        truth.append(present_items(class_fractions(ann, idx_names)))
        det = set(yolo(img).pandas().xyxy[0]["name"])
        coco_sets.append({COCO_TO_CANONICAL[d] for d in det if d in COCO_TO_CANONICAL})
        with torch.no_grad():
            inp = proc(text=queries, images=Image.fromarray(img), return_tensors="pt").to(dev)
            logits = owl(**inp).logits[0]          # (boxes, queries)
            best = torch.sigmoid(logits).max(0).values.cpu().numpy()
        owl_scores.append({k: float(best[i]) for i, k in enumerate(names)})
        if n % 50 == 0:
            print(f"{n}/{len(chosen)} {time.time()-t0:.0f}s", flush=True)

    tune = [i for i in range(len(chosen)) if i % 2 == 0]
    test = [i for i in range(len(chosen)) if i % 2 == 1]
    pick = lambda xs, ids: [xs[i] for i in ids]
    thr = best_threshold(pick(truth, tune), pick(owl_scores, tune))
    owl_test = score_sets(pick(truth, test), sets_at_threshold(pick(owl_scores, test), thr))
    coco_test = score_sets(pick(truth, test), pick(coco_sets, test))
    result = {
        "meta": {"images": len(chosen), "test_images": len(test), "owl_model": OWL,
                 "owl_threshold_tuned_on_other_half": thr, "ground_truth_min_fraction": 0.005},
        "coco_best_case_heldout": coco_test, "owlv2_heldout": owl_test,
        "items_coco_cannot_detect": [k for k in CANONICAL if k not in COCO_TO_CANONICAL.values()],
    }
    Path(args.out).write_text(json.dumps(result, indent=2))
    compact = {"thr": thr, "coco_micro": coco_test["micro"], "owl_micro": owl_test["micro"],
               "per_item_recall": {k: [round(coco_test["per_item"][k]["recall"], 2),
                                       round(owl_test["per_item"][k]["recall"], 2)] for k in CANONICAL}}
    print(json.dumps(compact))


if __name__ == "__main__":
    main()
