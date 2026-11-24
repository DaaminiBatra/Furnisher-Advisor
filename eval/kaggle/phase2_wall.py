"""Phase 2a: v2 wall colour (SegFormer + CIELAB) vs v1 (K-Means + CSS name), same photos.

    python -m eval.kaggle.phase2_wall --per-room 60 --out /kaggle/working/phase2_wall.json

Reuses the ADE20K data already downloaded by phase1_audit. Both methods are scored
against the median colour of the human-labelled wall pixels (CIEDE2000).
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
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

from eval.audit.room_items import scene_to_room
from eval.audit.v1_coverage import V1_ROOM_FURNITURE
from eval.audit.wall_colour import delta_e, true_wall_rgb, v1_closest_css_name, v1_dominant, css_name_to_rgb
from eval.audit.wall_v2 import mask_iou, v2_wall_rgb
from eval.kaggle.phase1_audit import download_ade, load_ade_meta

MODEL = "nvidia/segformer-b2-finetuned-ade-512-512"  # ADE20K-trained, class 0 = wall


def pct(xs, q):
    return float(np.percentile(xs, q))


def stats(de):
    de = np.asarray(de)
    return {"n": len(de), "median": float(np.median(de)), "mean": float(de.mean()),
            "p90": pct(de, 90), "share_gt_5": float((de > 5).mean()), "share_gt_10": float((de > 10).mean())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-room", type=int, default=60)
    ap.add_argument("--root", default="/kaggle/working/data")
    ap.add_argument("--out", default="/kaggle/working/phase2_wall.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ade = download_ade(Path(args.root))
    idx_names, scenes = load_ade_meta(ade)
    wall_idx = next(i for i, n in idx_names.items() if "wall" in n)  # 1-based in annotations
    by_room = defaultdict(list)
    for stem, scene in scenes.items():
        if "_val_" in stem and (room := scene_to_room(scene)):
            by_room[room].append(stem)
    rng = random.Random(args.seed)  # same seed/ordering as phase 1 -> same photos
    chosen = []
    for room in V1_ROOM_FURNITURE:
        stems = sorted(by_room.get(room, []))
        rng.shuffle(stems)
        chosen += stems[: args.per_room]

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = SegformerImageProcessor.from_pretrained(MODEL)
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL).to(dev).eval()
    wall_label = next(int(i) for i, n in model.config.id2label.items() if n.split(",")[0] == "wall")

    de_v1, de_v2, ious, rows = [], [], [], 0
    t0 = time.time()
    for n, stem in enumerate(chosen, 1):
        img = np.array(Image.open(ade / "images/validation" / f"{stem}.jpg").convert("RGB"))
        ann = np.array(Image.open(ade / "annotations/validation" / f"{stem}.png"))
        truth = ann == wall_idx
        if truth.mean() < 0.15:
            continue
        with torch.no_grad():
            inp = proc(images=img, return_tensors="pt").to(dev)
            logits = model(**inp).logits
            up = torch.nn.functional.interpolate(logits, size=img.shape[:2], mode="bilinear", align_corners=False)
            pred = (up.argmax(1)[0] == wall_label).cpu().numpy()
        if pred.sum() < 50:
            pred = truth & False
            continue
        true_rgb = true_wall_rgb(img, truth)
        centre, _ = v1_dominant(img)
        v1_named = css_name_to_rgb(v1_closest_css_name(centre))
        de_v1.append(delta_e(v1_named, true_rgb))
        de_v2.append(delta_e(v2_wall_rgb(img, pred), true_rgb))
        ious.append(mask_iou(pred, truth))
        if n % 20 == 0:
            print(f"{n}/{len(chosen)} {time.time()-t0:.0f}s", flush=True)

    result = {"meta": {"model": MODEL, "images": len(de_v2)},
              "wall_mask_iou_median": float(np.median(ious)),
              "v1_kmeans_css_name_dE": stats(de_v1), "v2_segformer_lab_dE": stats(de_v2),
              "share_images_v2_better": float((np.array(de_v2) < np.array(de_v1)).mean())}
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
