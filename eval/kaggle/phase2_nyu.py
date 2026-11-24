"""Phase 2b: out-of-distribution check of v2 wall colour on NYUv2 (not ADE20K).

    python -m eval.kaggle.phase2_nyu --limit 300

NYUv2 40-class labels (wall = 1, 0 = unlabelled; confirmed by eye with nyu_inspect).
Test split of wyrx/NYUDepthv2_seg, 640x480 RGB. Same scoring as phase 2a: both methods
vs the median colour of labelled wall pixels (CIEDE2000). The SegFormer model was NOT
trained on NYUv2 (it was trained on ADE20K), so this is an independent photo set,
though scene types overlap with ADE20K's indoor classes.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

from eval.audit.wall_colour import css_name_to_rgb, delta_e, true_wall_rgb, v1_closest_css_name, v1_dominant
from eval.audit.wall_v2 import mask_iou, v2_wall_rgb
from eval.kaggle.phase2_wall import MODEL, stats

NYU_WALL = 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="wyrx/NYUDepthv2_seg")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--out", default="/kaggle/working/phase2_nyu.json")
    args = ap.parse_args()

    ds = load_dataset(args.repo, split="test")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", dev, "| test rows:", len(ds), flush=True)
    proc = SegformerImageProcessor.from_pretrained(MODEL)
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL).to(dev).eval()
    wall_label = next(int(i) for i, n in model.config.id2label.items() if n.split(",")[0] == "wall")

    de_v1, de_v2, ious, seen = [], [], [], 0
    t0 = time.time()
    for i in range(len(ds)):
        if len(de_v2) >= args.limit:
            break
        row = ds[i]
        img = np.array(row["image"].convert("RGB"))
        lab = np.array(row["label"])
        if lab.ndim == 3:
            lab = lab[..., 0]
        truth = lab == NYU_WALL
        if truth.mean() < 0.15:
            continue
        seen += 1
        with torch.no_grad():
            inp = proc(images=img, return_tensors="pt").to(dev)
            up = torch.nn.functional.interpolate(model(**inp).logits, size=img.shape[:2],
                                                 mode="bilinear", align_corners=False)
            pred = (up.argmax(1)[0] == wall_label).cpu().numpy()
        if pred.sum() < 50:
            continue
        true_rgb = true_wall_rgb(img, truth)
        centre, _ = v1_dominant(img)
        de_v1.append(delta_e(css_name_to_rgb(v1_closest_css_name(centre)), true_rgb))
        de_v2.append(delta_e(v2_wall_rgb(img, pred), true_rgb))
        ious.append(mask_iou(pred, truth))
        if len(de_v2) % 50 == 0:
            print(f"{len(de_v2)} scored, {time.time()-t0:.0f}s", flush=True)

    result = {"meta": {"dataset": args.repo + " (NYUv2 test)", "scored": len(de_v2), "model": MODEL,
                       "wall_label": NYU_WALL},
              "wall_mask_iou_median": float(np.median(ious)),
              "v1_kmeans_css_name_dE": stats(de_v1), "v2_segformer_lab_dE": stats(de_v2),
              "share_images_v2_better": float((np.array(de_v2) < np.array(de_v1)).mean())}
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
