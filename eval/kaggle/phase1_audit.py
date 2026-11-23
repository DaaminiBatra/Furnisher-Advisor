"""Phase 1 audit runner (meant for a Kaggle notebook with internet on).

    python -m eval.kaggle.phase1_audit --per-room 60 --out /kaggle/working/phase1_results.json

Steps: download ADE20K SceneParsing (validation split used), keep home-interior scenes,
run the ORIGINAL v1 detector (COCO YOLOv5x, conf 0.25, v1's two renames) and v1's
K-Means wall-colour method, and score both against ADE20K's human labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from eval.audit.room_items import ITEM_TO_ADE, items_present, scene_to_room, score
from eval.audit.v1_coverage import V1_RENAMES, V1_ROOM_FURNITURE
from eval.audit.wall_colour import evaluate_image, summarise

ADE_URL = "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip"


def download_ade(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    target = root / "ADEChallengeData2016"
    if target.exists():
        return target
    z = root / "ade.zip"
    if not z.exists():
        print("downloading ADE20K (~1 GB)...", flush=True)
        urllib.request.urlretrieve(ADE_URL, z)
    print("extracting validation split + metadata...", flush=True)
    with zipfile.ZipFile(z) as zf:
        for n in zf.namelist():
            if ("validation" in n) or n.endswith(("objectInfo150.csv", "sceneCategories.txt")):
                zf.extract(n, root)
    return target


OBJECT_INFO_URL = "https://raw.githubusercontent.com/CSAILVision/sceneparsing/master/objectInfo150.csv"


def load_ade_meta(ade: Path):
    names = {}  # idx -> list of synonyms
    if not (ade / "objectInfo150.csv").exists():  # not shipped inside the zip
        urllib.request.urlretrieve(OBJECT_INFO_URL, ade / "objectInfo150.csv")
    with open(ade / "objectInfo150.csv", newline="") as f:
        for row in csv.DictReader(f):
            names[int(row["Idx"])] = [s.strip() for s in row["Name"].split(";")]
    scenes = {}
    with open(ade / "sceneCategories.txt") as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2:
                scenes[parts[0]] = parts[1]
    return names, scenes


def class_fractions(ann: np.ndarray, idx_names: dict[int, list[str]]) -> dict[str, float]:
    total = ann.size
    out: dict[str, float] = defaultdict(float)
    for idx, count in zip(*np.unique(ann, return_counts=True)):
        if idx == 0:
            continue
        for syn in idx_names.get(int(idx), []):
            out[syn] = max(out[syn], count / total)
    return out


def load_v1_detector():
    import torch

    model = torch.hub.load("ultralytics/yolov5", "yolov5x", pretrained=True)  # COCO, conf 0.25
    return model


def v1_labels(model, image_rgb: np.ndarray) -> set[str]:
    df = model(image_rgb).pandas().xyxy[0]
    return {V1_RENAMES.get(n, n) for n in df["name"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-room", type=int, default=60)
    ap.add_argument("--root", default="/kaggle/working/data")
    ap.add_argument("--out", default="/kaggle/working/phase1_results.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ade = download_ade(Path(args.root))
    idx_names, scenes = load_ade_meta(ade)
    wall_idx = next(i for i, n in idx_names.items() if "wall" in n)
    print("wall idx:", wall_idx, "| scene types in val:", len({s for k, s in scenes.items() if "val" in k}))

    val_scenes = {k: s for k, s in scenes.items() if "_val_" in k}
    by_room: dict[str, list[str]] = defaultdict(list)
    for stem, scene in val_scenes.items():
        room = scene_to_room(scene)
        if room:
            by_room[room].append(stem)
    rng = random.Random(args.seed)
    chosen: list[tuple[str, str]] = []
    for room in V1_ROOM_FURNITURE:
        stems = sorted(by_room.get(room, []))
        rng.shuffle(stems)
        print(f"{room}: {len(stems)} validation images available", flush=True)
        chosen += [(room, s) for s in stems[: args.per_room]]
    print("selected", len(chosen), "images", flush=True)

    model = load_v1_detector()
    item_records, wall_rows, errors = [], [], 0
    t0 = time.time()
    for n, (room, stem) in enumerate(chosen, 1):
        try:
            img = np.array(Image.open(ade / "images/validation" / f"{stem}.jpg").convert("RGB"))
            ann = np.array(Image.open(ade / "annotations/validation" / f"{stem}.png"))
            fr = class_fractions(ann, idx_names)
            item_records.append(
                {"room": room, "present": items_present(fr), "detected": v1_labels(model, img)}
            )
            wall_mask = ann == wall_idx
            if wall_mask.mean() >= 0.15:  # enough labelled wall to define a true colour
                wall_rows.append(evaluate_image(img, wall_mask))
        except Exception as e:  # keep going; report the count
            errors += 1
            print("skip", stem, repr(e)[:120], flush=True)
        if n % 20 == 0:
            print(f"{n}/{len(chosen)} done, {time.time()-t0:.0f}s", flush=True)

    result = {
        "meta": {"images": len(item_records), "skipped": errors, "per_room_cap": args.per_room,
                 "dataset": "ADE20K SceneParsing validation", "detector": "YOLOv5x COCO conf 0.25 (v1 renames)"},
        "room_items": score(item_records),
        "wall_colour": summarise(wall_rows) if wall_rows else None,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, default=lambda o: sorted(o) if isinstance(o, set) else str(o)))
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    sys.exit(main())
