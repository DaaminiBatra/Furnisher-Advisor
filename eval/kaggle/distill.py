"""Phase 3a: distil the open-vocabulary detector into a fast furniture detector.

    python -m eval.kaggle.distill --n-train 1500 --epochs 30

1. Pseudo-label ADE20K *training* photos of home interiors with OWLv2 (teacher).
2. Train a small YOLO (student) on those boxes.
3. Evaluate the student on the ADE20K *validation* photos (never used for training) with
   the same presence metric and held-out threshold protocol as phase2_detect.

No human box annotations are used: the teacher's boxes are the only supervision, so the
student can at best approach the teacher; the value is speed and a closed, fixed class list.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from eval.audit.detect_metrics import (CANONICAL, best_threshold, present_items, score_sets,
                                       sets_at_threshold)
from eval.audit.room_items import scene_to_room
from eval.kaggle.phase1_audit import class_fractions, download_ade, load_ade_meta
from eval.kaggle.phase2_detect import OWL, PROMPTS

NAMES = list(CANONICAL)


def extract_train(root: Path, stems: list[str]) -> None:
    want = {f"ADEChallengeData2016/images/training/{s}.jpg" for s in stems}
    with zipfile.ZipFile(root / "ade.zip") as zf:
        for n in zf.namelist():
            if n in want and not (root / n).exists():
                zf.extract(n, root)


def pseudo_label(root: Path, stems: list[str], out: Path, thr: float, dev: str) -> dict:
    from torchvision.ops import batched_nms
    from transformers import Owlv2ForObjectDetection, Owlv2Processor

    proc = Owlv2Processor.from_pretrained(OWL)
    owl = Owlv2ForObjectDetection.from_pretrained(OWL).to(dev).eval()
    queries = [[PROMPTS[n] for n in NAMES]]
    counts = defaultdict(int)
    for split, ids in (("train", stems[: int(len(stems) * 0.9)]), ("val", stems[int(len(stems) * 0.9):])):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for k, stem in enumerate(ids, 1):
            src = root / "ADEChallengeData2016/images/training" / f"{stem}.jpg"
            im = Image.open(src).convert("RGB")
            w, h = im.size
            with torch.no_grad():
                inp = proc(text=queries, images=im, return_tensors="pt").to(dev)
                o = owl(**inp)
                prob = torch.sigmoid(o.logits[0])            # (boxes, queries)
                scores, labels = prob.max(1)                   # best query per box
                cxcywh = o.pred_boxes[0]                     # normalised to the padded square
                side = float(max(w, h))
                boxes = torch.stack([(cxcywh[:, 0] - cxcywh[:, 2] / 2) * side, (cxcywh[:, 1] - cxcywh[:, 3] / 2) * side,
                                     (cxcywh[:, 0] + cxcywh[:, 2] / 2) * side, (cxcywh[:, 1] + cxcywh[:, 3] / 2) * side], 1)
                sel = scores >= thr
                boxes, scores, labels = boxes[sel], scores[sel], labels[sel]
            if len(boxes):
                keep = batched_nms(boxes, scores, labels, 0.5)
                boxes, labels = boxes[keep].cpu().numpy(), labels[keep].cpu().numpy()
            lines = []
            for b, c in zip(boxes, labels):
                x0, y0, x1, y1 = np.clip(b, 0, [w, h, w, h])
                if x1 - x0 < 8 or y1 - y0 < 8:
                    continue
                lines.append(f"{int(c)} {(x0+x1)/2/w:.6f} {(y0+y1)/2/h:.6f} {(x1-x0)/w:.6f} {(y1-y0)/h:.6f}")
                counts[NAMES[int(c)]] += 1
            shutil.copy(src, out / "images" / split / f"{stem}.jpg")
            (out / "labels" / split / f"{stem}.txt").write_text("\n".join(lines))
            if k % 100 == 0:
                print(f"labelled {split} {k}/{len(ids)}", flush=True)
    (out / "data.yaml").write_text(
        f"path: {out}\ntrain: images/train\nval: images/val\nnames:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES)))
    return dict(counts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=1500)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--thr", type=float, default=0.3, help="teacher score threshold for pseudo-labels")
    ap.add_argument("--root", default="/kaggle/working/data")
    ap.add_argument("--work", default="/kaggle/working/distill")
    ap.add_argument("--per-room", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    root, work = Path(args.root), Path(args.work)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ade = download_ade(root)
    idx_names, scenes = load_ade_meta(ade)
    train_stems = sorted(s for s, sc in scenes.items() if "_train_" in s and scene_to_room(sc))
    random.Random(args.seed).shuffle(train_stems)
    train_stems = train_stems[: args.n_train]
    print("train photos selected:", len(train_stems), flush=True)
    extract_train(root, train_stems)

    t0 = time.time()
    counts = pseudo_label(root, train_stems, work / "ds", args.thr, dev)
    print("pseudo-label boxes per class:", counts, f"| {time.time()-t0:.0f}s", flush=True)

    from ultralytics import YOLO
    model = YOLO("yolov8s.pt")
    model.train(data=str(work / "ds/data.yaml"), epochs=args.epochs, imgsz=640, batch=16,
                project=str(work / "runs"), name="student", device=0 if dev == "cuda" else "cpu",
                verbose=False, plots=False)
    best = work / "runs/student/weights/best.pt"
    student = YOLO(str(best))

    # held-out evaluation on ADE20K validation photos, same selection/protocol as phase2_detect
    by_room = defaultdict(list)
    for stem, scene in scenes.items():
        if "_val_" in stem and (room := scene_to_room(scene)):
            by_room[room].append(stem)
    from eval.audit.v1_coverage import V1_ROOM_FURNITURE
    rng = random.Random(args.seed)
    chosen = []
    for room in V1_ROOM_FURNITURE:
        stems = sorted(by_room.get(room, []))
        rng.shuffle(stems)
        chosen += stems[: args.per_room]
    truth, scores = [], []
    t1 = time.time()
    for stem in chosen:
        img = np.array(Image.open(ade / "images/validation" / f"{stem}.jpg").convert("RGB"))
        ann = np.array(Image.open(ade / "annotations/validation" / f"{stem}.png"))
        truth.append(present_items(class_fractions(ann, idx_names)))
        r = student.predict(img, conf=0.05, verbose=False)[0]
        d = {n: 0.0 for n in NAMES}
        for c, s in zip(r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            d[NAMES[int(c)]] = max(d[NAMES[int(c)]], float(s))
        scores.append(d)
    ms_per_img = (time.time() - t1) / len(chosen) * 1000
    tune = [i for i in range(len(chosen)) if i % 2 == 0]
    test = [i for i in range(len(chosen)) if i % 2 == 1]
    pick = lambda xs, ids: [xs[i] for i in ids]
    thr = best_threshold(pick(truth, tune), pick(scores, tune))
    res = score_sets(pick(truth, test), sets_at_threshold(pick(scores, test), thr))
    out = {"student": "yolov8s", "train_photos": len(train_stems), "epochs": args.epochs,
           "pseudo_label_threshold": args.thr, "pseudo_boxes": counts, "conf_threshold": thr,
           "heldout": res, "ms_per_image_incl_python": ms_per_img}
    (work / "student_eval.json").write_text(json.dumps(out, indent=2))
    shutil.copy(best, "/kaggle/working/student_best.pt")
    print(json.dumps({"thr": thr, "micro": res["micro"], "ms_per_img": round(ms_per_img, 1),
                      "per_item_recall": {k: round(v["recall"], 2) for k, v in res["per_item"].items()}}))


if __name__ == "__main__":
    main()
