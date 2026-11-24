"""Build the furniture catalogue (JSON lines) from ABO listings.

    python -m catalog.build_catalog --out /kaggle/working/catalog.jsonl [--upload]

Per product (colour = listing colour word, photo measurement as fallback): type (one of our room-item names), title, brand, material, style, dimensions,
listing colour words, catalogue image URL, a measured dominant colour (trimmed median in
CIELAB over non-background pixels of the product photo) and the ASIN.  Kept per type:
at most `--per-type` products, preferring ones with an Indian or US English title and a colour.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import os
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from .abo import BASE, download, first_value, iter_listings
from .colour_names import colour_from_words

# ABO product_type -> our room-item name.  Appliances (fridge, oven, dishwasher), TVs and
# live plants are not furniture in ABO, so they are not in this catalogue.
TYPE_MAP = {
    "SOFA": "sofa", "CHAIR": "chair", "STOOL_SEATING": "chair", "BED": "bed",
    "TABLE": "table", "DESK": "table", "LAMP": "lamp", "LIGHT_FIXTURE": "lamp",
    "BOOKCASE": "bookshelf", "SHELF": "bookshelf", "RUG": "rug", "WALL_ART": "painting",
    "OTTOMAN": "ottoman", "CABINET": "cabinet", "DRESSER": "wardrobe", "WARDROBE": "wardrobe",
    "VASE": "vase", "CLOCK": "clock", "PLANTER": "plant", "TV_STAND": "tv",
}


def item_colour_rgb(img: Image.Image):
    """Trimmed-median colour of the foreground (product photos sit on white)."""
    from skimage.color import lab2rgb, rgb2lab

    arr = np.asarray(img.convert("RGB"))
    mask = arr.min(axis=2) < 238  # drop white background
    if mask.sum() < 400:
        return None
    lab = rgb2lab(arr[mask][None, :, :] / 255.0)[0]
    lo, hi = np.percentile(lab, [20, 80], axis=0)
    keep = ((lab >= lo) & (lab <= hi)).all(axis=1)
    sel = lab[keep] if keep.sum() >= 50 else lab
    med = np.median(sel, axis=0)
    rgb = lab2rgb(med[None, None, :])[0, 0]
    return [int(round(c * 255)) for c in rgb]


def fetch_image(path: str):
    try:
        with urllib.request.urlopen(f"{BASE}/images/small/{path}", timeout=30) as r:
            return Image.open(io.BytesIO(r.read()))
    except Exception:
        return None


def image_paths() -> dict[str, str]:
    raw = gzip.decompress(download("images/metadata/images.csv.gz")).decode()
    return {row["image_id"]: row["path"] for row in csv.DictReader(io.StringIO(raw))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="catalog.jsonl")
    ap.add_argument("--per-type", type=int, default=600)
    ap.add_argument("--upload", action="store_true")
    args = ap.parse_args()

    paths = image_paths()
    cand = defaultdict(list)
    for it in iter_listings():
        pts = [p.get("value") for p in it.get("product_type") or []]
        ours = next((TYPE_MAP[p] for p in pts if p in TYPE_MAP), None)
        img_id = it.get("main_image_id")
        name = first_value(it.get("item_name"))
        if not (ours and name and img_id in paths):
            continue
        colour = first_value(it.get("color"))
        score = (it.get("marketplace") == "Amazon") + (colour is not None) + (it.get("country") in ("IN", "US"))
        cand[ours].append((score, it, name, colour, paths[img_id]))
    print({k: len(v) for k, v in cand.items()})

    picked = []
    for ours, rows in cand.items():
        rows.sort(key=lambda r: -r[0])
        picked += [(ours, *r[1:]) for r in rows[: args.per_type]]

    def work(row):
        ours, it, name, colour, path = row
        img = fetch_image(path)
        rgb = item_colour_rgb(img) if img is not None else None
        word_rgb = colour_from_words(colour)
        if rgb is None and word_rgb is None:
            return None
        dims = it.get("item_dimensions") or {}
        return {
            "asin": it.get("item_id"), "type": ours, "title": name[:160],
            "brand": first_value(it.get("brand")), "colour_words": colour,
            "material": first_value(it.get("material")), "style": first_value(it.get("style")),
            "dims_cm": {k: round(v["normalized_value"]["value"] * 2.54, 1) for k, v in dims.items()
                        if isinstance(v, dict) and v.get("normalized_value", {}).get("unit") == "inches"},
            "image": f"{BASE}/images/small/{path}", "rgb": word_rgb or rgb,
            "rgb_source": "listing" if word_rgb else "photo", "rgb_photo": rgb,
        }

    with ThreadPoolExecutor(16) as ex:
        out = [r for r in ex.map(work, picked) if r]
    Path(args.out).write_text("\n".join(json.dumps(r) for r in out) + "\n")
    print("items", len(out), Counter(r["type"] for r in out))

    if args.upload:
        token = os.environ.get("HF_TOKEN")
        if not token:
            from kaggle_secrets import UserSecretsClient
            token = UserSecretsClient().get_secret("HF_TOKEN")
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        repo = f"{api.whoami()['name']}/furnisher-advisor-catalog"
        api.create_repo(repo, repo_type="dataset", private=False, exist_ok=True)
        api.upload_file(path_or_fileobj=args.out, path_in_repo="catalog.jsonl", repo_id=repo, repo_type="dataset")
        print("uploaded to", repo)


if __name__ == "__main__":
    main()
