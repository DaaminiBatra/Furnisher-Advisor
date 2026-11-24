"""Look at an NYUv2 copy before trusting it: columns, sizes, label ids, and a montage
that overlays the biggest label ids on a few photos so the wall id can be confirmed by eye.

    python -m eval.kaggle.nyu_inspect --repo wyrx/NYUDepthv2_seg
"""

from __future__ import annotations

import argparse

import numpy as np
from datasets import load_dataset
from PIL import Image, ImageDraw


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="wyrx/NYUDepthv2_seg")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--out", default="/kaggle/working/nyu_montage.png")
    args = ap.parse_args()

    ds = load_dataset(args.repo)
    print(ds)
    split = next(iter(ds))
    d = ds[split]
    print("split:", split, "rows:", len(d), "features:", d.features)

    tiles = []
    for i in range(args.n):
        row = d[i]
        img = np.array(row["image"].convert("RGB"))
        lab = np.array(row["label"])
        if lab.ndim == 3:
            lab = lab[..., 0]
        ids, counts = np.unique(lab, return_counts=True)
        top = sorted(zip(counts, ids), reverse=True)[:4]
        print(i, img.shape, lab.dtype, "top label ids (id:share):",
              [(int(k), round(float(c) / lab.size, 3)) for c, k in top])
        row_tiles = [Image.fromarray(img).resize((240, 180))]
        for _, k in top[:3]:
            ov = img.copy()
            m = lab == k
            ov[m] = (0.4 * ov[m] + 0.6 * np.array([255, 0, 0])).astype(np.uint8)
            t = Image.fromarray(ov).resize((240, 180))
            ImageDraw.Draw(t).text((4, 4), f"id {int(k)}", fill=(255, 255, 0))
            row_tiles.append(t)
        tiles.append(row_tiles)
    W, H = 240 * 4, 180 * args.n
    sheet = Image.new("RGB", (W, H))
    for r, row_tiles in enumerate(tiles):
        for c, t in enumerate(row_tiles):
            sheet.paste(t, (c * 240, r * 180))
    sheet.save(args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
