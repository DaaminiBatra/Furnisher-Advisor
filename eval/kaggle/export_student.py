"""Upload the distilled furniture detector to a Hugging Face model repo (private by default).

    python -m eval.kaggle.export_student

Reads HF_TOKEN from Kaggle Secrets (or the environment). The token is never printed.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

CARD = """---
library_name: ultralytics
tags: [object-detection, furniture, yolov8]
---
# Furnisher Advisor furniture detector

YOLOv8s detector for 18 furniture types, distilled from an open-vocabulary detector (OWLv2)
using its boxes as the only supervision on 1,500 ADE20K training photos of home interiors.

Classes: {classes}

Held-out presence scores on ADE20K validation photos: precision {p:.2f}, recall {r:.2f},
F1 {f:.2f} (COCO detector baseline F1 0.54). Known weak spots: wardrobe, bookshelf, table,
television. See the project's docs/evaluation.md for method and caveats.

Licensing: trained with Ultralytics YOLOv8 (AGPL-3.0) on photos from ADE20K, which is
released for non-commercial research. Treat these weights as research use only.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="/kaggle/working/student_best.pt")
    ap.add_argument("--eval-json", default="/kaggle/working/distill/student_eval.json")
    ap.add_argument("--name", default="furnisher-advisor-detector")
    ap.add_argument("--public", action="store_true")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("HF_TOKEN")
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    user = api.whoami()["name"]
    repo = f"{user}/{args.name}"
    api.create_repo(repo, private=not args.public, exist_ok=True)

    ev = json.loads(Path(args.eval_json).read_text())["heldout"]["micro"]
    from eval.audit.detect_metrics import CANONICAL
    card = CARD.format(classes=", ".join(CANONICAL), p=ev["precision"], r=ev["recall"], f=ev["f1"])
    Path("/kaggle/working/README.md").write_text(card)
    api.upload_file(path_or_fileobj=args.weights, path_in_repo="student_best.pt", repo_id=repo)
    api.upload_file(path_or_fileobj="/kaggle/working/README.md", path_in_repo="README.md", repo_id=repo)
    print("uploaded to", repo, "| private:", not args.public)


if __name__ == "__main__":
    main()
