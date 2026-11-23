"""Audit of the v1 pipeline: which checklist items can its detector ever report?

v1 recommends  (room checklist) - (labels YOLOv5x detected).  The detector is the
COCO-trained YOLOv5x, so it can only ever output one of COCO's 80 class names,
after v1's two renames (bench -> table, couch -> sofa).  Matching is an exact
string comparison, so a checklist item can only be 'seen' if its text equals one of
those labels.  Everything else is recommended in every photo of that room type,
whatever the photo contains.

This needs no images: it is a property of the code, copied here from
flask-server/api/routes.py (v1) so the audit is reproducible.
"""

from __future__ import annotations

import json
import sys

# The 80 COCO classes the pretrained YOLOv5 checkpoint was trained on.
COCO80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]
assert len(COCO80) == 80 and len(set(COCO80)) == 80

V1_RENAMES = {"bench": "table", "couch": "sofa"}

# Copied verbatim from v1 (flask-server/api/routes.py, `room_furniture`).
V1_ROOM_FURNITURE = {
    "Living room": ["sofa", "side table", "chair", "TV panel", "bookshelf", "table", "lamp",
                    "vase", "ottoman", "potted plant", "clock", "Picture Frames", "carpet",
                    "Cabinets"],
    "Bed room": ["bed", "wardrobe", "dressing table", "side table", "chair", "lamp", "clock",
                 "Picture Frames", "vase", "potted plant", "table", "carpet"],
    "Dining room": ["dining table", "chair", "Cabinets", "napkin holder"],
    "Office": ["chair", "bookshelf", "Cabinet", "table", "lamp", "clock"],
    "Kitchen": ["Kitchen Cabinets doors", "refrigerator", "oven", "dishwasher",
                "food container", "cutlery"],
    "Study room": ["table", "chair", "bookshelf", "cabinets", "lamp", "clock"],
    "Library": ["bookshelf", "chair", "table", "clock"],
}


def possible_labels() -> set[str]:
    """Every string v1's detector can output after its renames."""
    return {V1_RENAMES.get(c, c) for c in COCO80}


def audit(rooms: dict[str, list[str]] | None = None) -> dict:
    rooms = rooms or V1_ROOM_FURNITURE
    labels = possible_labels()
    per_room = {}
    for room, items in rooms.items():
        # v1 compares item names to detected labels case-sensitively and exactly.
        seeable = [i for i in items if i in labels]
        blind = [i for i in items if i not in labels]
        per_room[room] = {
            "checklist_size": len(items),
            "can_be_detected": seeable,
            "never_detected": blind,
            "share_never_detected": round(len(blind) / len(items), 3),
        }
    total = sum(r["checklist_size"] for r in per_room.values())
    blind_total = sum(len(r["never_detected"]) for r in per_room.values())
    return {
        "per_room": per_room,
        "overall": {
            "checklist_slots": total,
            "never_detected_slots": blind_total,
            "share_never_detected": round(blind_total / total, 3),
        },
    }


def rename_only_items() -> dict[str, str]:
    """Labels v1's detector can only produce through one of its renames (new -> old).

    couch -> sofa is a faithful rename (same object). bench -> table is not: a real
    table is detected as 'dining table', which never equals the checklist's 'table',
    so for 'table' v1 can only ever be satisfied by a bench."""
    return {new: old for old, new in V1_RENAMES.items() if new not in COCO80}


def to_markdown(result: dict) -> str:
    lines = ["| Room | Checklist | Detectable | Never detected | Share blind |",
             "|---|---:|---:|---:|---:|"]
    for room, r in result["per_room"].items():
        lines.append(
            f"| {room} | {r['checklist_size']} | {len(r['can_be_detected'])} | "
            f"{len(r['never_detected'])} | {r['share_never_detected']:.0%} |"
        )
    o = result["overall"]
    lines.append(f"| **All rooms** | {o['checklist_slots']} | "
                 f"{o['checklist_slots'] - o['never_detected_slots']} | "
                 f"{o['never_detected_slots']} | {o['share_never_detected']:.0%} |")
    return "\n".join(lines)


if __name__ == "__main__":
    res = audit()
    if "--json" in sys.argv:
        print(json.dumps(res, indent=2))
    else:
        print(to_markdown(res))
        print()
        for room, r in res["per_room"].items():
            print(f"{room}: never detected -> {', '.join(r['never_detected']) or '-'}")
