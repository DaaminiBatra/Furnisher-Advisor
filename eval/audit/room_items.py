"""Score v1's 'what is missing from this room' decision against human labels.

v1 recommends (room checklist) - (labels its detector reported).  Ground truth comes
from ADE20K semantic labels: a checklist item counts as PRESENT in a photo when a
matching ADE class covers at least `min_fraction` of the image.

Only checklist items that have a clear ADE20K counterpart are scored; the rest are
reported as 'no ground truth' rather than guessed.
"""

from __future__ import annotations

from collections import defaultdict

from .v1_coverage import V1_ROOM_FURNITURE

# checklist item (as written in v1) -> ADE20K class synonyms (matched against the
# ';'-separated Name column of objectInfo150.csv).
ITEM_TO_ADE = {
    "sofa": ["sofa"],
    "chair": ["chair", "armchair", "swivel chair"],
    "bed": ["bed"],
    "potted plant": ["plant"],
    "clock": ["clock"],
    "vase": ["vase"],
    "table": ["table"],
    "dining table": ["table"],
    "refrigerator": ["refrigerator"],
    "oven": ["oven", "stove"],
    "lamp": ["lamp"],
    "bookshelf": ["bookcase", "shelf"],
    "wardrobe": ["wardrobe"],
    "carpet": ["rug"],
    "TV panel": ["television receiver"],
    "Picture Frames": ["painting"],
    "Cabinets": ["cabinet"],
    "Cabinet": ["cabinet"],
    "cabinets": ["cabinet"],
    "ottoman": ["ottoman"],
    "dishwasher": ["dishwasher"],
}

# ADE20K scene-name substring -> v1 room type.
SCENE_TO_ROOM = [
    ("bedroom", "Bed room"),
    ("living_room", "Living room"),
    ("dining_room", "Dining room"),
    ("kitchen", "Kitchen"),
    ("home_office", "Office"),
    ("office", "Office"),
    ("study", "Study room"),
    ("library", "Library"),
]


def scene_to_room(scene: str) -> str | None:
    for key, room in SCENE_TO_ROOM:
        if key in scene:
            return room
    return None


def items_present(class_fractions: dict[str, float], min_fraction: float = 0.005) -> set[str]:
    """Checklist items present, given {ADE synonym: fraction of image covered}."""
    present = set()
    for item, names in ITEM_TO_ADE.items():
        if any(class_fractions.get(n, 0.0) >= min_fraction for n in names):
            present.add(item)
    return present


def v1_recommendations(room: str, detected_labels: set[str]) -> set[str]:
    """Exactly v1's rule: checklist minus detected labels (case-sensitive)."""
    return set(V1_ROOM_FURNITURE[room]) - detected_labels


def score(records: list[dict]) -> dict:
    """records: {"room", "present": set[item], "detected": set[label]} per image.

    For each image, only checklist items with ground truth are scored:
      * recommended-but-present  -> v1 suggests buying something already in the room
      * missing-but-not-recommended -> v1 fails to suggest something the room lacks
    """
    per_room = defaultdict(lambda: defaultdict(int))
    for r in records:
        room = r["room"]
        scored = [i for i in V1_ROOM_FURNITURE[room] if i in ITEM_TO_ADE]
        recs = v1_recommendations(room, r["detected"])
        for item in scored:
            present = item in r["present"]
            recommended = item in recs
            c = per_room[room]
            c["item_checks"] += 1
            if recommended:
                c["recommended"] += 1
                if present:
                    c["recommended_but_present"] += 1
            if not present:
                c["truly_missing"] += 1
                if not recommended:
                    c["missing_not_recommended"] += 1
            if present and not recommended:
                c["present_and_skipped"] += 1
        per_room[room]["images"] += 1

    out, tot = {}, defaultdict(int)
    for room, c in per_room.items():
        out[room] = _rates(c)
        for k, v in c.items():
            tot[k] += v
    return {"per_room": out, "overall": _rates(tot)}


def _rates(c) -> dict:
    rec, miss = c["recommended"], c["truly_missing"]
    return {
        "images": c["images"],
        "item_checks": c["item_checks"],
        # Of everything v1 recommended, how much was already in the room?
        "share_recommended_already_present": (c["recommended_but_present"] / rec) if rec else None,
        # Of everything truly missing, how much did v1 fail to recommend?
        "share_missing_not_recommended": (c["missing_not_recommended"] / miss) if miss else None,
        "recommended": rec,
        "recommended_but_present": c["recommended_but_present"],
        "truly_missing": miss,
    }
