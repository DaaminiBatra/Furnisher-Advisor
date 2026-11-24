"""Rank catalogue products of one type for a wall colour."""

from __future__ import annotations

import json
import urllib.request
from functools import lru_cache
from pathlib import Path

from .harmony import harmony_score


@lru_cache
def load_catalog(source: str) -> tuple[dict, ...]:
    """Catalogue is JSON lines; `source` is a local path or an http(s) URL."""
    if source.startswith(("http://", "https://")):
        with urllib.request.urlopen(source, timeout=60) as r:
            text = r.read().decode("utf-8")
    else:
        text = Path(source).read_text(encoding="utf-8")
    return tuple(json.loads(line) for line in text.splitlines() if line.strip())


def top_products(catalog, item_type: str, wall_rgb, k: int = 3) -> list[dict]:
    scored = [(harmony_score(p["rgb"], wall_rgb), p) for p in catalog if p["type"] == item_type]
    scored.sort(key=lambda t: -t[0])
    out, seen = [], set()
    for s, p in scored:
        key = (p.get("brand"), p["title"][:40])
        if key in seen:
            continue
        seen.add(key)
        out.append({**p, "harmony": s})
        if len(out) == k:
            break
    return out
