"""Read the Amazon Berkeley Objects (ABO) product listings.

ABO is a public research dataset (https://amazon-berkeley-objects.s3.amazonaws.com/index.html,
Creative Commons, attribution required; the original release asks for non-commercial use).
It carries titles, product types, colours, materials, styles, dimensions and catalogue images,
but no prices, so prices in this project are always labelled as estimates.
"""

from __future__ import annotations

import gzip
import io
import json
import tarfile
import urllib.request
from collections import Counter

BASE = "https://amazon-berkeley-objects.s3.amazonaws.com"


def download(path: str) -> bytes:
    with urllib.request.urlopen(f"{BASE}/{path}", timeout=300) as r:
        return r.read()


def iter_listings():
    """Yield every listing dict from abo-listings.tar (30-odd gzipped JSON-lines files)."""
    blob = download("archives/abo-listings.tar")
    with tarfile.open(fileobj=io.BytesIO(blob)) as tar:
        for m in tar:
            if not m.name.endswith(".json.gz"):
                continue
            with gzip.open(tar.extractfile(m), "rt", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        yield json.loads(line)


def first_value(field, prefer=("en_IN", "en_US", "en_GB", "en_CA", "en_AU")):
    """ABO text fields are lists of {language_tag, value}; pick an English one."""
    if not field:
        return None
    if isinstance(field, str):
        return field
    for tag in prefer:
        for e in field:
            if e.get("language_tag") == tag and e.get("value"):
                return e["value"]
    for e in field:
        if e.get("value"):
            return e["value"]
    return None


def inspect(n: int = 40) -> None:
    types, markets = Counter(), Counter()
    total = 0
    for it in iter_listings():
        total += 1
        markets[it.get("marketplace")] += 1
        for pt in it.get("product_type") or []:
            types[pt.get("value")] += 1
    print("listings", total)
    print("marketplaces", markets.most_common(8))
    for t, c in types.most_common(n):
        print(f"{c:6d} {t}")


if __name__ == "__main__":
    inspect()
