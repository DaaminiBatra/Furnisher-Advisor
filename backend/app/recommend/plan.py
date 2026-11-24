"""Room plan: what is missing, what each missing piece costs, and the best set for a budget.

Prices: the open product data (ABO) has no prices, so budgets work on rough INR price bands
per item type (low / typical / high).  They are estimates for planning, shown as ranges and
labelled as such, and are replaced by live prices when a retailer feed is connected.
"""

from __future__ import annotations

from dataclasses import dataclass

# item -> (low, typical, high) rough INR for a single piece
PRICE_BANDS = {
    "sofa": (12000, 28000, 65000), "chair": (2500, 6000, 18000), "bed": (10000, 24000, 60000),
    "table": (3500, 9000, 28000), "lamp": (800, 2200, 7000), "bookshelf": (3000, 7500, 20000),
    "wardrobe": (9000, 20000, 50000), "rug": (1500, 5000, 20000), "painting": (600, 2000, 9000),
    "cabinet": (4000, 11000, 30000), "ottoman": (1500, 3800, 11000), "vase": (400, 1200, 4500),
    "clock": (500, 1500, 6000), "plant": (300, 900, 3500), "tv": (4000, 9000, 22000),
}

# room -> {item: importance weight}; higher = more worth buying first
ROOMS = {
    "living room": {"sofa": 10, "table": 7, "rug": 5, "lamp": 5, "tv": 6, "bookshelf": 4,
                    "painting": 3, "plant": 3, "ottoman": 3, "vase": 2, "clock": 2, "chair": 4},
    "bedroom": {"bed": 10, "wardrobe": 8, "lamp": 5, "rug": 4, "painting": 3, "chair": 3,
                "plant": 2, "clock": 2, "table": 4},
    "dining room": {"table": 10, "chair": 9, "cabinet": 5, "lamp": 4, "rug": 3, "painting": 3,
                    "vase": 2, "plant": 2, "clock": 2},
    "kitchen": {"table": 6, "chair": 5, "cabinet": 7, "clock": 2, "plant": 2, "vase": 1},
    "study": {"table": 10, "chair": 9, "bookshelf": 7, "lamp": 6, "cabinet": 4, "plant": 3,
              "clock": 2, "painting": 2},
}
ROOMS["office"] = ROOMS["study"]
ROOMS["library"] = {"bookshelf": 10, "chair": 7, "table": 6, "lamp": 6, "rug": 3, "plant": 2}

TIERS = ("low", "typical", "high")
TIER_VALUE = {"low": 0.55, "typical": 0.85, "high": 1.0}  # diminishing returns on spending more


@dataclass(frozen=True)
class Choice:
    item: str
    tier: str
    price: int


def missing_items(room_type: str, present: set[str]) -> list[str]:
    weights = ROOMS.get(room_type.lower().strip())
    if weights is None:
        raise KeyError(f"unknown room type: {room_type}")
    return sorted((i for i in weights if i not in present), key=lambda i: -weights[i])


def best_set(room_type: str, present: set[str], budget_inr: int, step: int = 100) -> list[Choice]:
    """Exact 0/1 multiple-choice knapsack: for each missing item pick none or one price tier
    to maximise sum(importance * tier value) within the budget."""
    weights = ROOMS[room_type.lower().strip()]
    items = missing_items(room_type, present)
    cap = max(0, budget_inr) // step
    # dp[c] = (value, chosen list) using at most c budget units
    dp: list[tuple[float, tuple[Choice, ...]]] = [(0.0, ())] * (cap + 1)
    for item in items:
        new = list(dp)
        for tier, price in zip(TIERS, PRICE_BANDS[item]):
            cost = -(-price // step)
            if cost > cap:
                continue
            gain = weights[item] * TIER_VALUE[tier]
            for c in range(cost, cap + 1):
                v, chosen = dp[c - cost]
                if v + gain > new[c][0] + 1e-9:
                    new[c] = (v + gain, chosen + (Choice(item, tier, price),))
        dp = new
    return list(max(dp, key=lambda t: t[0])[1])
