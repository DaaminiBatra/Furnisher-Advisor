"""Turn a group's votes into one proposal for the room.

Votes are love (+1), fine (0) or no (-1).  For each item type the winner is the pick with
the best consensus score = mean vote - 0.5 * standard deviation, so a pick everyone likes
beats one half the group loves and half hates.  Picks the group hasn't all voted on yet
count their missing votes as 0 (fine) but are flagged, and a type is flagged *contested*
when its two best picks are within 0.25 of each other or the winner got a "no" from anyone.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass


@dataclass
class PickResult:
    pick_id: int
    mean: float
    spread: float
    consensus: float
    votes_in: int
    vetoes: int


def score_pick(pick_id: int, scores: list[int], n_members: int) -> PickResult:
    padded = scores + [0] * max(0, n_members - len(scores))
    mean = statistics.fmean(padded) if padded else 0.0
    spread = statistics.pstdev(padded) if len(padded) > 1 else 0.0
    return PickResult(
        pick_id=pick_id, mean=round(mean, 3), spread=round(spread, 3),
        consensus=round(mean - 0.5 * spread, 3), votes_in=len(scores),
        vetoes=sum(1 for s in scores if s < 0),
    )


def decide(picks_by_type: dict[str, list[PickResult]]) -> dict[str, dict]:
    out = {}
    for item_type, results in picks_by_type.items():
        ranked = sorted(results, key=lambda r: (-r.consensus, r.vetoes, r.pick_id))
        win = ranked[0]
        runner = ranked[1] if len(ranked) > 1 else None
        contested = win.vetoes > 0 or (runner is not None and win.consensus - runner.consensus < 0.25)
        out[item_type] = {
            "winner": win.pick_id, "consensus": win.consensus, "contested": contested,
            "waiting_on_votes": win.votes_in, "ranking": [r.pick_id for r in ranked],
        }
    return out
