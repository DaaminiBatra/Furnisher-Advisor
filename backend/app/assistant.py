"""Tool-using chat assistant on Gemini.

The model never invents products or prices: it can only call the tools below, and the
answer has to be grounded in what they return.  Tools run server-side with the caller's own
permissions (a board is only readable by its members), arguments are validated, and the loop
is capped.  `generate` is injectable so the whole loop is tested without network access.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from collections.abc import Callable

from .recommend.plan import PRICE_BANDS, ROOMS, best_set, missing_items
from .recommend.rank import top_products

log = logging.getLogger(__name__)

SYSTEM = (
    "You are the Furnisher Advisor assistant. Help people furnish a room. Use the tools for "
    "anything about rooms, products, budgets or boards; never invent products, prices or "
    "colours. Prices from tools are rough planning ranges unless a product price is given: "
    "say so. Be brief and concrete (a few sentences, short lists). If a tool returns an "
    "error, explain it plainly and ask for what is missing."
)

TOOLS = [
    {"name": "plan_room", "description": "Budget plan for the furniture missing from a room.",
     "parameters": {"type": "OBJECT", "properties": {
         "room_type": {"type": "STRING", "description": "one of: " + ", ".join(sorted(ROOMS))},
         "wall_rgb": {"type": "ARRAY", "items": {"type": "INTEGER"}, "description": "[r,g,b] 0-255"},
         "present_items": {"type": "ARRAY", "items": {"type": "STRING"}},
         "budget_inr": {"type": "INTEGER"}},
         "required": ["room_type", "wall_rgb", "budget_inr"]}},
    {"name": "find_products", "description": "Best catalogue products of one type for a wall colour.",
     "parameters": {"type": "OBJECT", "properties": {
         "item_type": {"type": "STRING", "description": "one of: " + ", ".join(sorted(PRICE_BANDS))},
         "wall_rgb": {"type": "ARRAY", "items": {"type": "INTEGER"}},
         "count": {"type": "INTEGER"}},
         "required": ["item_type", "wall_rgb"]}},
    {"name": "board_decision", "description": "The group's current proposal for the board in this chat.",
     "parameters": {"type": "OBJECT", "properties": {}}},
]


class ToolError(Exception):
    pass


def _rgb(v) -> list[int]:
    if not (isinstance(v, list) and len(v) == 3 and all(isinstance(c, int) and 0 <= c <= 255 for c in v)):
        raise ToolError("wall_rgb must be [r,g,b] with integers 0-255")
    return v


def run_tool(name: str, args: dict, catalog, board_decision_fn: Callable[[], dict] | None) -> dict:
    try:
        if name == "plan_room":
            room = str(args.get("room_type", ""))
            if room.lower().strip() not in ROOMS:
                raise ToolError(f"unknown room type; choose from {sorted(ROOMS)}")
            _rgb(args.get("wall_rgb"))
            budget = int(args.get("budget_inr", -1))
            if not 0 <= budget <= 10_000_000:
                raise ToolError("budget_inr must be between 0 and 10,000,000")
            present = {str(i).lower().strip() for i in args.get("present_items") or []}
            chosen = best_set(room, present, budget)
            return {
                "missing": missing_items(room, present),
                "plan": [{"item": c.item, "tier": c.tier, "estimated_price_inr": c.price,
                          "products": [{"title": p["title"], "colour": p.get("colour_words"),
                                        "harmony": p["harmony"]}
                                       for p in top_products(catalog, c.item, args["wall_rgb"], 2)]}
                         for c in chosen],
                "estimated_total_inr": sum(c.price for c in chosen),
                "note": "prices are rough planning ranges, not shop prices",
            }
        if name == "find_products":
            item = str(args.get("item_type", ""))
            if item not in PRICE_BANDS:
                raise ToolError(f"unknown item_type; choose from {sorted(PRICE_BANDS)}")
            _rgb(args.get("wall_rgb"))
            k = max(1, min(5, int(args.get("count", 3))))
            found = top_products(catalog, item, args["wall_rgb"], k)
            return {"products": [{"title": p["title"], "brand": p.get("brand"),
                                  "colour": p.get("colour_words"), "harmony": p["harmony"]}
                                 for p in found] or "no products of this type in the catalogue"}
        if name == "board_decision":
            if board_decision_fn is None:
                raise ToolError("no board is attached to this chat")
            return board_decision_fn()
        raise ToolError(f"unknown tool {name}")
    except ToolError as e:
        return {"error": str(e)}
    except (TypeError, ValueError) as e:
        return {"error": f"bad arguments: {e}"}


def gemini_generate(api_key: str, model: str, contents: list[dict]) -> dict:
    body = json.dumps({
        "systemInstruction": {"parts": [{"text": SYSTEM}]},
        "contents": contents,
        "tools": [{"functionDeclarations": TOOLS}],
    }).encode()
    req = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        data=body, headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:300]  # error text never contains the key
        log.error("Gemini HTTP %s for model %s: %s", e.code, model, detail)
        raise RuntimeError(f"Gemini returned HTTP {e.code}") from None
    except (urllib.error.URLError, TimeoutError) as e:
        log.error("Gemini request failed: %s", e)
        raise RuntimeError("Gemini unreachable") from None


def chat(messages: list[dict], generate: Callable[[list[dict]], dict], catalog,
         board_decision_fn=None, max_rounds: int = 4) -> dict:
    """messages: [{"role": "user"|"assistant", "text": str}]. Returns reply + tool calls made."""
    contents = [{"role": "model" if m["role"] == "assistant" else "user",
                 "parts": [{"text": m["text"]}]} for m in messages]
    used: list[dict] = []
    for _ in range(max_rounds):
        resp = generate(contents)
        cands = resp.get("candidates") or []
        if not cands:
            return {"reply": "I couldn't produce an answer. Please try rephrasing.", "tool_calls": used}
        parts = (cands[0].get("content") or {}).get("parts") or []
        calls = [p["functionCall"] for p in parts if "functionCall" in p]
        if not calls:
            text = "".join(p.get("text", "") for p in parts).strip()
            return {"reply": text or "I don't have an answer for that.", "tool_calls": used}
        contents.append({"role": "model", "parts": parts})
        replies = []
        for c in calls:
            args = c.get("args") or {}
            used.append({"name": c["name"], "args": args})
            replies.append({"functionResponse": {
                "name": c["name"],
                "response": run_tool(c["name"], args, catalog, board_decision_fn)}})
        contents.append({"role": "user", "parts": replies})
    return {"reply": "That needed too many steps. Could you ask for one thing at a time?",
            "tool_calls": used}
