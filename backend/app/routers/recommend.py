from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..config import get_settings
from ..recommend.plan import PRICE_BANDS, ROOMS, TIERS, best_set, missing_items
from ..recommend.rank import load_catalog, top_products

router = APIRouter(prefix="/recommend", tags=["recommend"])

PRICE_NOTE = (
    "Prices are rough planning ranges per item type, not live shop prices: the open product "
    "data used for the catalogue has no prices."
)


def get_catalog() -> tuple[dict, ...]:
    source = get_settings().catalog_source
    if not source:
        return ()
    try:
        return load_catalog(source)
    except Exception:  # catalogue is optional; the plan still works without products
        return ()


class RecommendIn(BaseModel):
    room_type: str
    wall_rgb: list[int] = Field(min_length=3, max_length=3)
    present_items: list[str] = []
    budget_inr: int = Field(ge=0, le=10_000_000)
    products_per_item: int = Field(default=3, ge=0, le=6)


@router.get("/rooms")
def rooms() -> dict:
    return {"rooms": sorted(ROOMS)}


@router.post("")
def recommend(body: RecommendIn, catalog=Depends(get_catalog)) -> dict:
    if any(not 0 <= c <= 255 for c in body.wall_rgb):
        raise HTTPException(422, "wall_rgb values must be 0-255")
    present = {i.lower().strip() for i in body.present_items}
    try:
        missing = missing_items(body.room_type, present)
        chosen = best_set(body.room_type, present, body.budget_inr)
    except KeyError:
        raise HTTPException(404, f"unknown room type: {body.room_type}") from None
    plan = []
    for c in chosen:
        low, typical, high = PRICE_BANDS[c.item]
        plan.append({
            "item": c.item, "tier": c.tier, "estimated_price_inr": c.price,
            "typical_range_inr": [low, high],
            "products": top_products(catalog, c.item, body.wall_rgb, body.products_per_item),
        })
    chosen_items = {c.item for c in chosen}
    return {
        "room_type": body.room_type, "tiers": list(TIERS), "missing": missing,
        "plan": plan, "not_in_budget": [i for i in missing if i not in chosen_items],
        "estimated_total_inr": sum(c.price for c in chosen), "price_note": PRICE_NOTE,
    }
