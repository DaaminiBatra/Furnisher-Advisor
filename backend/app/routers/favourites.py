from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import joinedload

from ..deps import CurrentUser, DbSession
from ..models import Item, SavedRecommendation, SubCategory
from ..schemas import (
    FavouriteSession,
    MessageOut,
    SaveColorsRequest,
    SaveFurnitureRequest,
    SavedColorOut,
    SavedFurnitureOut,
    SavedItemGroup,
)

router = APIRouter(prefix="/favourites", tags=["favourites"])


@router.post("/furniture", response_model=MessageOut, status_code=status.HTTP_201_CREATED)
def save_furniture(body: SaveFurnitureRequest, user: CurrentUser, db: DbSession) -> MessageOut:
    # Validate every reference up front so we never half-save a batch.
    item_ids = {r.item_id for r in body.recommendations}
    sub_ids = {r.subcategory_id for r in body.recommendations}
    found_items = set(db.scalars(select(Item.id).where(Item.id.in_(item_ids))))
    found_subs = {
        s.id: s.item_id
        for s in db.scalars(select(SubCategory).where(SubCategory.id.in_(sub_ids)))
    }
    for r in body.recommendations:
        if r.item_id not in found_items or found_subs.get(r.subcategory_id) != r.item_id:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                f"Unknown item/subcategory pair ({r.item_id}, {r.subcategory_id})",
            )

    for r in body.recommendations:
        db.add(
            SavedRecommendation(
                user_id=user.id,
                kind="furniture",
                room_type=body.room_type,
                session_ts=body.timestamp,
                item_id=r.item_id,
                subcategory_id=r.subcategory_id,
                link=r.link,
                color=r.color,
                estimated_price=r.estimated_price,
                set_budget=body.budget_set,
                original_image_path=body.original_image_path,
                processed_image_path=body.processed_image_path,
            )
        )
    db.commit()
    return MessageOut(message="Recommendations saved successfully")


@router.post("/colors", response_model=MessageOut, status_code=status.HTTP_201_CREATED)
def save_colors(body: SaveColorsRequest, user: CurrentUser, db: DbSession) -> MessageOut:
    for c in body.colors:
        if any(not 0 <= int(p) <= 255 for p in c.rgb_value.split(",")):
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "RGB values must be 0-255")
    for c in body.colors:
        db.add(
            SavedRecommendation(
                user_id=user.id,
                kind="color",
                room_type=body.room_type,
                session_ts=body.timestamp,
                color_name=c.color_name,
                rgb_value=c.rgb_value,
            )
        )
    db.commit()
    return MessageOut(message="Colors saved successfully")


@router.get("", response_model=list[FavouriteSession])
def list_favourites(user: CurrentUser, db: DbSession) -> list[FavouriteSession]:
    rows = db.scalars(
        select(SavedRecommendation)
        .where(SavedRecommendation.user_id == user.id)
        .options(joinedload(SavedRecommendation.item), joinedload(SavedRecommendation.sub_category))
        .order_by(SavedRecommendation.session_ts.desc(), SavedRecommendation.id)
    ).all()

    sessions: dict[tuple, FavouriteSession] = {}
    groups: dict[tuple, dict[str, SavedItemGroup]] = {}

    for row in rows:
        key = (row.session_ts, row.room_type)
        if key not in sessions:
            sessions[key] = FavouriteSession(
                timestamp=row.session_ts,
                room_type=row.room_type,
                original_image=row.original_image_path,
                processed_image=row.processed_image_path,
                saved_items=[],
                saved_colors=[],
            )
            groups[key] = {}
        s = sessions[key]
        if row.kind == "color":
            s.saved_colors.append(
                SavedColorOut(id=row.id, color_name=row.color_name, rgb_value=row.rgb_value)
            )
            continue
        name = row.item.name if row.item else "Removed item"
        group = groups[key].get(name)
        if group is None:
            group = SavedItemGroup(item_name=name, recommendations=[])
            groups[key][name] = group
            s.saved_items.append(group)
        sub = row.sub_category
        group.recommendations.append(
            SavedFurnitureOut(
                id=row.id,
                category=sub.name if sub else None,
                image=sub.image_path if sub else None,
                color=row.color,
                link=row.link,
                estimated_price=row.estimated_price,
                set_budget=row.set_budget,
            )
        )
    return list(sessions.values())


@router.delete("/{rec_id}", response_model=MessageOut)
def delete_favourite(rec_id: int, user: CurrentUser, db: DbSession) -> MessageOut:
    rec = db.scalar(
        select(SavedRecommendation).where(
            SavedRecommendation.id == rec_id, SavedRecommendation.user_id == user.id
        )
    )
    if rec is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Recommendation not found")
    db.delete(rec)
    db.commit()
    return MessageOut(message="Recommendation deleted successfully")
