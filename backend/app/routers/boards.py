import secrets
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select

from ..deps import CurrentUser, DbSession
from ..models import Board, BoardMember, BoardPick, BoardVote
from ..recommend.group import decide, score_pick
from ..recommend.plan import PRICE_BANDS, ROOMS

router = APIRouter(prefix="/boards", tags=["boards"])


class BoardIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    room_type: str
    wall_rgb: list[int] = Field(min_length=3, max_length=3)
    budget_inr: int = Field(ge=0, le=10_000_000)


class JoinIn(BaseModel):
    invite_code: str = Field(min_length=4, max_length=32)


class PickIn(BaseModel):
    item_type: str
    asin: str = Field(min_length=1, max_length=40)
    title: str = Field(min_length=1, max_length=200)
    image: str | None = None
    rgb: list[int] | None = Field(default=None, min_length=3, max_length=3)
    price_inr: int | None = Field(default=None, ge=0, le=10_000_000)
    note: str | None = Field(default=None, max_length=300)


class VoteIn(BaseModel):
    score: int = Field(ge=-1, le=1)


def _member_board(db, board_id: int, user) -> Board:
    board = db.get(Board, board_id)
    is_member = board is not None and db.scalar(
        select(BoardMember.id).where(BoardMember.board_id == board_id, BoardMember.user_id == user.id)
    )
    if not is_member:  # same answer for "missing" and "not yours": do not leak boards
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Board not found")
    return board


def _summary(board: Board) -> dict:
    return {
        "id": board.id, "name": board.name, "room_type": board.room_type,
        "wall_rgb": [int(v) for v in board.wall_rgb.split(",")], "budget_inr": board.budget_inr,
        "member_count": len(board.members), "pick_count": len(board.picks),
    }


@router.post("", status_code=status.HTTP_201_CREATED)
def create_board(body: BoardIn, user: CurrentUser, db: DbSession) -> dict:
    if body.room_type.lower().strip() not in ROOMS:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "Unknown room type")
    if any(not 0 <= c <= 255 for c in body.wall_rgb):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "wall_rgb values must be 0-255")
    board = Board(
        name=body.name, room_type=body.room_type.lower().strip(),
        wall_rgb=",".join(map(str, body.wall_rgb)), budget_inr=body.budget_inr,
        owner_id=user.id, invite_code=secrets.token_urlsafe(6),
    )
    board.members.append(BoardMember(user_id=user.id, role="owner"))
    db.add(board)
    db.commit()
    return {**_summary(board), "invite_code": board.invite_code}


@router.get("")
def my_boards(user: CurrentUser, db: DbSession) -> list[dict]:
    boards = db.scalars(
        select(Board).join(BoardMember).where(BoardMember.user_id == user.id).order_by(Board.id.desc())
    ).all()
    return [_summary(b) for b in boards]


@router.post("/join")
def join_board(body: JoinIn, user: CurrentUser, db: DbSession) -> dict:
    board = db.scalar(select(Board).where(Board.invite_code == body.invite_code))
    if board is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Invalid invite code")
    if not any(m.user_id == user.id for m in board.members):
        db.add(BoardMember(board_id=board.id, user_id=user.id))
        db.commit()
        db.refresh(board)
    return _summary(board)


def _pick_out(pick: BoardPick, user_id: int, n_members: int) -> dict:
    scores = [v.score for v in pick.votes]
    r = score_pick(pick.id, scores, n_members)
    mine = next((v.score for v in pick.votes if v.user_id == user_id), None)
    return {
        "id": pick.id, "item_type": pick.item_type, "asin": pick.asin, "title": pick.title,
        "image": pick.image, "rgb": [int(v) for v in pick.rgb.split(",")] if pick.rgb else None,
        "price_inr": pick.price_inr, "note": pick.note, "added_by": pick.added_by,
        "my_vote": mine, "votes_in": r.votes_in, "mean": r.mean, "consensus": r.consensus,
        "vetoes": r.vetoes,
    }


@router.get("/{board_id}")
def get_board(board_id: int, user: CurrentUser, db: DbSession) -> dict:
    board = _member_board(db, board_id, user)
    n = len(board.members)
    return {
        **_summary(board),
        "invite_code": board.invite_code if board.owner_id == user.id else None,
        "members": [{"user_id": m.user_id, "name": m.user.name, "role": m.role} for m in board.members],
        "picks": [_pick_out(p, user.id, n) for p in board.picks],
    }


@router.post("/{board_id}/picks", status_code=status.HTTP_201_CREATED)
def add_pick(board_id: int, body: PickIn, user: CurrentUser, db: DbSession) -> dict:
    board = _member_board(db, board_id, user)
    if body.item_type not in PRICE_BANDS:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "Unknown item type")
    if any(p.asin == body.asin for p in board.picks):
        raise HTTPException(status.HTTP_409_CONFLICT, "Already on the board")
    if body.rgb and any(not 0 <= c <= 255 for c in body.rgb):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "rgb values must be 0-255")
    pick = BoardPick(
        board_id=board.id, added_by=user.id, item_type=body.item_type, asin=body.asin,
        title=body.title, image=body.image, rgb=",".join(map(str, body.rgb)) if body.rgb else None,
        price_inr=body.price_inr, note=body.note,
    )
    db.add(pick)
    db.commit()
    return _pick_out(pick, user.id, len(board.members))


@router.delete("/{board_id}/picks/{pick_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove_pick(board_id: int, pick_id: int, user: CurrentUser, db: DbSession) -> None:
    board = _member_board(db, board_id, user)
    pick = db.get(BoardPick, pick_id)
    if pick is None or pick.board_id != board.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Pick not found")
    if pick.added_by != user.id and board.owner_id != user.id:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Only the person who added it or the owner can remove it")
    db.delete(pick)
    db.commit()


@router.put("/{board_id}/picks/{pick_id}/vote")
def vote(board_id: int, pick_id: int, body: VoteIn, user: CurrentUser, db: DbSession) -> dict:
    board = _member_board(db, board_id, user)
    pick = db.get(BoardPick, pick_id)
    if pick is None or pick.board_id != board.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Pick not found")
    existing = next((v for v in pick.votes if v.user_id == user.id), None)
    if existing:
        existing.score = body.score
    else:
        db.add(BoardVote(pick_id=pick.id, user_id=user.id, score=body.score))
    db.commit()
    db.refresh(pick)
    return _pick_out(pick, user.id, len(board.members))


@router.get("/{board_id}/decision")
def decision(board_id: int, user: CurrentUser, db: DbSession) -> dict:
    board = _member_board(db, board_id, user)
    n = len(board.members)
    by_type: dict[str, list] = {}
    pick_by_id = {p.id: p for p in board.picks}
    for p in board.picks:
        by_type.setdefault(p.item_type, []).append(score_pick(p.id, [v.score for v in p.votes], n))
    chosen = decide(by_type)
    total = 0
    items = []
    for item_type, d in chosen.items():
        p = pick_by_id[d["winner"]]
        price = p.price_inr if p.price_inr is not None else PRICE_BANDS[item_type][1]
        total += price
        items.append({
            "item_type": item_type, "pick": _pick_out(p, user.id, n), "contested": d["contested"],
            "estimated_price_inr": price, "price_is_estimate": p.price_inr is None,
        })
    return {
        "items": sorted(items, key=lambda i: i["item_type"]),
        "estimated_total_inr": total, "budget_inr": board.budget_inr,
        "within_budget": total <= board.budget_inr,
        "over_by_inr": max(0, total - board.budget_inr),
    }
