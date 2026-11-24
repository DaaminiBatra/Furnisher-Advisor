import time
from collections import defaultdict, deque

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from .. import assistant
from ..config import get_settings
from ..deps import CurrentUser, DbSession
from ..models import BoardMember
from ..routers.boards import decision
from ..routers.recommend import get_catalog

router = APIRouter(prefix="/assistant", tags=["assistant"])

_LIMIT, _WINDOW = 20, 3600  # requests per user per hour (the free Gemini quota is shared)
_hits: dict[int, deque] = defaultdict(deque)


class Message(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    text: str = Field(min_length=1, max_length=2000)


class ChatIn(BaseModel):
    messages: list[Message] = Field(min_length=1, max_length=20)
    board_id: int | None = None


def get_generate():
    s = get_settings()
    if not s.gemini_api_key:
        return None
    return lambda contents: assistant.gemini_generate(s.gemini_api_key, s.gemini_model, contents)


def _rate_limit(user_id: int) -> None:
    now = time.monotonic()
    q = _hits[user_id]
    while q and now - q[0] > _WINDOW:
        q.popleft()
    if len(q) >= _LIMIT:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "Too many assistant requests; try again later")
    q.append(now)


@router.post("/chat")
def chat(body: ChatIn, user: CurrentUser, db: DbSession, catalog=Depends(get_catalog),
         generate=Depends(get_generate)) -> dict:
    if generate is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "The assistant is not configured yet")
    if body.messages[-1].role != "user":
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "Last message must be from the user")
    board_fn = None
    if body.board_id is not None:
        is_member = db.query(BoardMember.id).filter_by(board_id=body.board_id, user_id=user.id).first()
        if not is_member:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Board not found")
        board_fn = lambda: decision(body.board_id, user, db)  # noqa: E731
    _rate_limit(user.id)
    try:
        return assistant.chat([m.model_dump() for m in body.messages], generate, catalog, board_fn)
    except RuntimeError:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "The assistant is unavailable right now") from None
