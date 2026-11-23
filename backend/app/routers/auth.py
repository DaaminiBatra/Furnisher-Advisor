from fastapi import APIRouter, BackgroundTasks, HTTPException, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from .. import mailer
from ..config import get_settings
from ..deps import CurrentUser, DbSession
from ..models import User
from ..schemas import (
    ForgotIn,
    LoginIn,
    MeOut,
    MessageOut,
    RegisterIn,
    ResetIn,
    TokenOut,
)
from ..security import (
    create_access_token,
    create_reset_token,
    hash_password,
    read_reset_token,
    reset_token_matches,
    verify_password,
)

router = APIRouter(prefix="/auth", tags=["auth"])

# Hash compared against when the email is unknown, so login takes about the same
# time either way and doesn't reveal which emails are registered.
_DUMMY_HASH = hash_password("not-a-real-password")


def _find_user(db, email: str) -> User | None:
    return db.scalar(select(User).where(User.email == email.lower()))


@router.post("/register", response_model=TokenOut, status_code=status.HTTP_201_CREATED)
def register(body: RegisterIn, db: DbSession) -> TokenOut:
    email = body.email.lower()
    if _find_user(db, email):
        raise HTTPException(status.HTTP_409_CONFLICT, "Email already in use")
    user = User(name=body.name, email=email, password_hash=hash_password(body.password))
    db.add(user)
    try:
        db.commit()
    except IntegrityError:  # two simultaneous sign-ups with the same email
        db.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, "Email already in use") from None
    return TokenOut(token=create_access_token(user.id), username=user.name)


@router.post("/login", response_model=TokenOut)
def login(body: LoginIn, db: DbSession) -> TokenOut:
    user = _find_user(db, body.email)
    ok = verify_password(body.password, user.password_hash if user else _DUMMY_HASH)
    if not user or not ok:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid credentials")
    return TokenOut(token=create_access_token(user.id), username=user.name)


@router.get("/me", response_model=MeOut)
def me(user: CurrentUser) -> User:
    return user


@router.post("/forgot-password", response_model=MessageOut)
def forgot_password(body: ForgotIn, db: DbSession, tasks: BackgroundTasks) -> MessageOut:
    user = _find_user(db, body.email)
    if user:
        token = create_reset_token(user.email, user.password_hash)
        link = f"{get_settings().frontend_url.rstrip('/')}/reset-password?token={token}"
        tasks.add_task(mailer.send_reset_email, user.email, user.name, link)
    # Same answer whether or not the account exists.
    return MessageOut(message="If the email exists, a reset link has been sent.")


@router.post("/reset-password", response_model=MessageOut)
def reset_password(body: ResetIn, db: DbSession) -> MessageOut:
    data = read_reset_token(body.token)
    user = _find_user(db, data["email"]) if data else None
    if not data or not user or not reset_token_matches(data, user.password_hash):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid or expired link")
    user.password_hash = hash_password(body.new_password)
    db.commit()
    return MessageOut(message="Password updated successfully")
