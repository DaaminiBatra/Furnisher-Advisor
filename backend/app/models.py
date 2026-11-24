from datetime import UTC, datetime

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


def utcnow() -> datetime:
    return datetime.now(UTC)


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(150))
    email: Mapped[str] = mapped_column(String(254), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    saved: Mapped[list["SavedRecommendation"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


class ContactSubmission(Base):
    __tablename__ = "contact_submissions"

    id: Mapped[int] = mapped_column(primary_key=True)
    first_name: Mapped[str] = mapped_column(String(50))
    last_name: Mapped[str | None] = mapped_column(String(50))
    email: Mapped[str] = mapped_column(String(254))
    phone: Mapped[str] = mapped_column(String(20))
    message: Mapped[str] = mapped_column(Text)
    submitted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Upload(Base):
    """Log of room analyses (kept from v1; used later for evaluation data)."""

    __tablename__ = "uploads"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    room_type: Mapped[str] = mapped_column(String(50))
    success: Mapped[bool] = mapped_column(Boolean)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


# --- Catalog (Phase 3 replaces this with the live product catalog) ----------


class Item(Base):
    __tablename__ = "items"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100), index=True)

    sub_categories: Mapped[list["SubCategory"]] = relationship(
        back_populates="item", cascade="all, delete-orphan"
    )


class SubCategory(Base):
    __tablename__ = "sub_categories"

    id: Mapped[int] = mapped_column(primary_key=True)
    item_id: Mapped[int] = mapped_column(ForeignKey("items.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(100))
    link: Mapped[str] = mapped_column(Text)
    image_path: Mapped[str | None] = mapped_column(String(200))
    colors: Mapped[list | None] = mapped_column(JSON)

    item: Mapped[Item] = relationship(back_populates="sub_categories")


# --- Saved by users ----------------------------------------------------------


class SavedRecommendation(Base):
    """A saved furniture pick (kind='furniture') or saved colour (kind='color')."""

    __tablename__ = "saved_recommendations"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    kind: Mapped[str] = mapped_column(String(20), default="furniture")
    room_type: Mapped[str] = mapped_column(String(50))
    # Groups everything saved from one analysis session together.
    session_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)

    item_id: Mapped[int | None] = mapped_column(ForeignKey("items.id", ondelete="SET NULL"))
    subcategory_id: Mapped[int | None] = mapped_column(
        ForeignKey("sub_categories.id", ondelete="SET NULL")
    )
    link: Mapped[str | None] = mapped_column(Text)
    color: Mapped[str | None] = mapped_column(String(100))
    estimated_price: Mapped[int | None] = mapped_column(Integer)
    set_budget: Mapped[int | None] = mapped_column(Integer)

    color_name: Mapped[str | None] = mapped_column(String(100))
    rgb_value: Mapped[str | None] = mapped_column(String(50))  # "R,G,B"

    original_image_path: Mapped[str | None] = mapped_column(String(255))
    processed_image_path: Mapped[str | None] = mapped_column(String(255))

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    user: Mapped[User] = relationship(back_populates="saved")
    item: Mapped[Item | None] = relationship()
    sub_category: Mapped[SubCategory | None] = relationship()


# --- Group boards (shared room decisions) -------------------------------------


class Board(Base):
    """A room several people decide on together."""

    __tablename__ = "boards"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    room_type: Mapped[str] = mapped_column(String(50))
    wall_rgb: Mapped[str] = mapped_column(String(20))  # "R,G,B"
    budget_inr: Mapped[int] = mapped_column(Integer)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    invite_code: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    members: Mapped[list["BoardMember"]] = relationship(
        back_populates="board", cascade="all, delete-orphan"
    )
    picks: Mapped[list["BoardPick"]] = relationship(
        back_populates="board", cascade="all, delete-orphan"
    )


class BoardMember(Base):
    __tablename__ = "board_members"
    __table_args__ = (UniqueConstraint("board_id", "user_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    board_id: Mapped[int] = mapped_column(ForeignKey("boards.id", ondelete="CASCADE"), index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    role: Mapped[str] = mapped_column(String(10), default="member")  # owner | member
    joined_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    board: Mapped[Board] = relationship(back_populates="members")
    user: Mapped[User] = relationship()


class BoardPick(Base):
    """A candidate product proposed for the room. The product is snapshotted so the board
    survives catalogue updates."""

    __tablename__ = "board_picks"
    __table_args__ = (UniqueConstraint("board_id", "asin"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    board_id: Mapped[int] = mapped_column(ForeignKey("boards.id", ondelete="CASCADE"), index=True)
    added_by: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    item_type: Mapped[str] = mapped_column(String(40), index=True)
    asin: Mapped[str] = mapped_column(String(40))
    title: Mapped[str] = mapped_column(String(200))
    image: Mapped[str | None] = mapped_column(Text)
    rgb: Mapped[str | None] = mapped_column(String(20))
    price_inr: Mapped[int | None] = mapped_column(Integer)  # estimate, may be null
    note: Mapped[str | None] = mapped_column(String(300))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    board: Mapped[Board] = relationship(back_populates="picks")
    votes: Mapped[list["BoardVote"]] = relationship(
        back_populates="pick", cascade="all, delete-orphan"
    )


class BoardVote(Base):
    __tablename__ = "board_votes"
    __table_args__ = (UniqueConstraint("pick_id", "user_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    pick_id: Mapped[int] = mapped_column(ForeignKey("board_picks.id", ondelete="CASCADE"), index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    score: Mapped[int] = mapped_column(Integer)  # 1 love, 0 fine, -1 no

    pick: Mapped[BoardPick] = relationship(back_populates="votes")
