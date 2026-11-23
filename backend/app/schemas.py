import re
from datetime import datetime

from pydantic import BaseModel, EmailStr, Field, field_validator


def _check_password(v: str) -> str:
    if len(v) < 8:
        raise ValueError("Password must be at least 8 characters")
    if len(v.encode()) > 72:  # bcrypt limit
        raise ValueError("Password is too long (max 72 bytes)")
    return v


class RegisterIn(BaseModel):
    name: str = Field(min_length=1, max_length=150)
    email: EmailStr
    password: str

    _pw = field_validator("password")(_check_password)

    @field_validator("name")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Name is required")
        return v


class LoginIn(BaseModel):
    email: EmailStr
    password: str


class TokenOut(BaseModel):
    token: str
    username: str


class MeOut(BaseModel):
    id: int
    name: str
    email: EmailStr


class ForgotIn(BaseModel):
    email: EmailStr


class ResetIn(BaseModel):
    token: str
    new_password: str

    _pw = field_validator("new_password")(_check_password)


class MessageOut(BaseModel):
    message: str


class ContactIn(BaseModel):
    first_name: str = Field(min_length=1, max_length=50)
    last_name: str | None = Field(default=None, max_length=50)
    email: EmailStr
    phone: str
    message: str = Field(min_length=1, max_length=5000)

    @field_validator("phone")
    @classmethod
    def _phone(cls, v: str) -> str:
        digits = re.sub(r"[\s\-+()]", "", v)
        if not digits.isdigit() or not 7 <= len(digits) <= 15:
            raise ValueError("Enter a valid phone number")
        return digits


# --- Favourites ---------------------------------------------------------------


class SavedFurnitureIn(BaseModel):
    item_id: int
    subcategory_id: int
    link: str = Field(min_length=1)
    color: str | None = None
    estimated_price: int | None = Field(default=None, ge=0)


class SaveFurnitureRequest(BaseModel):
    room_type: str = Field(min_length=1, max_length=50)
    timestamp: datetime
    budget_set: int | None = Field(default=None, ge=0)
    original_image_path: str | None = None
    processed_image_path: str | None = None
    recommendations: list[SavedFurnitureIn] = Field(min_length=1, max_length=100)


class SavedColorIn(BaseModel):
    color_name: str = Field(min_length=1, max_length=100)
    rgb_value: str = Field(pattern=r"^\d{1,3},\d{1,3},\d{1,3}$")


class SaveColorsRequest(BaseModel):
    room_type: str = Field(min_length=1, max_length=50)
    timestamp: datetime
    colors: list[SavedColorIn] = Field(min_length=1, max_length=50)


class SavedFurnitureOut(BaseModel):
    id: int
    category: str | None
    image: str | None
    color: str | None
    link: str | None
    estimated_price: int | None
    set_budget: int | None


class SavedItemGroup(BaseModel):
    item_name: str
    recommendations: list[SavedFurnitureOut]


class SavedColorOut(BaseModel):
    id: int
    color_name: str | None
    rgb_value: str | None


class FavouriteSession(BaseModel):
    timestamp: datetime
    room_type: str
    original_image: str | None
    processed_image: str | None
    saved_items: list[SavedItemGroup]
    saved_colors: list[SavedColorOut]
