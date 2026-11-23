import hashlib
from datetime import UTC, datetime, timedelta

import bcrypt
import jwt
from itsdangerous import BadSignature, URLSafeTimedSerializer

from .config import get_settings

_RESET_SALT = "password-reset"


def hash_password(password: str) -> str:
    # bcrypt only reads the first 72 bytes; reject longer input upstream (schemas).
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode(), password_hash.encode())
    except ValueError:
        return False


def create_access_token(user_id: int) -> str:
    s = get_settings()
    payload = {
        "sub": str(user_id),
        "exp": datetime.now(UTC) + timedelta(hours=s.access_token_hours),
    }
    return jwt.encode(payload, s.secret_key, algorithm="HS256")


def decode_access_token(token: str) -> int | None:
    try:
        payload = jwt.decode(token, get_settings().secret_key, algorithms=["HS256"])
        return int(payload["sub"])
    except (jwt.PyJWTError, KeyError, ValueError):
        return None


def _fingerprint(password_hash: str) -> str:
    return hashlib.sha256(password_hash.encode()).hexdigest()[:16]


def create_reset_token(email: str, password_hash: str) -> str:
    """Token is bound to the current password hash, so it stops working the moment
    the password changes: single-use without storing anything extra."""
    s = URLSafeTimedSerializer(get_settings().secret_key)
    return s.dumps({"email": email, "fp": _fingerprint(password_hash)}, salt=_RESET_SALT)


def read_reset_token(token: str) -> dict | None:
    s = get_settings()
    try:
        data = URLSafeTimedSerializer(s.secret_key).loads(
            token, salt=_RESET_SALT, max_age=s.reset_token_minutes * 60
        )
    except BadSignature:
        return None
    if not isinstance(data, dict) or "email" not in data or "fp" not in data:
        return None
    return data


def reset_token_matches(data: dict, password_hash: str) -> bool:
    return data.get("fp") == _fingerprint(password_hash)
