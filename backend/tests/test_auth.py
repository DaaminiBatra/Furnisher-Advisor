from urllib.parse import parse_qs, urlparse

import jwt

from app.config import get_settings

CREDS = {"name": "Asha", "email": "Asha@Example.com", "password": "password123"}


def test_register_and_login(client):
    r = client.post("/api/auth/register", json=CREDS)
    assert r.status_code == 201
    assert r.json()["username"] == "Asha"

    # Email is case-insensitive
    r = client.post("/api/auth/login", json={"email": "asha@example.com", "password": "password123"})
    assert r.status_code == 200
    assert r.json()["token"]


def test_duplicate_email_rejected(client):
    client.post("/api/auth/register", json=CREDS)
    r = client.post("/api/auth/register", json={**CREDS, "email": "asha@example.com"})
    assert r.status_code == 409


def test_wrong_password_and_unknown_user_look_the_same(client):
    client.post("/api/auth/register", json=CREDS)
    wrong = client.post("/api/auth/login", json={"email": "asha@example.com", "password": "nope-nope-1"})
    unknown = client.post("/api/auth/login", json={"email": "who@example.com", "password": "password123"})
    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json() == unknown.json()


def test_password_validation(client):
    short = client.post("/api/auth/register", json={**CREDS, "password": "short"})
    assert short.status_code == 422
    long = client.post("/api/auth/register", json={**CREDS, "password": "x" * 80})
    assert long.status_code == 422
    bad_email = client.post("/api/auth/register", json={**CREDS, "email": "not-an-email"})
    assert bad_email.status_code == 422


def test_me_requires_valid_token(client, auth_headers):
    assert client.get("/api/auth/me").status_code == 401
    assert client.get("/api/auth/me", headers={"Authorization": "Bearer garbage"}).status_code == 401
    r = client.get("/api/auth/me", headers=auth_headers)
    assert r.status_code == 200
    assert r.json()["email"] == "test@example.com"


def test_expired_token_rejected(client, auth_headers):
    expired = jwt.encode(
        {"sub": "1", "exp": 1}, get_settings().secret_key, algorithm="HS256"
    )
    r = client.get("/api/auth/me", headers={"Authorization": f"Bearer {expired}"})
    assert r.status_code == 401


def test_token_for_deleted_user_rejected(client, db_session):
    from app.security import create_access_token

    token = create_access_token(9999)
    r = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 401


def _token_from(sent_emails):
    assert len(sent_emails) == 1
    link = sent_emails[0][2]
    return parse_qs(urlparse(link).query)["token"][0]


def test_forgot_password_does_not_reveal_accounts(client, sent_emails):
    client.post("/api/auth/register", json=CREDS)
    known = client.post("/api/auth/forgot-password", json={"email": "asha@example.com"})
    unknown = client.post("/api/auth/forgot-password", json={"email": "ghost@example.com"})
    assert known.status_code == unknown.status_code == 200
    assert known.json() == unknown.json()
    assert len(sent_emails) == 1  # only the real account got an email


def test_reset_flow_and_single_use(client, sent_emails):
    client.post("/api/auth/register", json=CREDS)
    client.post("/api/auth/forgot-password", json={"email": "asha@example.com"})
    token = _token_from(sent_emails)

    r = client.post("/api/auth/reset-password", json={"token": token, "new_password": "brand-new-pass"})
    assert r.status_code == 200

    assert client.post(
        "/api/auth/login", json={"email": "asha@example.com", "password": "password123"}
    ).status_code == 401
    assert client.post(
        "/api/auth/login", json={"email": "asha@example.com", "password": "brand-new-pass"}
    ).status_code == 200

    # The same link must not work a second time.
    again = client.post("/api/auth/reset-password", json={"token": token, "new_password": "another-pass-1"})
    assert again.status_code == 400


def test_reset_rejects_garbage_and_weak_passwords(client, sent_emails):
    assert client.post(
        "/api/auth/reset-password", json={"token": "garbage", "new_password": "long-enough-1"}
    ).status_code == 400
    client.post("/api/auth/register", json=CREDS)
    client.post("/api/auth/forgot-password", json={"email": "asha@example.com"})
    token = _token_from(sent_emails)
    assert client.post(
        "/api/auth/reset-password", json={"token": token, "new_password": "short"}
    ).status_code == 422


def test_reset_token_cannot_be_used_as_login_token(client, sent_emails):
    client.post("/api/auth/register", json=CREDS)
    client.post("/api/auth/forgot-password", json={"email": "asha@example.com"})
    token = _token_from(sent_emails)
    assert client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"}).status_code == 401
