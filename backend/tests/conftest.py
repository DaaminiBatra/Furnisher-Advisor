import os

# Must be set before the app is imported (settings are read at import time).
os.environ["SECRET_KEY"] = "test-secret-key-not-for-production"
os.environ["DATABASE_URL"] = "sqlite://"

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app import mailer
from app.db import Base, get_db
from app.main import app
from app.models import Item, SubCategory


@pytest.fixture()
def db_session():
    # Default: in-memory SQLite. Set TEST_DATABASE_URL=postgresql://... to run the
    # same suite against real Postgres (CI does this).
    pg_url = os.environ.get("TEST_DATABASE_URL")
    if pg_url:
        if pg_url.startswith("postgresql://"):
            pg_url = "postgresql+psycopg://" + pg_url[len("postgresql://"):]
        engine = create_engine(pg_url)
        Base.metadata.drop_all(engine)
    else:
        engine = create_engine(
            "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
        )
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    session = Session()
    try:
        yield session
    finally:
        session.close()
        if pg_url:
            Base.metadata.drop_all(engine)
        engine.dispose()


@pytest.fixture()
def sent_emails(monkeypatch):
    sent: list[tuple[str, str, str]] = []
    monkeypatch.setattr(mailer, "send_reset_email", lambda to, name, link: sent.append((to, name, link)))
    return sent


@pytest.fixture()
def client(db_session, sent_emails):
    def override():
        yield db_session

    app.dependency_overrides[get_db] = override
    # No lifespan context: tables are created by the db_session fixture.
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture()
def auth_headers(client):
    r = client.post(
        "/api/auth/register",
        json={"name": "Test User", "email": "test@example.com", "password": "password123"},
    )
    assert r.status_code == 201
    return {"Authorization": f"Bearer {r.json()['token']}"}


@pytest.fixture()
def catalog(db_session):
    item = Item(name="sofa")
    other = Item(name="lamp")
    db_session.add_all([item, other])
    db_session.flush()
    sub = SubCategory(item_id=item.id, name="Fabric sofa", link="https://example.com/sofa", image_path="sofa.jpg")
    other_sub = SubCategory(item_id=other.id, name="Floor lamp", link="https://example.com/lamp")
    db_session.add_all([sub, other_sub])
    db_session.commit()
    return {"item": item, "sub": sub, "other": other, "other_sub": other_sub}
