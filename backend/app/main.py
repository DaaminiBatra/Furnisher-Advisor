from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from . import models  # noqa: F401  (registers tables on Base.metadata)
from .config import get_settings
from .db import Base, get_engine
from .routers import auth, boards, contact, favourites, recommend


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Phase 0 uses create_all; Alembic migrations arrive once the schema settles.
    Base.metadata.create_all(get_engine())
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="Furnisher Advisor API", version="0.1.0", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization"],
    )

    @app.get("/health", tags=["meta"])
    def health() -> dict:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "ok"}

    app.include_router(auth.router, prefix="/api")
    app.include_router(contact.router, prefix="/api")
    app.include_router(favourites.router, prefix="/api")
    app.include_router(recommend.router, prefix="/api")
    app.include_router(boards.router, prefix="/api")
    return app


app = create_app()
