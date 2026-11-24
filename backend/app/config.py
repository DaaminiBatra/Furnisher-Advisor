from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Required: there is deliberately no default, so the app refuses to start
    # without a real secret (v1 silently fell back to a hard-coded string).
    secret_key: str = Field(min_length=16)

    database_url: str = "sqlite:///./dev.db"
    cors_origins: str = "http://localhost:3000,http://localhost:5173"
    frontend_url: str = "http://localhost:3000"

    # JSON-lines product catalogue (local path or URL); set empty to run without products.
    catalog_source: str = (
        "https://huggingface.co/datasets/daami/furnisher-advisor-catalog/resolve/main/catalog.jsonl"
    )

    # Chat assistant (Gemini via Google AI Studio). Empty key = assistant disabled.
    gemini_api_key: str = ""
    gemini_model: str = "gemini-flash-latest"

    access_token_hours: int = 24
    reset_token_minutes: int = 15

    # Email (optional). If smtp_host is empty, reset links are only logged.
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    mail_from: str = ""

    @property
    def sqlalchemy_url(self) -> str:
        """Render/Heroku-style URLs use postgres:// which SQLAlchemy rejects."""
        url = self.database_url
        if url.startswith("postgres://"):
            url = "postgresql://" + url[len("postgres://"):]
        if url.startswith("postgresql://"):
            url = "postgresql+psycopg://" + url[len("postgresql://"):]
        return url

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
