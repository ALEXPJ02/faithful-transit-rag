"""Central configuration, read once from the environment.

Deliberately stdlib-only. The delay collector runs unattended in
environments (GitHub Actions, a small always-on box) where installing the
retrieval and model stack would be dead weight, so nothing here may import
a third-party package at module scope.

Values come from the process environment; ``.env`` is loaded first when
python-dotenv happens to be installed. See ``.env.example`` for the keys.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Repository root: src/transit_rag/config.py -> transit_rag -> src -> root
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv_if_available() -> None:
    """Load .env when python-dotenv is present. A no-op in CI, by design."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(PROJECT_ROOT / ".env")


_load_dotenv_if_available()


class ConfigError(RuntimeError):
    """Raised when a required setting is missing or malformed."""


def _env(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value or default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


def _env_path(name: str, default: str) -> Path:
    raw = _env(name, default)
    path = Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _env_csv(name: str, default: str) -> tuple[str, ...]:
    raw = _env(name, default)
    return tuple(part.strip() for part in raw.split(",") if part.strip())


@dataclass(frozen=True)
class TfnswConfig:
    """Transport for NSW Open Data Hub access.

    ``trip_update_url`` defaults to the documented Sydney Trains realtime
    path. The exact resource path sits behind login on the Open Data Hub,
    so confirm it against your own Applications page before a long
    collection run and override it via ``TFNSW_TRIP_UPDATE_URL`` if it
    differs. See docs/05-setup-checklist.md.
    """

    api_key: str
    trip_update_url: str
    alerts_url: str

    @classmethod
    def from_env(cls) -> TfnswConfig:
        api_key = os.environ.get("TFNSW_API_KEY", "").strip()
        if not api_key:
            raise ConfigError(
                "TFNSW_API_KEY is not set. Copy .env.example to .env and add the key "
                "from your Open Data Hub account (docs/05-setup-checklist.md)."
            )
        return cls(
            api_key=api_key,
            trip_update_url=_env(
                "TFNSW_TRIP_UPDATE_URL",
                "https://api.transport.nsw.gov.au/v1/gtfs/realtime/sydneytrains",
            ),
            alerts_url=_env(
                "TFNSW_ALERTS_URL",
                "https://api.transport.nsw.gov.au/v1/gtfs/alerts/sydneytrains",
            ),
        )


@dataclass(frozen=True)
class CollectionConfig:
    """Settings for the unattended delay-observation collector."""

    db_path: Path = field(
        default_factory=lambda: _env_path("COLLECTION_DB_PATH", "data/delay_observations.db")
    )
    routes_lookup_path: Path = field(
        default_factory=lambda: _env_path("COLLECTION_ROUTES_LOOKUP", "data/routes_lookup.csv")
    )
    tracked_routes: tuple[str, ...] = field(
        default_factory=lambda: _env_csv("POLLER_ROUTES", "T1,T4")
    )
    interval_seconds: int = field(default_factory=lambda: _env_int("POLLER_INTERVAL_SECONDS", 120))
    request_timeout_seconds: int = field(
        default_factory=lambda: _env_int("POLLER_REQUEST_TIMEOUT", 30)
    )


@dataclass(frozen=True)
class ModelConfig:
    """Anthropic + Voyage model selection."""

    anthropic_api_key: str
    voyage_api_key: str
    generation_model: str
    judge_model: str
    embedding_model: str

    @classmethod
    def from_env(cls) -> ModelConfig:
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        voyage_key = os.environ.get("VOYAGE_API_KEY", "").strip()
        missing = [
            name
            for name, value in (
                ("ANTHROPIC_API_KEY", anthropic_key),
                ("VOYAGE_API_KEY", voyage_key),
            )
            if not value
        ]
        if missing:
            raise ConfigError(f"Missing required environment variables: {', '.join(missing)}")
        return cls(
            anthropic_api_key=anthropic_key,
            voyage_api_key=voyage_key,
            generation_model=_env("ANTHROPIC_GENERATION_MODEL", "claude-sonnet-5"),
            judge_model=_env("ANTHROPIC_JUDGE_MODEL", "claude-haiku-4-5"),
            embedding_model=_env("VOYAGE_EMBEDDING_MODEL", "voyage-4-lite"),
        )


def chroma_persist_dir() -> Path:
    """Where the vector index lives. Baked into the image at build time for
    deployment, since the RAG corpus is static (docs/02-tech-stack.md)."""
    return _env_path("CHROMA_PERSIST_DIR", ".chroma")
