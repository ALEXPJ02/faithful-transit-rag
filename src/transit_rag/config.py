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


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name, "1" if default else "0").lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"{name} must be a boolean (true/false), got {raw!r}")


def _env_choice(name: str, default: str, allowed: tuple[str, ...]) -> str:
    value = _env(name, default).lower()
    if value not in allowed:
        raise ConfigError(f"{name} must be one of {', '.join(allowed)}, got {value!r}")
    return value


def _env_csv(name: str, default: str) -> tuple[str, ...]:
    raw = _env(name, default)
    return tuple(part.strip() for part in raw.split(",") if part.strip())


@dataclass(frozen=True)
class TfnswConfig:
    """Transport for NSW Open Data Hub access.

    Both URLs are confirmed against the products' own OpenAPI consoles:

    * trip updates — base ``api.transport.nsw.gov.au/v2/gtfs/realtime``,
      path ``/sydneytrains``
    * service alerts — base ``api.transport.nsw.gov.au/v2/gtfs/alerts``,
      path ``/sydneytrains``
    * static schedule — ``api.transport.nsw.gov.au/v1/gtfs/schedule/
      sydneytrains``, the "For Realtime" bundle whose ids match the feeds

    The version is a path *prefix* in both cases, not a suffix; the two
    products differ only in the segment after it. The alerts product also
    serves ``/all`` across every operator, which the agent's live tool layer
    may want later — the per-mode path is the right scope for the
    prediction layer's active-alert feature.
    """

    api_key: str
    trip_update_url: str
    alerts_url: str
    schedule_url: str

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
                "https://api.transport.nsw.gov.au/v2/gtfs/realtime/sydneytrains",
            ),
            alerts_url=_env(
                "TFNSW_ALERTS_URL",
                "https://api.transport.nsw.gov.au/v2/gtfs/alerts/sydneytrains",
            ),
            # The static bundle whose identifiers match the realtime feeds.
            # Deliberately NOT "Timetables Complete GTFS", which states plainly
            # that its identifiers do not match the realtime APIs — using it
            # would resolve nothing and collect zero rows.
            schedule_url=_env(
                "TFNSW_SCHEDULE_URL",
                "https://api.transport.nsw.gov.au/v1/gtfs/schedule/sydneytrains",
            ),
        )


@dataclass(frozen=True)
class CollectionConfig:
    """Settings for the unattended delay-observation collector.

    ``sink`` selects where observations go: ``sqlite`` for a long-running
    process that keeps its own state, ``csv`` for a stateless scheduled run
    that can only append files to a repository. See
    ``transit_rag.prediction.collection.store``.
    """

    db_path: Path = field(
        default_factory=lambda: _env_path("COLLECTION_DB_PATH", "data/delay_observations.db")
    )
    snapshot_dir: Path = field(
        default_factory=lambda: _env_path("COLLECTION_SNAPSHOT_DIR", "data/observations")
    )
    routes_lookup_path: Path = field(
        default_factory=lambda: _env_path("COLLECTION_ROUTES_LOOKUP", "data/routes_lookup.csv")
    )
    sink: str = field(
        default_factory=lambda: _env_choice("COLLECTION_SINK", "sqlite", ("sqlite", "csv"))
    )
    tracked_routes: tuple[str, ...] = field(
        default_factory=lambda: _env_csv("POLLER_ROUTES", "T1,T4")
    )
    interval_seconds: int = field(default_factory=lambda: _env_int("POLLER_INTERVAL_SECONDS", 120))
    request_timeout_seconds: int = field(
        default_factory=lambda: _env_int("POLLER_REQUEST_TIMEOUT", 30)
    )
    max_upcoming_stops: int = field(
        default_factory=lambda: _env_int("POLLER_MAX_UPCOMING_STOPS", 3)
    )
    # Burst mode. A scheduled runner is admitted rarely and unpredictably —
    # GitHub honoured roughly one run every two hours against a five-minute
    # request — so each admission has to be worth more than a single poll.
    # Several polls spaced a minute or two apart let a trip be seen more than
    # once as it approaches, which is the whole basis of treating the last
    # observation as an outcome. One snapshot every two hours cannot do that.
    burst_polls: int = field(default_factory=lambda: _env_int("POLLER_BURST_POLLS", 5))
    burst_spacing_seconds: int = field(default_factory=lambda: _env_int("POLLER_BURST_SPACING", 90))
    # Unattended runs must refuse to collect without a route lookup. Locally an
    # unfiltered run is a useful way to see what the feed contains; in a
    # scheduled job it is weeks of unattributed rows behind a green tick,
    # because the only signal is a log line nobody reads.
    require_routes_lookup: bool = field(
        default_factory=lambda: _env_bool("COLLECTION_REQUIRE_ROUTES_LOOKUP", False)
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
