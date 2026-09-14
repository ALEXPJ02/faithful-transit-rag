"""Tests for environment-driven configuration."""

from __future__ import annotations

import pytest

from transit_rag.config import PROJECT_ROOT, CollectionConfig, ConfigError, TfnswConfig


def test_tfnsw_config_requires_an_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TFNSW_API_KEY", raising=False)

    with pytest.raises(ConfigError, match="TFNSW_API_KEY"):
        TfnswConfig.from_env()


def test_tfnsw_config_treats_whitespace_as_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TFNSW_API_KEY", "   ")

    with pytest.raises(ConfigError):
        TfnswConfig.from_env()


def test_collection_config_defaults_to_tracking_t1_and_t4(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("POLLER_ROUTES", raising=False)
    monkeypatch.delenv("POLLER_INTERVAL_SECONDS", raising=False)

    config = CollectionConfig()

    assert config.tracked_routes == ("T1", "T4")
    assert config.interval_seconds == 120


def test_collection_config_reads_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLLER_ROUTES", " T1 , T2 ,, T9 ")
    monkeypatch.setenv("POLLER_INTERVAL_SECONDS", "300")

    config = CollectionConfig()

    assert config.tracked_routes == ("T1", "T2", "T9")
    assert config.interval_seconds == 300


def test_relative_paths_resolve_against_the_project_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """The collector is launched from cron and systemd units whose working
    directory is not the repo; a relative default must not follow it."""
    monkeypatch.setenv("COLLECTION_DB_PATH", "data/obs.db")

    config = CollectionConfig()

    assert config.db_path == PROJECT_ROOT / "data" / "obs.db"


def test_bad_integer_is_reported_not_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLLER_INTERVAL_SECONDS", "two minutes")

    with pytest.raises(ConfigError, match="POLLER_INTERVAL_SECONDS"):
        CollectionConfig()


def test_alerts_are_collected_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("COLLECT_ALERTS", raising=False)
    monkeypatch.delenv("ALERTS_POLL_EVERY_N_POLLS", raising=False)
    config = CollectionConfig()
    assert config.collect_alerts is True
    # 15 x 120s = 30 minutes, finer than the hours-long trackwork windows
    # these alerts describe.
    assert config.alerts_every_n_polls == 15


def test_alert_collection_can_be_turned_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COLLECT_ALERTS", "false")
    monkeypatch.setenv("ALERTS_POLL_EVERY_N_POLLS", "5")
    config = CollectionConfig()
    assert config.collect_alerts is False
    assert config.alerts_every_n_polls == 5


def test_a_bad_alert_interval_is_reported_not_swallowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ALERTS_POLL_EVERY_N_POLLS", "half-hourly")
    with pytest.raises(ConfigError):
        CollectionConfig()
