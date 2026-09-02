"""Tests for the collection loop.

The failure path gets the most attention here on purpose. A collector that
crashes is obvious; a collector that keeps running while every poll fails
costs a week of irreplaceable training data before anyone notices.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from transit_rag.config import CollectionConfig, ConfigError
from transit_rag.prediction.collection.poller import build_sink, poll_once, print_status
from transit_rag.prediction.collection.store import CsvSnapshotStore, SqliteObservationStore
from transit_rag.realtime.client import FeedFetchError

LOOKUP = {"APS_1a": "T1"}
TRACKED = ("T1", "T4")


class FakeClient:
    """Stands in for GtfsRealtimeClient: returns a feed, or raises."""

    def __init__(self, feed: Any = None, error: Exception | None = None) -> None:
        self._feed = feed
        self._error = error
        self.calls = 0

    def fetch_trip_updates(self) -> Any:
        self.calls += 1
        if self._error is not None:
            raise self._error
        return self._feed


def test_successful_poll_writes_observations_and_logs_ok(tmp_path: Path, make_feed: Any) -> None:
    feed = make_feed([("trip-1", "APS_1a", [("stop-a", 1, 60, None), ("stop-b", 2, 90, None)])])
    client = FakeClient(feed=feed)

    with SqliteObservationStore(tmp_path / "obs.db") as store:
        ok = poll_once(client, store, LOOKUP, TRACKED)  # type: ignore[arg-type]

        assert ok is True
        assert store.observation_count() == 2
        assert store.recent_poll_status()[0][3] == "ok"


def test_failed_poll_is_recorded_rather_than_raised(tmp_path: Path) -> None:
    """A transient 502 must not kill a collector that has been running for
    weeks — but it must leave a trace in the poll log."""
    client = FakeClient(error=FeedFetchError("GET ... failed: 502 Bad Gateway"))

    with SqliteObservationStore(tmp_path / "obs.db") as store:
        ok = poll_once(client, store, LOOKUP, TRACKED)  # type: ignore[arg-type]

        assert ok is False
        assert store.observation_count() == 0
        status = store.recent_poll_status()[0][3]
        assert status.startswith("error:")
        assert "502" in status


def test_poll_with_no_tracked_trips_still_logs_the_poll(tmp_path: Path, make_feed: Any) -> None:
    """Zero rows is a real outcome at 3am. It has to be distinguishable from
    a poll that never happened, or the volume checkpoint reads wrong."""
    feed = make_feed([("trip-x", "OTHER_9z", [("stop-z", 1, 30, None)])])
    client = FakeClient(feed=feed)

    with SqliteObservationStore(tmp_path / "obs.db") as store:
        poll_once(client, store, LOOKUP, TRACKED)  # type: ignore[arg-type]

        entities_seen, rows_written, status = store.recent_poll_status()[0][1:]
        assert (entities_seen, rows_written, status) == (1, 0, "ok")


def test_poll_respects_the_upcoming_stop_limit(tmp_path: Path, make_feed: Any) -> None:
    stops = [(f"stop-{i}", i, i * 10, None) for i in range(12)]
    client = FakeClient(feed=make_feed([("trip-1", "APS_1a", stops)]))

    with SqliteObservationStore(tmp_path / "obs.db") as store:
        poll_once(client, store, LOOKUP, TRACKED, max_upcoming_stops=2)  # type: ignore[arg-type]

        assert store.observation_count() == 2


def test_poll_writes_a_snapshot_when_given_the_csv_sink(tmp_path: Path, make_feed: Any) -> None:
    feed = make_feed([("trip-1", "APS_1a", [("stop-a", 1, 60, None)])])
    client = FakeClient(feed=feed)

    with CsvSnapshotStore(tmp_path / "snapshots") as store:
        poll_once(client, store, LOOKUP, TRACKED)  # type: ignore[arg-type]

        assert store.observation_count() == 1
        assert len(list((tmp_path / "snapshots").glob("*/*.csv"))) == 2  # snapshot + _polls


def test_status_flags_a_run_where_every_poll_failed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    client = FakeClient(error=FeedFetchError("401 Unauthorized"))

    with SqliteObservationStore(tmp_path / "obs.db") as store:
        poll_once(client, store, LOOKUP, TRACKED)  # type: ignore[arg-type]
        print_status(store)

    output = capsys.readouterr().out
    assert "Every recent poll failed" in output


def test_status_on_an_empty_database_says_so(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with SqliteObservationStore(tmp_path / "obs.db") as store:
        print_status(store)

    assert "No polls recorded yet." in capsys.readouterr().out


def test_build_sink_honours_the_cli_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("COLLECTION_SINK", "sqlite")
    monkeypatch.setenv("COLLECTION_DB_PATH", str(tmp_path / "obs.db"))
    monkeypatch.setenv("COLLECTION_SNAPSHOT_DIR", str(tmp_path / "snapshots"))

    assert isinstance(build_sink(CollectionConfig()), SqliteObservationStore)
    assert isinstance(build_sink(CollectionConfig(), "csv"), CsvSnapshotStore)


def test_an_unknown_sink_is_rejected_at_config_time(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COLLECTION_SINK", "postgres")

    with pytest.raises(ConfigError, match="COLLECTION_SINK"):
        CollectionConfig()
