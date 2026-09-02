"""Tests for the append-only observation store."""

from __future__ import annotations

from pathlib import Path

from transit_rag.prediction.collection.store import ObservationStore
from transit_rag.realtime.parsing import StopDelayObservation


def _observation(
    poll_time: str = "2026-09-02T09:00:00+00:00", stop_id: str = "stop-a", delay: int = 60
) -> StopDelayObservation:
    return StopDelayObservation(
        poll_time_utc=poll_time,
        trip_id="trip-1",
        route_id="APS_1a",
        route_short_name="T1",
        stop_id=stop_id,
        stop_sequence=1,
        arrival_delay_s=delay,
        departure_delay_s=None,
        schedule_relationship="SCHEDULED",
    )


def test_writes_and_counts_observations(tmp_path: Path) -> None:
    with ObservationStore(tmp_path / "obs.db") as store:
        written = store.record_observations([_observation(stop_id="a"), _observation(stop_id="b")])

        assert written == 2
        assert store.observation_count() == 2


def test_creates_missing_parent_directories(tmp_path: Path) -> None:
    """The default path is data/, which is gitignored and so absent on a
    fresh clone — collection must not fail on first run because of it."""
    with ObservationStore(tmp_path / "nested" / "deeper" / "obs.db") as store:
        assert store.db_path.exists()


def test_repolling_the_same_stop_replaces_rather_than_duplicates(tmp_path: Path) -> None:
    """(poll_time, trip, stop) is the primary key: a retried poll must not
    inflate the row count and skew the collection-volume checkpoint."""
    with ObservationStore(tmp_path / "obs.db") as store:
        store.record_observations([_observation(delay=60)])
        store.record_observations([_observation(delay=90)])

        assert store.observation_count() == 1


def test_records_poll_outcomes_newest_first(tmp_path: Path) -> None:
    with ObservationStore(tmp_path / "obs.db") as store:
        store.record_poll("2026-09-02T09:00:00+00:00", 500, 120, "ok")
        store.record_poll("2026-09-02T09:02:00+00:00", 0, 0, "error: boom")

        recent = store.recent_poll_status()

        assert [row[0] for row in recent] == [
            "2026-09-02T09:02:00+00:00",
            "2026-09-02T09:00:00+00:00",
        ]
        assert recent[0][3].startswith("error")


def test_empty_write_is_a_no_op(tmp_path: Path) -> None:
    with ObservationStore(tmp_path / "obs.db") as store:
        assert store.record_observations([]) == 0
