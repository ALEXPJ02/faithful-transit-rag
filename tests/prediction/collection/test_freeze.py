"""Tests for the frozen evaluation dataset (``prediction/collection/freeze.py``).

The snapshot is written by the real store, with rows on both sides of the cut-off
instant and one alert poll exactly on it. The tests pin what is kept, what is
removed, how an alert still in the feed is clipped, that the source is never
written, and the two refusals: an existing output, and a snapshot that ends early.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from transit_rag.prediction.collection.freeze import cutoff_instant, freeze, main, sha256
from transit_rag.prediction.collection.store import SqliteObservationStore
from transit_rag.realtime.parsing import AlertScope, ServiceAlert, StopDelayObservation, TripStatus

CUTOFF = "2026-10-08"
INSTANT = datetime(2026, 10, 8, 16, 0, tzinfo=UTC)  # 03:00 daylight time on the 9th


def _at(hour: int, minute: int = 0, day: int = 8) -> str:
    return datetime(2026, 10, day, hour, minute, tzinfo=UTC).isoformat()


def _observation(service_date: str, trip: str, seen: str) -> StopDelayObservation:
    return StopDelayObservation(
        service_date=service_date,
        trip_id=trip,
        stop_id="2000331",
        route_id="NSN_1a",
        route_short_name="T1",
        stop_sequence=None,
        stops_ahead=0,
        arrival_delay_s=60,
        departure_delay_s=None,
        schedule_relationship="SCHEDULED",
        observed_at_utc=seen,
    )


def _alert(alert_id: str, *seen: str) -> list[tuple[list[ServiceAlert], list[AlertScope]]]:
    return [
        (
            [ServiceAlert(alert_id, "UNKNOWN_CAUSE", "UNKNOWN_EFFECT", "UNKNOWN_SEVERITY",
                          "North Shore Line", "Allow extra travel time.", "", at)],
            [AlertScope(alert_id, "NSN_1a", "T1", 0, "", 0, 0, at)],
        )
        for at in seen
    ]  # fmt: skip


def _snapshot(tmp_path: Path) -> Path:
    db = tmp_path / "snapshot.db"
    with SqliteObservationStore(db) as store:
        store.record_observations(
            [
                _observation("2026-10-07", "a", _at(1)),
                # The cut-off date's last train, at 02:30 Sydney time on the 9th: kept.
                _observation(CUTOFF, "late-night", _at(15, 30)),
                _observation("2026-10-09", "after", _at(21)),
            ]
        )
        for status_date in (CUTOFF, "2026-10-09"):
            store.record_trip_statuses(
                [TripStatus(status_date, f"c-{status_date}", "NSN_1a", "T1", "CANCELED", "",
                            _at(14, 0 if status_date == CUTOFF else 20, 8 if status_date == CUTOFF else 9))]
            )  # fmt: skip
        start = INSTANT - timedelta(hours=1)
        for step in range(61):  # a poll every two minutes, 15:00 to 17:00
            store.record_poll((start + timedelta(minutes=2 * step)).isoformat(), 5, 5, "ok")
        for when in (_at(15, 30), _at(16), _at(16, 30)):  # one exactly on the instant
            store.record_alert_poll(when, 1, 1, "ok")
        for alerts, scopes in (
            _alert("kept", _at(10), _at(12))
            + _alert("spanning", _at(15, 30), _at(16, 30))
            + _alert("late", _at(16, 30))
        ):
            store.record_alerts(alerts, scopes)
    return db


def _one(db: Path, sql: str) -> Any:
    connection = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        return connection.execute(sql).fetchone()[0]
    finally:
        connection.close()


def test_the_cut_off_instant_is_three_in_the_morning_sydney_time_either_side_of_a_clock_change() -> (
    None
):
    assert cutoff_instant(CUTOFF) == INSTANT  # daylight time, UTC+11
    assert cutoff_instant("2026-10-02") == datetime(2026, 10, 2, 17, 0, tzinfo=UTC)  # UTC+10


def test_nothing_after_the_cut_off_survives_and_the_source_is_untouched(tmp_path: Path) -> None:
    db = _snapshot(tmp_path)
    before = sha256(db)
    frozen = freeze(db, CUTOFF, tmp_path / "frozen.db")
    out = frozen.out
    assert sha256(db) == before

    assert frozen.removed == {
        "stop_observations": 1,
        "trip_statuses": 1,
        "poll_log": 31,  # 16:00 to 17:00 inclusive
        "alert_poll_log": 2,  # the one on the instant goes too
        "service_alerts": 1,
        "alert_scopes": 1,
    }
    assert frozen.last_service_date == CUTOFF
    assert _one(out, "SELECT COUNT(*) FROM stop_observations WHERE trip_id = 'late-night'") == 1
    assert _one(out, "SELECT MAX(service_date) FROM trip_statuses") == CUTOFF
    assert _one(out, "SELECT MAX(poll_time_utc) FROM poll_log") < INSTANT.isoformat()
    assert _one(out, "SELECT COUNT(*) FROM service_alerts WHERE alert_id = 'late'") == 0
    assert _one(out, "SELECT COUNT(*) FROM alert_scopes WHERE alert_id = 'late'") == 0
    # Still in the feed at the cut-off: last seen at the last alert poll before it.
    assert frozen.clipped_alerts == 1
    for table in ("service_alerts", "alert_scopes"):
        sql = f"SELECT last_seen_utc FROM {table} WHERE alert_id = "
        assert _one(out, sql + "'spanning'") == _at(15, 30)
        assert _one(out, sql + "'kept'") == _at(12)
    assert _one(out, "PRAGMA journal_mode") == "delete"
    assert frozen.sha256 == sha256(out)


def test_an_existing_output_and_a_snapshot_that_ends_early_are_refused(tmp_path: Path) -> None:
    db = _snapshot(tmp_path)
    with pytest.raises(ValueError, match="never overwritten"):
        freeze(db, CUTOFF, db)
    # The next day's service runs to 03:00 on the 10th; this snapshot stops at 17:00 UTC on the 8th.
    with pytest.raises(ValueError, match="pull a snapshot after it"):
        freeze(db, "2026-10-09", tmp_path / "early.db")
    assert not (tmp_path / "early.db").exists()


def test_the_command_reports_what_it_kept_and_refuses_without_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db = _snapshot(tmp_path)
    argv = ["--db", str(db), "--cutoff", CUTOFF, "--out", str(tmp_path / "frozen.db")]
    assert main(argv) == 0
    printed = capsys.readouterr().out
    assert "last service date kept: 2026-10-08" in printed
    assert "removed        1 row(s) from service_alerts" in printed
    assert main(argv) == 1
    assert "already exists" in capsys.readouterr().err
