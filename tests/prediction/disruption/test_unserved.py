"""Tests for the label check's unserved services (``prediction/disruption/unserved.py``).

They pin where a cancelled or skipped call is placed, the condition that the status
was in the feed at the call's time, that a trip seen running is not also counted as
cancelled, the GTFS time origin on a daylight-saving day, and that the check can
turn a window disrupted while the label itself never changes.
"""

from __future__ import annotations

import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from transit_rag.prediction.collection.store import SqliteObservationStore
from transit_rag.prediction.disruption import cli
from transit_rag.prediction.disruption.cli import main as label_main
from transit_rag.prediction.disruption.labels import LabelRule, services_by_window
from transit_rag.prediction.disruption.unserved import (
    CANCELLED,
    SKIPPED,
    Placement,
    unserved_calls,
)
from transit_rag.prediction.features.schedule import service_day_origin
from transit_rag.realtime.parsing import SYDNEY, StopDelayObservation, TripStatus

DATE = "2026-09-29"


def _local(clock: str, date: str = DATE) -> pd.Timestamp:
    return pd.Timestamp(f"{date} {clock}").tz_localize(SYDNEY).tz_convert(UTC)


def _bundle(tmp_path: Path) -> Path:
    path = tmp_path / "gtfs_schedule_20260929_test.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "stop_times.txt",
            "trip_id,arrival_time,departure_time,stop_id,stop_sequence\n"
            "T,08:00:00,08:00:00,S1,1\n"
            "T,08:05:00,08:05:00,S5,2\n"
            "T,08:20:00,08:20:00,S2,3\n"
            "U,08:05:00,08:05:00,S1,1\n"
            "U,08:10:00,08:10:00,S2,2\n"
            "U,08:40:00,08:40:00,S3,3\n"
            "D,,08:00:00,S1,1\n"
            "c1,08:07:00,08:07:00,2000331,1\n"
            "c1,08:22:00,08:22:00,2000331,2\n",
        )
    return path


def _status(
    trip: str, relationship: str, first: str, last: str, skipped: str = "", date: str = DATE
) -> dict[str, object]:
    return {
        "service_date": date,
        "trip_id": trip,
        "trip_relationship": relationship,
        "line": "T1",
        "skipped_stop_ids": skipped,
        "first_seen_utc": _local(first, date),
        "last_seen_utc": _local(last, date),
    }


def _events(*seen: tuple[str, str]) -> pd.DataFrame:
    """Reliable, on-time stop events as ``(trip_id, local clock)`` on T1."""
    return pd.DataFrame(
        {
            "line": pd.Series(["T1"] * len(seen), dtype="string"),
            "service_date": pd.Series([DATE] * len(seen), dtype="string"),
            "trip_id": pd.Series([trip for trip, _ in seen], dtype="string"),
            "stop_id": pd.Series(["S1"] * len(seen), dtype="string"),
            "stops_ahead": [0] * len(seen),
            "delay_s": pd.Series([0.0] * len(seen), dtype="Float64"),
            "observed_at": pd.Series(
                [_local(clock) for _, clock in seen], dtype="datetime64[us, UTC]"
            ),
        }
    )


def _placed(placement_calls: pd.DataFrame) -> set[tuple[str, pd.Timestamp, str]]:
    return {
        (row.trip_id, row.window_start_utc, row.why)
        for row in placement_calls.itertuples(index=False)
    }


class TestPlacement:
    def test_a_call_counts_only_while_its_status_stood(self, tmp_path: Path) -> None:
        statuses = pd.DataFrame(
            [
                _status("T", "CANCELED", "07:30", "08:10"),  # withdrawn before its 08:20 call
                _status("U", "SCHEDULED", "07:00", "09:00", skipped="S2,S9"),
                _status("X", "CANCELED", "07:00", "09:00"),  # in no timetable
            ]
        )
        placement = unserved_calls(statuses, _events(), [_bundle(tmp_path)])
        assert _placed(placement.calls) == {
            ("T", _local("08:00"), CANCELLED),
            ("U", _local("08:00"), SKIPPED),  # its 08:10 call at S2; S9 is not on its path
        }
        assert len(placement.calls) == 2  # T's 08:00 and 08:05 calls are one service there
        assert placement.unplaced == [(DATE, "X", CANCELLED)]
        assert placement.considered == {CANCELLED: 2, SKIPPED: 1}
        assert placement.trips(CANCELLED) == 1

    def test_a_status_holds_until_the_next_poll(self, tmp_path: Path) -> None:
        held = unserved_calls(
            pd.DataFrame([_status("T", "CANCELED", "07:30", "08:18")]),
            _events(),
            [_bundle(tmp_path)],
        )
        assert ("T", _local("08:15"), CANCELLED) in _placed(held.calls)  # 08:20 <= 08:18 + 2 min
        lapsed = unserved_calls(
            pd.DataFrame([_status("T", "CANCELED", "07:30", "08:17")]),
            _events(),
            [_bundle(tmp_path)],
        )
        assert ("T", _local("08:15"), CANCELLED) not in _placed(lapsed.calls)

    def test_a_call_before_the_status_was_published_does_not_count(self, tmp_path: Path) -> None:
        late = unserved_calls(
            pd.DataFrame([_status("T", "CANCELED", "08:06", "09:00")]),  # after 08:00 and 08:05
            _events(),
            [_bundle(tmp_path)],
        )
        assert _placed(late.calls) == {("T", _local("08:15"), CANCELLED)}

    def test_a_trip_seen_running_is_not_also_cancelled_but_a_skip_still_counts(
        self, tmp_path: Path
    ) -> None:
        statuses = pd.DataFrame(
            [
                _status("T", "CANCELED", "07:30", "09:00"),
                _status("U", "SCHEDULED", "07:00", "09:00", skipped="S2"),
            ]
        )
        seen = _events(("T", "08:05"), ("U", "08:06"))
        placement = unserved_calls(statuses, seen, [_bundle(tmp_path)])
        assert _placed(placement.calls) == {
            ("T", _local("08:15"), CANCELLED),  # not seen in the 08:15 window
            ("U", _local("08:00"), SKIPPED),
        }

    def test_times_count_from_noon_less_twelve_hours_on_a_clock_change(
        self, tmp_path: Path
    ) -> None:
        # Sydney's clocks went forward at 02:00 on 2026-10-04.
        assert service_day_origin("2026-10-04") == datetime(2026, 10, 3, 13, 0, tzinfo=UTC)
        assert service_day_origin(DATE) == datetime(2026, 9, 28, 14, 0, tzinfo=UTC)
        statuses = pd.DataFrame([_status("D", "CANCELED", "05:00", "10:00", date="2026-10-04")])
        placement = unserved_calls(statuses, _events(), [_bundle(tmp_path)])
        # Timetabled 08:00, a departure with no arrival time, is 08:00 daylight time and
        # 21:00 UTC: not 09:00, as counting from midnight would say.
        assert _placed(placement.calls) == {
            ("D", pd.Timestamp("2026-10-03 21:00", tz=UTC), CANCELLED)
        }


def test_an_unserved_service_is_late_and_never_counted_twice() -> None:
    seen = _events(("a", "08:01"), ("b", "08:02"))
    rule = LabelRule()
    base = services_by_window(seen, rule)
    assert (int(base["n_services"].iloc[0]), int(base["n_late"].iloc[0])) == (2, 0)
    unserved = pd.DataFrame(
        {
            "line": ["T1", "T1"],
            "window_start_utc": [_local("08:00"), _local("08:00")],
            "service_date": [DATE, DATE],
            "trip_id": ["a", "c"],  # "a" was seen there; "c" never was
        }
    )
    checked = services_by_window(seen, rule, unserved)
    assert (int(checked["n_services"].iloc[0]), int(checked["n_late"].iloc[0])) == (3, 2)


def _observation(trip: str, delay_s: int, clock: str) -> StopDelayObservation:
    return StopDelayObservation(
        DATE, trip, "2000331", "ESI_1a", "T4", None, 0, delay_s, None, "SCHEDULED",
        _local(clock).isoformat(),
    )  # fmt: skip


def test_the_check_reports_a_window_it_turns_disrupted_and_leaves_the_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Four T4 services, one late, is below the minimum; one cancellation makes it two of five."""
    db = tmp_path / "snapshot.db"
    with SqliteObservationStore(db) as store:
        for clock in ("07:55", "09:05"):
            store.record_alert_poll(_local(clock).isoformat(), 0, 0, "ok")
        start = _local("07:56")
        for step in range(35):
            store.record_poll((start + timedelta(minutes=2 * step)).isoformat(), 4, 4, "ok")
        store.record_observations(
            [_observation("late", 600, "08:05")]
            + [_observation(f"on-time-{i}", 0, "08:05") for i in range(3)]
        )
        for clock in ("07:30", "09:00"):
            store.record_trip_statuses(
                [TripStatus(DATE, "c1", "ESI_1a", "T4", "CANCELED", "", _local(clock).isoformat())]
            )
    overrides = tmp_path / "none.csv"
    overrides.write_text("alert_id,is_incident,reason\n", encoding="utf-8")
    monkeypatch.setattr(cli, "discover", lambda: [_bundle(tmp_path)])
    out = tmp_path / "labels.csv"

    argv = ["--db", str(db), "--overrides", str(overrides), "--out", str(out)]
    assert label_main([*argv, "--cancellation-check"]) == 0
    printed = capsys.readouterr().out
    assert "Cancelled: 1 trip(s), 1 unserved in 2 window(s) while the cancellation stood" in printed
    # 08:15 had no T4 service seen at all: the cancelled call brings it into the population.
    assert "Rule (b) windows: 0 -> 1. Disrupted windows: 0 -> 1. Newly labelled: 1." in printed
    assert "T4 2026-09-29 08:00: 1/4 late -> 2/5, now disrupted" in printed
    assert "No window changes its label." not in printed
    # The label written is the label, with or without the check.
    written = pd.read_csv(out)
    assert not written["disrupted"].fillna(False).astype(bool).any()


def test_with_nothing_flipped_the_check_names_the_nearest_window() -> None:
    windows = pd.DataFrame(
        {
            "line": ["T1", "T4", "T1"],
            "window_start_utc": [_local("03:00"), _local("08:00"), _local("08:00")],
            "n_services": [1, 16, 40],
            "n_late": [0, 3, 0],
            "rule_b": [False, False, False],
            "disrupted": pd.array([False, False, False], dtype="boolean"),
            "n_services_check": [2, 17, 41],
            "n_late_check": [0, 4, 1],
            "rule_b_check": [False, False, False],
            "disrupted_check": pd.array([False, False, False], dtype="boolean"),
        }
    )
    placement = Placement(
        calls=pd.DataFrame(columns=["line", "window_start_utc", "service_date", "trip_id", "why"]),
        considered={CANCELLED: 0, SKIPPED: 0},
    )
    lines = cli._check_lines(placement, windows, LabelRule())
    assert "  No window changes its label." in lines
    # T4 needs ceil(17 / 4) = 5 late and has 4. The 03:00 window is also one short, but
    # with two services rule (b) could never fire on it, so it is not near.
    assert lines[-1] == (
        "  Nearest: T4 2026-09-29 08:00, 4 of 17 late with the check, 1 short of rule (b)."
    )
