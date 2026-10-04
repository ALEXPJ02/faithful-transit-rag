"""Tests for the disruption label (``docs/08`` §3.2).

The label is the ground truth every RQ2 detector is scored against, so each
test pins one sentence of the definition. A wrong label would not make a
detector look bad; it would make the comparison between detectors mean
nothing.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from transit_rag.ingestion.alerts import AlertRecord, Incident, audit, load_alerts, load_overrides
from transit_rag.prediction.collection.store import SqliteObservationStore
from transit_rag.prediction.disruption.cli import main as label_main
from transit_rag.prediction.disruption.labels import (
    LABEL_COLUMNS,
    Coverage,
    LabelRule,
    incident_spans,
    label_snapshot,
    label_windows,
    late_needed,
    load_coverage,
    service_dates,
    services_by_window,
)
from transit_rag.realtime.parsing import AlertScope, ServiceAlert, StopDelayObservation

# 10:00 Sydney time on a weekday, before daylight saving began.
BASE = datetime(2026, 9, 29, 0, 0, tzinfo=UTC)


def _at(minutes: float) -> pd.Timestamp:
    return pd.Timestamp(BASE + timedelta(minutes=minutes))


def _events(*rows: tuple[str, str, float, float]) -> pd.DataFrame:
    """Stop events as ``(line, trip_id, delay_s, minutes after BASE)``, all reliable."""
    return pd.DataFrame(
        {
            "line": [row[0] for row in rows],
            "service_date": "2026-09-29",
            "trip_id": [row[1] for row in rows],
            "stop_id": "200060",
            "stops_ahead": 0,
            "delay_s": pd.array([row[2] for row in rows], dtype="Float64"),
            "observed_at": pd.to_datetime([_at(row[3]) for row in rows], utc=True),
        }
    )


def _coverage(windows: int = 4) -> Coverage:
    return Coverage(start=_at(0), end=_at(15 * windows), longest_alert_gap=pd.Timedelta(minutes=30))


def _incident(*alerts: tuple[str, float, float, tuple[str, ...]]) -> Incident:
    """One incident from ``(alert_id, first seen, last seen, lines)``, in minutes after BASE."""
    return Incident(
        alerts=tuple(
            AlertRecord(
                alert_id=alert_id,
                cause="TECHNICAL_PROBLEM",
                effect="UNKNOWN_EFFECT",
                header_text="North Shore Line",
                description_text="Allow extra travel time due to urgent train repairs.",
                first_seen=_at(first).to_pydatetime(),
                last_seen=_at(last).to_pydatetime(),
                lines=lines,
            )
            for alert_id, first, last, lines in alerts
        )
    )


def _window(labels: pd.DataFrame, line: str, minutes: float) -> pd.Series:
    [row] = labels[(labels["line"] == line) & (labels["window_start_utc"] == _at(minutes))].index
    return labels.loc[row]


def _label(labels: pd.DataFrame, line: str, minutes: float) -> bool | None:
    """The window's label, with ``None`` for unlabelled rather than a pandas NA."""
    value = _window(labels, line, minutes)["disrupted"]
    return None if pd.isna(value) else bool(value)


def _many(
    line: str, count: int, delay_s: float, minutes: float, prefix: str
) -> list[tuple[str, str, float, float]]:
    return [(line, f"{prefix}{i}", delay_s, minutes) for i in range(count)]


class TestRuleB:
    def test_a_quarter_late_marks_the_window_and_fewer_does_not(self) -> None:
        events = _events(
            *_many("T1", 2, 400, 1, "late"),
            *_many("T1", 6, 0, 1, "ok"),
            *_many("T1", 1, 400, 16, "late-b"),
            *_many("T1", 7, 0, 16, "ok-b"),
        )
        labels = label_windows(events, [], _coverage(), lines=("T1",))
        assert _window(labels, "T1", 0)["rule_b"]  # 2 of 8
        assert not _window(labels, "T1", 15)["rule_b"]  # 1 of 8

    def test_exactly_five_minutes_late_is_on_time(self) -> None:
        """ "More than" 300 s: TfNSW's tolerance includes the fifth minute."""
        on_time = _events(*_many("T1", 2, 300, 1, "a"), *_many("T1", 3, 0, 1, "b"))
        late = _events(*_many("T1", 2, 301, 1, "a"), *_many("T1", 3, 0, 1, "b"))
        assert not _window(label_windows(on_time, [], _coverage(), lines=("T1",)), "T1", 0)[
            "rule_b"
        ]
        assert _window(label_windows(late, [], _coverage(), lines=("T1",)), "T1", 0)["rule_b"]

    def test_a_service_is_judged_by_its_latest_observation_in_the_window(self) -> None:
        recovers = _events(("T1", "recovers", 400, 1), ("T1", "recovers", 200, 10))
        slips = _events(("T1", "slips", 0, 1), ("T1", "slips", 400, 10))
        assert services_by_window(recovers, LabelRule()).iloc[0]["n_late"] == 0
        assert services_by_window(slips, LabelRule()).iloc[0]["n_late"] == 1

    def test_rule_b_cannot_fire_on_fewer_than_the_minimum_services(self) -> None:
        """One late train at 02:00 is not a disrupted line."""
        events = _events(*_many("T1", 2, 900, 1, "night"))
        default = label_windows(events, [], _coverage(), lines=("T1",))
        literal = label_windows(events, [], _coverage(), LabelRule(min_services=1), ("T1",))
        assert not _window(default, "T1", 0)["rule_b"]
        assert _window(literal, "T1", 0)["rule_b"]
        # Below the minimum the window is still labelled, so rule (a) can mark it.
        assert _label(default, "T1", 0) is False

    def test_observations_the_delay_model_rejects_do_not_decide_a_label(self) -> None:
        events = _events(*_many("T1", 5, 0, 1, "ok"), *_many("T1", 3, 600, 1, "far"))
        events.loc[events["trip_id"].str.startswith("far"), "stops_ahead"] = 3
        ghost = _events(*_many("T1", 3, 86_000, 1, "ghost"))
        counts = services_by_window(pd.concat([events, ghost]), LabelRule()).iloc[0]
        assert (counts["n_services"], counts["n_late"]) == (5, 0)


def test_late_needed_counts_without_floating_point_drift() -> None:
    """0.3 * 10 is 3.0000000000000004; three of ten must still reach 30%."""
    needed = late_needed(pd.Series([10, 8, 5, 4]), 0.3)
    assert list(needed) == [3, 3, 2, 2]
    assert list(late_needed(pd.Series([8, 5, 4]), 0.25)) == [2, 2, 1]


class TestRuleA:
    def test_an_incident_marks_each_window_the_feed_carries_it_on_its_line_only(self) -> None:
        incident = _incident(("a", 16, 61, ("T1",)))
        labels = label_windows(_events(), [incident], _coverage(6))
        marked = labels[labels["rule_a"]]
        assert set(marked["line"]) == {"T1"}
        assert list(marked["window_start_utc"]) == [_at(15), _at(30), _at(45), _at(60)]
        assert set(marked["incident_ids"]) == {incident.incident_id}

    def test_an_alert_seen_in_a_single_poll_marks_one_window(self) -> None:
        incident = _incident(("a", 36.5, 36.5, ("T4",)))
        labels = label_windows(_events(), [incident], _coverage())
        assert list(labels.loc[labels["rule_a"], "window_start_utc"]) == [_at(30)]

    def test_the_span_is_per_line_not_per_incident(self) -> None:
        """Edgecliff named T1 for one poll and T4 for hours."""
        incident = _incident(("t4", 1, 50, ("T4",)), ("both", 20, 20, ("T1", "T4")))
        spans = {span.line: span for span in incident_spans([incident])}
        assert (spans["T1"].start, spans["T1"].end) == (_at(20), _at(20))
        assert (spans["T4"].start, spans["T4"].end) == (_at(1), _at(50))
        labels = label_windows(_events(), [incident], _coverage())
        assert labels.loc[labels["rule_a"]].groupby("line").size().to_dict() == {"T1": 1, "T4": 4}

    def test_the_gap_a_republication_leaves_is_still_covered(self) -> None:
        incident = _incident(("first", 1, 14, ("T1",)), ("second", 46, 50, ("T1",)))
        labels = label_windows(_events(), [incident], _coverage(), lines=("T1",))
        assert labels["rule_a"].all()  # 00, 15 (the gap), 30 and 45


class TestPopulation:
    def test_a_window_with_no_service_is_unlabelled_not_undisrupted(self) -> None:
        """Absent is a different claim from False, even with an incident."""
        incident = _incident(("a", 16, 20, ("T1",)))
        events = _events(*_many("T1", 5, 0, 1, "ok"))
        labels = label_windows(events, [incident], _coverage(), lines=("T1",))
        assert _label(labels, "T1", 0) is False
        assert _window(labels, "T1", 15)["rule_a"]
        assert _label(labels, "T1", 15) is None
        assert labels["disrupted"].dtype == "boolean"

    def test_either_rule_disrupts_a_labelled_window(self) -> None:
        incident = _incident(("a", 1, 2, ("T1",)))
        events = _events(*_many("T1", 5, 0, 1, "ok"), *_many("T1", 5, 900, 16, "late"))
        labels = label_windows(events, [incident], _coverage(), lines=("T1",))
        assert _label(labels, "T1", 0) is True  # rule (a)
        assert _label(labels, "T1", 15) is True  # rule (b)

    def test_the_label_table_has_its_documented_columns(self) -> None:
        labels = label_windows(_events(), [], _coverage(2))
        assert tuple(labels.columns) == LABEL_COLUMNS
        assert len(labels) == 4  # two lines x two windows


class TestServiceDates:
    def test_the_service_day_turns_over_at_three_in_the_morning(self) -> None:
        starts = pd.Series(
            pd.to_datetime(
                [
                    "2026-09-29T02:45:00+10:00",  # still the 28th's service
                    "2026-09-29T03:00:00+10:00",
                ],
                utc=True,
            )
        )
        assert list(service_dates(starts)) == ["2026-09-28", "2026-09-29"]

    def test_the_turnover_is_on_the_wall_clock_on_a_daylight_saving_day(self) -> None:
        """03:30 AEDT on 2026-10-04 is 2.5 hours after 01:59 AEST. Absolute
        arithmetic would put it back on the 3rd; the collector puts it on the 4th."""
        starts = pd.Series(pd.to_datetime(["2026-10-04T03:30:00+11:00"], utc=True))
        assert list(service_dates(starts)) == ["2026-10-04"]


def test_a_rule_that_cannot_mean_anything_is_refused() -> None:
    with pytest.raises(ValueError, match="late_share"):
        LabelRule(late_share=0)
    with pytest.raises(ValueError, match="min_services"):
        LabelRule(min_services=0)


def _observation(
    trip_id: str, delay_s: int, at: datetime, line: str = "T1"
) -> StopDelayObservation:
    return StopDelayObservation(
        service_date="2026-09-29",
        trip_id=trip_id,
        stop_id="2000331",
        route_id="NSN_1a" if line == "T1" else "ESI_1a",
        route_short_name=line,
        stop_sequence=None,
        stops_ahead=0,
        arrival_delay_s=delay_s,
        departure_delay_s=None,
        schedule_relationship="SCHEDULED",
        observed_at_utc=at.isoformat(),
    )


def collection_snapshot(tmp_path: Path) -> Path:
    """A snapshot written by the real store: an incident on T1 from 10:16 to
    10:46 Sydney time, five late T4 services at 10:20, and polls from 10:10 to
    11:07."""
    db = tmp_path / "snapshot.db"
    alert = ServiceAlert(
        "repairs",
        "TECHNICAL_PROBLEM",
        "UNKNOWN_EFFECT",
        "UNKNOWN_SEVERITY",
        "North Shore Line",
        "Allow extra travel time due to urgent train repairs at Chatswood earlier.",
        "",
        "",
    )
    with SqliteObservationStore(db) as store:
        for minutes in (10, 40):
            store.record_alert_poll(_at(minutes).isoformat(), 1, 2, "ok")
        for minutes in (16, 46):
            seen = _at(minutes).isoformat()
            store.record_alerts(
                [ServiceAlert(**{**alert.as_dict(), "observed_at_utc": seen})],
                [AlertScope("repairs", "NSN_1a", "T1", 0, "", 0, 0, seen)],
            )
        store.record_alert_poll(_at(70).isoformat(), 0, 0, "ok")
        for minutes in range(10, 68, 2):
            store.record_poll(_at(minutes).isoformat(), 10, 10, "ok")
        at = BASE + timedelta(minutes=20)
        store.record_observations(
            [_observation(f"t4-late-{i}", 600, at, "T4") for i in range(5)]
            + [_observation(f"t1-{i}", 0, at, "T1") for i in range(5)]
        )
    return db


class TestSnapshot:
    def test_coverage_rounds_in_to_complete_windows(self, tmp_path: Path) -> None:
        coverage = load_coverage(collection_snapshot(tmp_path))
        # First alert poll 10:10 -> 10:15; last trip poll 11:06 -> windows end at 11:00.
        assert (coverage.start, coverage.end) == (_at(15), _at(60))
        assert coverage.windows == 3

    def test_a_snapshot_is_labelled_and_never_written(self, tmp_path: Path) -> None:
        """Snapshots are the only copy of data that cannot be re-collected."""
        db = collection_snapshot(tmp_path)
        before = db.read_bytes()
        _, incidents = audit(load_alerts(db), load_overrides())
        labels, _ = label_snapshot(db, incidents)
        assert db.read_bytes() == before

        window = _window(labels, "T4", 15)
        assert (window["n_services"], window["n_late"], window["rule_b"]) == (5, 5, True)
        t1 = labels[labels["line"] == "T1"].set_index("window_start_utc")
        assert list(t1["rule_a"]) == [True, True, True]  # 10:16 to 10:46
        assert t1["disrupted"].isna().tolist() == [False, True, True]  # no T1 trains after 10:20

    def test_a_missing_snapshot_says_what_to_do(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="pull a snapshot"):
            load_coverage(tmp_path / "absent.db")


class TestCommand:
    def test_it_prints_the_audit_and_writes_only_where_asked(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        db = collection_snapshot(tmp_path)
        overrides = tmp_path / "none.csv"
        overrides.write_text("alert_id,is_incident,reason\n", encoding="utf-8")
        out = tmp_path / "labels.csv"

        assert label_main(["--db", str(db), "--overrides", str(overrides)]) == 0
        assert not out.exists()
        printed = capsys.readouterr().out
        assert "Disruption labels from" in printed
        assert "Incidents behind rule (a): 1" in printed
        assert "Rule (a) windows among them: 2" in printed

        assert label_main(["--db", str(db), "--overrides", str(overrides), "--out", str(out)]) == 0
        written = pd.read_csv(out)
        assert tuple(written.columns) == LABEL_COLUMNS
        # Two lines x three windows, each line observed only in the first.
        assert len(written) == 6
        assert written["disrupted"].isna().sum() == 4
        assert written["window_start_utc"].iloc[0] == "2026-09-29T00:15:00+00:00"

    def test_a_missing_snapshot_is_an_error_not_a_traceback(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert label_main(["--db", str(tmp_path / "absent.db")]) == 1
        assert "pull a snapshot" in capsys.readouterr().err
