"""Tests for what a detector may see, what it is asked, and how it is scored.

A leaked feature would not make a detector look bad; it would make it look
excellent. So most of these tests pin a boundary: features look back, targets
look forward, the two never share a window, and the alert feature knows only
what the feed had shown by then.
"""

from __future__ import annotations

import dataclasses
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from transit_rag.ingestion.alerts import AlertRecord
from transit_rag.prediction.collection.store import SqliteObservationStore
from transit_rag.prediction.disruption.detect_cli import main as detect_main
from transit_rag.prediction.disruption.detection import (
    best_f1_threshold,
    persistence,
    score_detector,
)
from transit_rag.prediction.disruption.features import (
    FEATURE_COLUMNS,
    FORBIDDEN,
    HISTORY_WINDOWS,
    alert_in_feed,
    detection_table,
)
from transit_rag.prediction.disruption.labels import Coverage, label_windows
from transit_rag.realtime.parsing import StopDelayObservation

# 10:00 Sydney time, a Tuesday before daylight saving.
BASE = datetime(2026, 9, 29, 0, 0, tzinfo=UTC)


def _at(minutes: float) -> pd.Timestamp:
    return pd.Timestamp(BASE + timedelta(minutes=minutes))


def _events(*rows: tuple[str, str, float, float]) -> pd.DataFrame:
    """``(line, trip_id, delay_s, minutes after BASE)``, all reliable."""
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


def _window_of_services(
    window: int, late: int, on_time: int = 5
) -> list[tuple[str, str, float, float]]:
    minute = 15.0 * window + 1
    return [("T1", f"w{window}-late{i}", 900.0, minute) for i in range(late)] + [
        ("T1", f"w{window}-ok{i}", 0.0, minute) for i in range(on_time)
    ]


def _alert(
    alert_id: str, first: float, last: float, *, cause: str = "TECHNICAL_PROBLEM"
) -> AlertRecord:
    return AlertRecord(
        alert_id=alert_id,
        cause=cause,
        effect="UNKNOWN_EFFECT",
        header_text="North Shore Line",
        description_text="Allow extra travel time due to urgent train repairs earlier.",
        first_seen=_at(first).to_pydatetime(),
        last_seen=_at(last).to_pydatetime(),
        lines=("T1",),
    )


def _table(windows: int, *service_windows: list[tuple[str, str, float, float]]) -> pd.DataFrame:
    events = _events(*(row for rows in service_windows for row in rows))
    coverage = Coverage(_at(0), _at(15 * windows), pd.Timedelta(minutes=30))
    labels = label_windows(events, [], coverage, lines=("T1",))
    polls = pd.Series(pd.to_datetime([_at(m) for m in range(0, 15 * windows, 30)], utc=True))
    return detection_table(labels, events, [], polls)


class TestFeaturesLookBack:
    def test_lags_are_earlier_windows_of_the_same_line(self) -> None:
        table = _table(5, *(_window_of_services(w, late=w) for w in range(5)))
        row = table.iloc[4]
        assert row["share_late"] == pytest.approx(4 / 9)
        assert row["share_late_lag1"] == pytest.approx(3 / 8)
        assert row["share_late_lag3"] == pytest.approx(1 / 6)
        assert pd.isna(table.iloc[0]["share_late_lag1"])

    def test_no_feature_carries_the_target_windows_values(self) -> None:
        """Window 3 is the only disrupted one; the row predicting it must not know."""
        table = _table(6, *(_window_of_services(w, late=5 if w == 3 else 0) for w in range(6)))
        predicting = table.iloc[2]  # its target covers windows 3 and 4
        assert bool(predicting["target"]) is True
        for column in FEATURE_COLUMNS:
            value = predicting[column]
            if column == "line" or pd.isna(value):
                continue
            assert value != pytest.approx(5 / 10), column

    def test_the_feature_list_and_the_forbidden_list_are_disjoint(self) -> None:
        assert not set(FEATURE_COLUMNS) & set(FORBIDDEN)
        table = _table(4, *(_window_of_services(w, late=0) for w in range(4)))
        assert set(FEATURE_COLUMNS) <= set(table.columns)
        assert sum(name.endswith(f"_lag{HISTORY_WINDOWS}") for name in FEATURE_COLUMNS) == 3


class TestTarget:
    def test_it_is_either_of_the_next_two_windows(self) -> None:
        table = _table(6, *(_window_of_services(w, late=5 if w == 3 else 0) for w in range(6)))
        assert list(table["target"].iloc[:4]) == [False, True, True, False]

    def test_unknown_unless_both_windows_are_known_or_one_is_disrupted(self) -> None:
        # Windows 0-2 have services; window 3 has none; window 4 is disrupted.
        rows = [_window_of_services(w, late=0) for w in range(3)] + [_window_of_services(4, late=5)]
        target = _table(5, *rows)["target"]
        assert bool(target.iloc[0]) is False  # windows 1 and 2 both known, both fine
        assert pd.isna(target.iloc[1])  # window 2 fine, window 3 unknown
        assert bool(target.iloc[2]) is True  # window 3 unknown, window 4 disrupted
        assert pd.isna(target.iloc[4])  # nothing after the last window

    def test_a_target_crossing_the_service_day_is_unknown(self) -> None:
        """The 02:30 window's target reaches 03:00, which is the next service date."""
        start = datetime(2026, 9, 29, 16, 15, tzinfo=UTC)  # 02:15 Sydney on the 30th
        events = pd.DataFrame(
            {
                "line": "T1",
                "service_date": "2026-09-29",
                "trip_id": [f"t{i}" for i in range(20)],
                "stop_id": "1",
                "stops_ahead": 0,
                "delay_s": pd.array([0.0] * 20, dtype="Float64"),
                "observed_at": pd.to_datetime(
                    [start + timedelta(minutes=15 * (i % 4) + 1) for i in range(20)], utc=True
                ),
            }
        )
        coverage = Coverage(
            pd.Timestamp(start), pd.Timestamp(start + timedelta(hours=1)), pd.Timedelta(0)
        )
        labels = label_windows(events, [], coverage, lines=("T1",))
        table = detection_table(labels, events, [], pd.Series([pd.Timestamp(start)]))
        assert list(table["service_date"]) == ["2026-09-29"] * 3 + ["2026-09-30"]
        assert bool(table["target"].iloc[0]) is False  # 02:30 and 02:45, same date
        assert pd.isna(table["target"].iloc[1])  # 02:45 and 03:00 straddle the turnover
        assert pd.isna(table["target"].iloc[2])


class TestAlertInFeed:
    def _frame(self, windows: int) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "line": "T1",
                "window_start_utc": pd.to_datetime([_at(15 * w) for w in range(windows)], utc=True),
            }
        )

    def test_it_is_known_only_from_the_poll_that_first_shows_it(self) -> None:
        polls = pd.Series(pd.to_datetime([_at(m) for m in (0, 30, 60, 90)], utc=True))
        flag = alert_in_feed(self._frame(8), [_alert("a", 30, 60)], polls)
        # Windows end at 15, 30, ...; the last poll by each end is 0, 30, 30, 60, 60, 90, 90, 90.
        assert list(flag) == [False, True, True, True, True, False, False, False]

    def test_an_alert_that_will_become_a_standing_notice_counts_until_it_has(self) -> None:
        polls = pd.Series(pd.to_datetime([_at(m) for m in range(0, 30 * 60, 30)], utc=True))
        frame = pd.DataFrame(
            {"line": "T1", "window_start_utc": pd.to_datetime([_at(60), _at(25 * 60)], utc=True)}
        )
        flag = alert_in_feed(frame, [_alert("notice", 0, 29 * 60)], polls)
        assert list(flag) == [True, False]  # in hour one nobody knew; at hour 25 everyone did

    def test_maintenance_and_other_lines_never_count(self) -> None:
        polls = pd.Series(pd.to_datetime([_at(0)], utc=True))
        frame = self._frame(1)
        assert not alert_in_feed(frame, [_alert("m", 0, 0, cause="MAINTENANCE")], polls).any()
        t4_only = dataclasses.replace(_alert("t4", 0, 0), lines=("T4",))
        assert not alert_in_feed(frame, [t4_only], polls).any()


def _scored(target: list[object], score: list[float], **extra: list[bool]) -> pd.DataFrame:
    n = len(target)
    return pd.DataFrame(
        {
            "target": pd.array(target, dtype="boolean"),
            "target_rule_a": extra.get(
                "rule_a", [bool(t) if t is not pd.NA else False for t in target]
            ),
            "target_rule_b": extra.get("rule_b", [False] * n),
            "service_date": ["2026-09-29"] * (n // 2) + ["2026-09-30"] * (n - n // 2),
            "score": score,
        }
    )


class TestScoring:
    def test_precision_recall_f1_and_alarms_on_known_rows_only(self) -> None:
        table = _scored([True, True, False, False, pd.NA, False], [0.9, 0.2, 0.8, 0.1, 0.9, 0.1])
        scores = score_detector(table, table["score"], threshold=0.5)
        assert (scores.n, scores.positives) == (5, 2)
        assert scores.precision == pytest.approx(0.5)
        assert scores.recall == pytest.approx(0.5)
        assert scores.f1 == pytest.approx(0.5)
        assert scores.false_alarms_per_day == pytest.approx(0.5)  # one false alarm, two days
        # Ranked 0.9 hit, 0.8 miss, 0.2 hit: (1 x 1/2) + (2/3 x 1/2).
        assert scores.average_precision == pytest.approx(5 / 6)

    def test_flags_that_never_hit_score_zero_not_undefined(self) -> None:
        table = _scored([True, False, False, False], [0.1, 0.9, 0.9, 0.1])
        assert score_detector(table, table["score"], 0.5).f1 == 0.0

    def test_with_nothing_to_find_ap_is_undefined(self) -> None:
        table = _scored([False, False], [0.3, 0.6])
        assert math.isnan(score_detector(table, table["score"], 0.5).average_precision)

    def test_recall_is_split_by_the_rule_behind_each_target(self) -> None:
        table = _scored(
            [True, True, False],
            [0.9, 0.1, 0.1],
            rule_a=[True, False, False],
            rule_b=[True, True, False],
        )
        scores = score_detector(table, table["score"], 0.5)
        assert (scores.recall_rule_a, scores.recall_rule_b) == (1.0, 0.5)

    def test_the_threshold_maximises_f1_and_prefers_fewer_alarms_on_a_tie(self) -> None:
        table = _scored([True, False, True, False], [0.9, 0.7, 0.6, 0.2])
        # >=0.9: F1 2/3; >=0.6: F1 0.8; >=0.2: F1 2/3.
        assert best_f1_threshold(table, table["score"]) == pytest.approx(0.6)
        # >=0.9: 2 x 1 / (1 + 2) = 2/3; >=0.5: 2 x 2 / (4 + 2) = 2/3.
        tie = _scored([True, False, False, True], [0.9, 0.5, 0.5, 0.5])
        assert best_f1_threshold(tie, tie["score"]) == pytest.approx(0.9)


def test_persistence_reads_the_current_window_as_not_disrupted_when_nothing_ran() -> None:
    table = pd.DataFrame({"disrupted_now": [True, False, False]})
    assert list(persistence(table)) == [1.0, 0.0, 0.0]
    assert np.issubdtype(persistence(table).dtype, np.floating)


def _observation(trip_id: str, delay_s: int, at: datetime, day: int) -> StopDelayObservation:
    return StopDelayObservation(
        service_date=(BASE + timedelta(days=day)).date().isoformat(),
        trip_id=trip_id,
        stop_id="2000331",
        route_id="NSN_1a",
        route_short_name="T1",
        stop_sequence=None,
        stops_ahead=0,
        arrival_delay_s=delay_s,
        departure_delay_s=None,
        schedule_relationship="SCHEDULED",
        observed_at_utc=at.isoformat(),
    )


def _week_snapshot(tmp_path: Path) -> Path:
    """Seven days, each with one disrupted T1 window at 10:15 Sydney time."""
    db = tmp_path / "week.db"
    with SqliteObservationStore(db) as store:
        for day in range(7):
            start = BASE + timedelta(days=day)
            store.record_alert_poll(start.isoformat(), 0, 0, "ok")
            for minute in range(0, 60, 2):
                store.record_poll((start + timedelta(minutes=minute)).isoformat(), 10, 10, "ok")
            seen = start + timedelta(minutes=20)
            store.record_observations(
                [_observation(f"d{day}-t{i}", 900 if i < 3 else 0, seen, day) for i in range(6)]
            )
    return db


def test_the_detect_command_scores_the_baseline_and_writes_only_where_asked(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db = _week_snapshot(tmp_path)
    overrides = tmp_path / "none.csv"
    overrides.write_text("alert_id,is_incident,reason\n", encoding="utf-8")
    out = tmp_path / "detection.csv"

    assert detect_main(["--db", str(db), "--overrides", str(overrides)]) == 0
    assert not out.exists()
    printed = capsys.readouterr().out
    assert "Split by service date, 70/15/15:" in printed
    assert "Persistence -- the next 30 minutes are disrupted if this window is:" in printed

    assert detect_main(["--db", str(db), "--overrides", str(overrides), "--out", str(out)]) == 0
    written = pd.read_csv(out)
    assert set(FEATURE_COLUMNS) | {"target", "disrupted_now"} <= set(written.columns)
    assert written["target"].dropna().astype(bool).sum() >= 7  # each day's 10:15 window
