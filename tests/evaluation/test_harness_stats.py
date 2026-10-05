"""Tests for the harness's statistics: date-resampled intervals and lead time.

Both are where a plausible-looking number can be quietly wrong: an interval that
resampled rows instead of days, or a lead time whose sign or search window is off.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from transit_rag.evaluation.lead_time import LOOKBACK, lead_times, summarise
from transit_rag.evaluation.reasons_cli import macro_f1_interval
from transit_rag.evaluation.stats import bootstrap_by_date
from transit_rag.ingestion.alerts import AlertRecord, Incident
from transit_rag.prediction.disruption.labels import WINDOW


class TestBootstrapByDate:
    def test_whole_dates_are_resampled_never_single_rows(self) -> None:
        """Day one holds 1 row and day two holds 10: every resampled total is 2, 11 or 20."""
        frame = pd.DataFrame({"service_date": ["d1"] + ["d2"] * 10, "x": 1})
        seen: list[float] = []

        def total(rows: pd.DataFrame) -> float:
            seen.append(float(len(rows)))
            return float(len(rows))

        interval = bootstrap_by_date(frame, total, resamples=300)
        assert interval.point == 11
        assert set(seen[1:]) <= {2.0, 11.0, 20.0}
        assert interval.dates == 2

    def test_the_seed_makes_the_interval_reproducible(self) -> None:
        frame = pd.DataFrame({"service_date": [f"d{i % 5}" for i in range(50)], "x": range(50)})
        first = bootstrap_by_date(frame, lambda rows: float(rows["x"].mean()), resamples=200)
        second = bootstrap_by_date(frame, lambda rows: float(rows["x"].mean()), resamples=200)
        assert (first.low, first.high) == (second.low, second.high)
        assert first.low <= first.point <= first.high

    def test_resamples_where_the_metric_is_undefined_are_counted(self) -> None:
        """Only one of four dates has a positive: most resamples have none."""
        frame = pd.DataFrame({"service_date": ["d1", "d2", "d3", "d4"], "positive": [1, 0, 0, 0]})

        def share(rows: pd.DataFrame) -> float:
            return 1.0 if rows["positive"].any() else math.nan

        interval = bootstrap_by_date(frame, share, resamples=500)
        assert 0 < interval.defined < 1
        assert "defined in" in interval.describe()

    def test_one_date_has_no_interval(self) -> None:
        frame = pd.DataFrame({"service_date": ["d1", "d1"], "x": [1, 2]})
        interval = bootstrap_by_date(frame, lambda rows: float(rows["x"].sum()))
        assert interval.point == 3 and math.isnan(interval.low)
        assert "no interval" in interval.describe()


# 10:00 Sydney time
ALERT = datetime(2026, 9, 29, 0, 0, tzinfo=UTC)


def _incident(line: str = "T1", minutes: int = 60) -> Incident:
    return Incident(
        alerts=(
            AlertRecord(
                "a", "TECHNICAL_PROBLEM", "UNKNOWN_EFFECT", "h", "Allow extra travel time.",
                ALERT, ALERT + timedelta(minutes=minutes), (line,),
            ),
        )
    )  # fmt: skip


def _rows(flag_ends_minutes: list[int], line: str = "T1") -> tuple[pd.DataFrame, pd.Series]:
    """Windows from 3 h before to 2 h after the alert; flagged where they end at the given offsets."""
    starts = pd.date_range(ALERT - timedelta(hours=3), ALERT + timedelta(hours=2), freq=WINDOW)
    table = pd.DataFrame({"line": line, "window_start_utc": starts})
    ends = table["window_start_utc"] + WINDOW
    offsets = ((ends - pd.Timestamp(ALERT)).dt.total_seconds() // 60).astype(int)
    return table, offsets.isin(flag_ends_minutes)


class TestLeadTime:
    def test_a_flag_before_the_alert_is_a_positive_lead(self) -> None:
        table, flagged = _rows([-30, 15])
        [lead] = lead_times(table, flagged, [_incident()])
        assert lead.lead_minutes == 30

    def test_a_flag_only_after_the_alert_is_a_negative_lead(self) -> None:
        table, flagged = _rows([15])
        [lead] = lead_times(table, flagged, [_incident()])
        assert lead.lead_minutes == -15

    def test_a_flag_earlier_than_the_lookback_is_not_this_incidents_warning(self) -> None:
        early = -int(LOOKBACK.total_seconds() // 60) - 15
        table, flagged = _rows([early])
        [lead] = lead_times(table, flagged, [_incident()])
        assert lead.lead_minutes is None

    def test_a_flag_on_the_other_line_does_not_count(self) -> None:
        t1, unflagged = _rows([])
        t4, flagged_t4 = _rows([-15], line="T4")
        table = pd.concat([t1, t4], ignore_index=True)
        flagged = pd.concat([unflagged, flagged_t4], ignore_index=True)
        [lead] = lead_times(table, flagged, [_incident(line="T1")])
        assert lead.lead_minutes is None  # T4 was flagged; the T1 incident was missed

    def test_a_flag_after_the_incident_left_the_feed_does_not_count(self) -> None:
        table, flagged = _rows([90])  # the incident's last sighting is at +60
        [lead] = lead_times(table, flagged, [_incident(minutes=60)])
        assert lead.lead_minutes is None

    def test_an_incident_outside_the_rows_is_not_scored(self) -> None:
        table, flagged = _rows([-15])
        later = Incident(
            alerts=(
                AlertRecord(
                    "b", "TECHNICAL_PROBLEM", "UNKNOWN_EFFECT", "h", "Allow extra travel time.",
                    ALERT + timedelta(days=2), ALERT + timedelta(days=2, hours=1), ("T1",),
                ),
            )
        )  # fmt: skip
        assert lead_times(table, flagged, [later]) == []

    def test_the_summary_counts_and_takes_the_median(self) -> None:
        table, flagged = _rows([-30])
        leads = lead_times(table, flagged, [_incident()]) * 2
        summary = summarise(leads)
        assert (summary.incidents, summary.detected, summary.before_alert) == (2, 2, 2)
        assert summary.median_lead_minutes == 30
        assert "2/2 incident(s) flagged" in summary.describe()


def test_the_reasons_interval_averages_repeats_inside_each_resample() -> None:
    truth = ["technical", "network_incident", "technical", "other_unknown"]
    runs: list[list[str | None]] = [
        ["technical", "technical", "technical", "other_unknown"],
        ["technical", "network_incident", None, "other_unknown"],
    ]
    interval = macro_f1_interval(truth, runs, ["d1", "d2", "d3", "d4"])
    # Run one: technical 2x2/(3+2) = 0.8, network 0, unknown 1 -> 0.6.
    # Run two: technical 2x1/(1+2) = 2/3, network 1, unknown 1 -> 8/9.
    assert interval.point == pytest.approx((0.6 + 8 / 9) / 2)
    assert interval.low <= interval.point <= interval.high
