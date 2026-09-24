"""Tests for ``transit-train``'s pipeline order.

    Named ``test_train_cli`` rather than ``test_cli``: the suite has no
    ``__init__.py`` files, so pytest imports each test module by bare
    basename and ``tests/retrieval/test_cli.py`` already holds that name.

The filters in ``main`` are only correct if they run *before* the split, and
nothing else in the suite can tell: move the ``filter_schedule_covered`` block
below ``_fit_and_score`` and every other test still passes, while every
reported metric quietly starts describing a different set of rows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from transit_rag.prediction.model import cli


def make_table(coverage_by_date: dict[str, float], per_date: int = 40) -> pd.DataFrame:
    """A table `transit-train` will accept, with known per-date join coverage."""
    rows = []
    for date, coverage in coverage_by_date.items():
        joined = round(coverage * per_date)
        for i in range(per_date):
            rows.append(
                {
                    "service_date": date,
                    "trip_id": f"{date}-trip-{i}",
                    "stop_id": 200 + i % 5,
                    "route_id": "NSN_1a",
                    "route_short_name": "T1" if i % 2 else "T4",
                    "stop_sequence": float(i % 7) if i < joined else None,
                    "scheduled_arrival_s": float(21600 + 300 * i) if i < joined else None,
                    "delay_s": float(30 + (i % 11) * 6),
                    "prev_stop_delay_s": float(20 + (i % 7) * 5),
                    "stops_ahead_final": 0,
                    "observation_count": 3,
                    "hour_local": 6 + i % 12,
                    "day_of_week": i % 7,
                    "is_weekend": False,
                    "is_peak": bool(i % 2),
                    "schedule_matched": i < joined,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def captured_split_input(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Intercept the table that reaches the split, without fitting anything."""
    captured: dict[str, Any] = {}

    def _spy(table: pd.DataFrame) -> tuple[object, dict[str, object]]:
        captured["table"] = table.copy()
        raise SystemExit("stopped before fitting")

    monkeypatch.setattr(cli, "_fit_and_score", _spy)
    return captured


def _run(monkeypatch: pytest.MonkeyPatch, table: pd.DataFrame, tmp_path: Path, *flags: str) -> None:
    path = tmp_path / "table.csv"
    table.to_csv(path, index=False)
    monkeypatch.setattr("sys.argv", ["transit-train", "--table", str(path), *flags])
    cli.main()


class TestScheduleFilterRunsBeforeTheSplit:
    def test_blind_dates_are_gone_before_anything_splits(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        captured_split_input: dict[str, Any],
    ) -> None:
        table = make_table(
            {
                "2026-09-03": 0.99,
                "2026-09-04": 0.99,
                "2026-09-05": 0.0,  # blind, and late enough to land in test
                "2026-09-06": 0.99,
            }
        )

        with pytest.raises(SystemExit):
            _run(monkeypatch, table, tmp_path)

        reached = captured_split_input["table"]
        assert "2026-09-05" not in set(reached["service_date"])
        assert sorted(reached["service_date"].unique()) == [
            "2026-09-03",
            "2026-09-04",
            "2026-09-06",
        ]

    def test_keep_schedule_blind_lets_them_through(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        captured_split_input: dict[str, Any],
    ) -> None:
        table = make_table({"2026-09-03": 0.99, "2026-09-04": 0.0})

        with pytest.raises(SystemExit):
            _run(monkeypatch, table, tmp_path, "--keep-schedule-blind")

        assert "2026-09-04" in set(captured_split_input["table"]["service_date"])


class TestEverythingBlind:
    def test_says_to_pass_a_bundle_not_to_collect_more_days(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # transit-reconcile only *warns* when it runs with no bundle, so a
        # fully unjoined table is reachable. The old message sent the user off
        # to collect more days, which is advice for a different problem.
        table = make_table({"2026-09-03": 0.0, "2026-09-04": 0.0, "2026-09-05": 0.0})

        with pytest.raises(SystemExit) as raised:
            _run(monkeypatch, table, tmp_path)

        message = str(raised.value)
        assert "schedule-blind" in message
        assert "bundle" in message
        assert "collect more days" not in message.lower()
