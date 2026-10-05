"""Tests for the agent's delay tool (``agent/delays.py``).

The tool is where the hard requirement lives: a prediction leaves it only
together with its interval. These tests pin which train it predicts for, at
which stop, from which features, and with which interval.
"""

from __future__ import annotations

import json
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from transit_rag.agent.delays import (
    DelayModel,
    eras_for,
    expected_delays,
    next_call,
    running_trains,
)
from transit_rag.agent.loop import SYSTEM_TEMPLATE
from transit_rag.prediction.features.schedule import ScheduleIndex
from transit_rag.prediction.model.dataset import FEATURE_COLUMNS

AT = datetime(2026, 10, 1, 22, 30, tzinfo=UTC)  # 08:30 on Friday 2 October in Sydney
CONFORMAL = {
    "alpha": 0.1,
    "key": ["line", "lateness"],
    "global_half_width_s": 50.0,
    "half_widths_s": {"T1|late > 5 min": 120.0, "T1|on time (<= 60 s)": 30.0},
}


def _bundle(tmp_path: Path, name: str = "gtfs_schedule_20261001_abc.zip") -> Path:
    path = tmp_path / name
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "stop_times.txt",
            "trip_id,arrival_time,departure_time,stop_id,stop_sequence\n"
            "t1,08:20:00,08:20:30,A,1\n"
            "t1,08:26:00,08:26:30,B,2\n"
            "t1,08:33:00,08:33:30,C,3\n",
        )
    return path


def _events(*rows: tuple[str, str, str, float, float]) -> pd.DataFrame:
    """``(line, trip_id, stop_id, delay_s, minutes before AT)``."""
    return pd.DataFrame(
        {
            "line": [row[0] for row in rows],
            "service_date": "2026-10-02",
            "trip_id": [row[1] for row in rows],
            "stop_id": [row[2] for row in rows],
            "stops_ahead": 0,
            "delay_s": pd.array([row[3] for row in rows], dtype="Float64"),
            "observed_at": pd.to_datetime(
                [AT - timedelta(minutes=row[4]) for row in rows], utc=True
            ),
        }
    )


class _Recording:
    """A delay model that returns a fixed delay and keeps what it was given."""

    def __init__(self, delay_s: float) -> None:
        self.delay_s = delay_s
        self.frames: list[pd.DataFrame] = []

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        self.frames.append(frame)
        return np.full(len(frame), self.delay_s)


def _model(delay_s: float = 450.0, provenance: dict[str, Any] | None = None) -> DelayModel:
    return DelayModel(
        model=_Recording(delay_s),
        feature_columns=list(FEATURE_COLUMNS),
        category_levels={"route_short_name": ["T1", "T4"], "stop_id": ["A", "B", "C"]},
        conformal=CONFORMAL,
        provenance=provenance or {},
    )


def test_eras_are_the_newest_fetched_on_or_before_the_date() -> None:
    bundles = [Path(f"gtfs_schedule_202609{day}_x.zip") for day in ("28", "29", "30")] + [
        Path("gtfs_schedule_20261003_x.zip"),
        Path("gtfs_schedule.zip"),
    ]
    assert [p.name[14:22] for p in eras_for("2026-10-01", bundles)] == [
        "20260930",
        "20260929",
        "20260928",
    ]
    # A bundle fetched on the service date itself is that date's era.
    assert eras_for("2026-09-30", bundles)[0].name.startswith("gtfs_schedule_20260930")


def test_a_running_train_is_seen_in_the_last_ten_minutes_at_its_latest_stop() -> None:
    events = _events(
        ("T1", "t1", "A", 60, 9),
        ("T1", "t1", "B", 400, 3),
        ("T1", "stale", "A", 0, 15),
        ("T4", "other", "A", 0, 3),
        ("T1", "future", "A", 0, -1),
        ("T1", "at-the-moment", "A", 0, 0),
    )
    trains = running_trains(events, "T1", AT)
    assert list(trains["trip_id"]) == ["t1"]
    assert (trains["stop_id"].iloc[0], float(trains["delay_s"].iloc[0])) == ("B", 400.0)


def test_the_next_call_follows_the_timetable_order(tmp_path: Path) -> None:
    schedule = ScheduleIndex.for_trips(_bundle(tmp_path), ["t1"])
    assert [stop for stop, _ in schedule.stops_of("t1")] == ["A", "B", "C"]
    call = next_call(schedule, "t1", "B")
    assert call is not None and call[0] == "C"
    assert next_call(schedule, "t1", "C") is None  # its last stop
    assert next_call(schedule, "t1", "Z") is None  # a stop the trip does not call at


class TestDelayModel:
    def test_the_interval_is_the_one_for_the_line_and_its_lateness(self) -> None:
        model = _model()
        assert model.half_width("T1", 400) == 120.0
        assert model.half_width("T1", 30) == 30.0
        assert model.half_width("T4", 400) == 50.0  # an uncalibrated bin falls back to global

    @pytest.mark.filterwarnings("error::pandas.errors.Pandas4Warning")
    def test_an_unseen_stop_is_missing_not_some_other_stop(self) -> None:
        """Made missing explicitly: the coercion pandas does today will raise later."""
        model = _model()
        frame = pd.DataFrame(
            {
                "scheduled_arrival_s": [30000], "stop_sequence": [3], "prev_stop_delay_s": [400.0],
                "hour_local": [8], "day_of_week": [4], "is_weekend": [False], "is_peak": [True],
                "route_short_name": ["T1"], "stop_id": ["NEVER-SEEN"],
            }
        )  # fmt: skip
        model.predict(frame)
        seen = model.model.frames[0]
        assert pd.isna(seen["stop_id"].iloc[0])
        assert list(seen["stop_id"].cat.categories) == ["A", "B", "C"]

    def test_an_artefact_without_an_interval_is_refused(self, tmp_path: Path) -> None:
        import joblib

        path = tmp_path / "old.joblib"
        joblib.dump({"model": None, "feature_columns": [], "category_levels": {}}, path)
        with pytest.raises(ValueError, match="no conformal interval"):
            DelayModel.load(path)

    def test_the_split_dates_beside_the_artefact_are_read(self, tmp_path: Path) -> None:
        import joblib

        path = tmp_path / "delay.joblib"
        joblib.dump(
            {"model": None, "feature_columns": [], "category_levels": {}, "conformal": CONFORMAL},
            path,
        )
        (tmp_path / "delay_metrics.json").write_text(
            json.dumps({"split": {"train_dates": ["2026-09-03"], "test_dates": ["2026-10-02"]}})
        )
        assert DelayModel.load(path).provenance["train_dates"] == ["2026-09-03"]


def test_each_prediction_leaves_with_its_interval(tmp_path: Path) -> None:
    model = _model(450.0, provenance={"test_dates": ["2026-10-02"]})
    events = _events(("T1", "t1", "A", 60, 9), ("T1", "t1", "B", 400, 3))
    result = expected_delays(events, "T1", AT, model, [_bundle(tmp_path)], {"C": "Central"})

    assert (result["trains_running"], result["trains_predicted"]) == (1, 1)
    [train] = result["trains"]
    assert train == {
        "next_station": "Central",
        "scheduled": "08:33",
        "late_now_minutes": 6.7,
        "expected_delay_minutes": 7.5,
        "interval_90_minutes": [5.5, 9.5],  # 450 +/- 120 s: T1, already more than 5 min late
    }
    assert result["expected_more_than_5_min_late"] == 1
    assert result["this_date_was_in_training"] is False

    [features] = model.model.frames
    row = features.iloc[0]
    assert (row["stop_id"], row["stop_sequence"], row["prev_stop_delay_s"]) == ("C", 3, 400.0)
    assert row["scheduled_arrival_s"] == 8 * 3600 + 33 * 60
    assert (row["hour_local"], row["is_peak"], row["is_weekend"]) == (8, True, False)


def test_the_prompt_requires_the_interval_unnarrowed() -> None:
    assert "State the interval exactly as the tool returned it" in SYSTEM_TEMPLATE


def test_a_date_the_model_trained_on_is_said_so(tmp_path: Path) -> None:
    model = _model(provenance={"train_dates": ["2026-10-02"]})
    events = _events(("T1", "t1", "B", 400, 3))
    result = expected_delays(events, "T1", AT, model, [_bundle(tmp_path)], {})
    assert result["this_date_was_in_training"] is True


def _loop_bundle(tmp_path: Path) -> Path:
    """A trip that calls at A twice, as a City Circle service would."""
    path = tmp_path / "gtfs_schedule_20261001_loop.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "stop_times.txt",
            "trip_id,arrival_time,departure_time,stop_id,stop_sequence\n"
            "loop,08:00:00,08:00:30,A,1\n"
            "loop,08:05:00,08:05:30,B,2\n"
            "loop,08:10:00,08:10:30,A,3\n"
            "loop,08:15:00,08:15:30,C,4\n",
        )
    return path


class TestRepeatedCalls:
    """Found by Cursor Bugbot on #23: keyed by stop, the second visit overwrote the first."""

    def test_every_visit_is_kept_in_timetable_order(self, tmp_path: Path) -> None:
        schedule = ScheduleIndex.for_trips(_loop_bundle(tmp_path), ["loop"])
        assert [stop for stop, _ in schedule.stops_of("loop")] == ["A", "B", "A", "C"]

    def test_the_visit_meant_is_the_one_nearest_in_time(self, tmp_path: Path) -> None:
        schedule = ScheduleIndex.for_trips(_loop_bundle(tmp_path), ["loop"])
        first_pass = next_call(schedule, "loop", "A", scheduled_estimate_s=8 * 3600 + 60)
        second_pass = next_call(schedule, "loop", "A", scheduled_estimate_s=8 * 3600 + 10 * 60)
        assert first_pass is not None and first_pass[0] == "B"
        assert second_pass is not None and second_pass[0] == "C"

    def test_a_train_seen_on_its_second_pass_is_predicted_at_the_stop_after_it(
        self, tmp_path: Path
    ) -> None:
        # Seen at A at 08:11 Sydney time, a minute late: the 08:10 visit.
        at = datetime(2026, 10, 1, 22, 13, tzinfo=UTC)
        events = pd.DataFrame(
            {
                "line": "T1",
                "service_date": "2026-10-02",
                "trip_id": ["loop"],
                "stop_id": ["A"],
                "stops_ahead": 0,
                "delay_s": pd.array([60.0], dtype="Float64"),
                "observed_at": pd.to_datetime([at - timedelta(minutes=2)], utc=True),
            }
        )
        result = expected_delays(events, "T1", at, _model(), [_loop_bundle(tmp_path)], {})
        assert result["trains"][0]["scheduled"] == "08:15"


def test_trains_all_at_their_last_stop_are_zero_predictions_not_an_error(
    tmp_path: Path,
) -> None:
    events = _events(("T1", "t1", "C", 400, 3))  # C is t1's last call
    result = expected_delays(events, "T1", AT, _model(), [_bundle(tmp_path)], {})
    assert (result["trains_running"], result["trains_predicted"], result["trains"]) == (1, 0, [])


def test_a_bundle_given_for_station_names_is_a_timetable_era_too(tmp_path: Path) -> None:
    """Found by Cursor Bugbot on #23: --bundle was left out of the eras searched."""
    from transit_rag.agent.cli import timetable_eras

    given = _bundle(tmp_path)
    assert timetable_eras(given, []) == [given]
    assert timetable_eras(given, [given]) == [given]  # once, not twice
    assert timetable_eras(None, [given]) == [given]
