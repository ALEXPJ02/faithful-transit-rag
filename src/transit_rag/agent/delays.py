"""O2's delay half for the agent: each running train's next stop, with its 90% interval.

The hard requirement (``docs/01`` §2): an answer resting on the prediction model
states the interval the tool returned, unnarrowed. This module produces the
prediction and the interval together, so they cannot be separated on the way to
the model.

**As of a moment, like every other tool.** A train is *running* if it was
observed in the ten minutes before the moment. Its present delay is its latest
reliable observation before the moment, its last completed stop. Its *next* stop
and scheduled arrival come from the timetable, because the feed's
``stop_sequence`` is always the sentinel. The era used is the newest one fetched on
or before the service date, then up to two older ones for trips planned under them.

**Features are assembled exactly as training built them** (``dataset.FEATURE_COLUMNS``),
with the categorical levels saved beside the model. A stop the model never saw
becomes a missing category, which XGBoost treats as missing rather than as some
other stop. The time features come from the moment, which the next stop is minutes
from.

**The interval is the one calibrated with the model**: split conformal by line and
by how late the train already is (``docs/08`` §4), read from the artefact.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from transit_rag.prediction.collection.bundles import eras_for
from transit_rag.prediction.disruption.labels import LabelRule
from transit_rag.prediction.features.reconcile import AM_PEAK, PM_PEAK
from transit_rag.prediction.features.schedule import (
    ScheduledStop,
    ScheduleIndex,
    service_day_origin,
)
from transit_rag.prediction.model.conformal import lateness_band
from transit_rag.realtime.parsing import SYDNEY

#: Observed this recently, a train is taken to be running now.
RUNNING_WITHIN = timedelta(minutes=10)

#: How many trains the tool lists, most delayed first. Enough for an answer,
#: few enough not to bury it.
MAX_TRAINS = 8


@dataclass
class DelayModel:
    """The saved delay regressor, its categories, its interval, and what it trained on."""

    model: Any
    feature_columns: list[str]
    category_levels: dict[str, list[Any]]
    conformal: dict[str, Any]
    provenance: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> DelayModel:
        """Read a ``transit-train`` artefact, and its metrics file's split dates if beside it."""
        import joblib

        artefact = joblib.load(path)
        if "conformal" not in artefact:
            raise ValueError(
                f"{path} has no conformal interval; retrain it with the current transit-train"
            )
        provenance: dict[str, Any] = {}
        metrics = path.with_name(f"{path.stem}_metrics.json")
        if metrics.exists():
            split = json.loads(metrics.read_text(encoding="utf-8")).get("split", {})
            provenance = {
                key: split.get(key, []) for key in ("train_dates", "validation_dates", "test_dates")
            }
        return cls(
            model=artefact["model"],
            feature_columns=list(artefact["feature_columns"]),
            category_levels=dict(artefact["category_levels"]),
            conformal=dict(artefact["conformal"]),
            provenance=provenance,
        )

    def half_width(self, line: str, previous_delay_s: float | None) -> float:
        """The 90% half-width for a train on ``line`` this late at its last stop."""
        band = lateness_band(pd.Series([previous_delay_s], dtype=float)).iloc[0]
        widths: dict[str, float] = self.conformal["half_widths_s"]
        return float(widths.get(f"{line}|{band}", self.conformal["global_half_width_s"]))

    def predict(self, features: pd.DataFrame) -> pd.Series:
        frame = features.loc[:, self.feature_columns].copy()
        for column, levels in self.category_levels.items():
            # A value training never saw is made missing *before* the cast. Left to
            # the cast, pandas coerces it today and will raise in a later version.
            known = frame[column].where(frame[column].isin(levels))
            frame[column] = known.astype(pd.CategoricalDtype(categories=levels))
        for column in ("is_weekend", "is_peak"):
            frame[column] = frame[column].astype(bool)
        return pd.Series(self.model.predict(frame), index=features.index, dtype=float)


def running_trains(
    events: pd.DataFrame, line: str, at: datetime, rule: LabelRule | None = None
) -> pd.DataFrame:
    """Each train seen on ``line`` in the ten minutes before ``at``, at its last completed stop."""
    rule = rule or LabelRule()
    frame = events[
        (events["line"] == line)
        & (events["observed_at"] >= pd.Timestamp(at - RUNNING_WITHIN))
        & (events["observed_at"] < pd.Timestamp(at))
        & events["delay_s"].notna()
        & (events["stops_ahead"] <= rule.max_stops_ahead)
        & (events["delay_s"].abs() <= rule.max_plausible_delay_s)
    ]
    return (
        frame.sort_values("observed_at", kind="stable")
        .drop_duplicates(["service_date", "trip_id"], keep="last")
        .reset_index(drop=True)
    )


def next_call(
    schedule: ScheduleIndex,
    trip_id: str,
    last_stop: str,
    scheduled_estimate_s: float | None = None,
) -> tuple[str, ScheduledStop] | None:
    """The call after ``last_stop`` in the trip's timetable, or ``None`` at its last.

    A trip can call at one stop twice. The visit meant is then the one whose
    scheduled time is nearest ``scheduled_estimate_s``: when the train was seen
    there, less how late it was.
    """
    calls = schedule.stops_of(trip_id)
    positions = [i for i, (stop_id, _) in enumerate(calls) if stop_id == last_stop]
    if not positions:
        return None
    position = positions[0]
    if len(positions) > 1 and scheduled_estimate_s is not None:
        position = min(
            positions,
            key=lambda i: abs(_scheduled_s(calls[i][1]) - scheduled_estimate_s),
        )
    return calls[position + 1] if position + 1 < len(calls) else None


def _scheduled_s(stop: ScheduledStop) -> float:
    seconds = stop.scheduled_arrival_s
    if seconds is None:
        seconds = stop.scheduled_departure_s
    return float("inf") if seconds is None else float(seconds)


def _service_day_seconds(observed_at: pd.Timestamp, service_date: str) -> float:
    """An instant as seconds into its service date, counted as GTFS times are."""
    return (observed_at - pd.Timestamp(service_day_origin(service_date))).total_seconds()


def _clock(seconds: int | None) -> str:
    """A GTFS time (seconds after the service day's start) as Sydney ``HH:MM``."""
    if seconds is None:
        return "unknown"
    return f"{(seconds // 3600) % 24:02d}:{(seconds % 3600) // 60:02d}"


def expected_delays(
    events: pd.DataFrame,
    line: str,
    at: datetime,
    model: DelayModel,
    bundles: Sequence[Path],
    stations: dict[str, str],
) -> dict[str, Any]:
    """The tool's answer: running trains, their next stop, and the delay with its interval."""
    trains = running_trains(events, line, at)
    if trains.empty:
        return {"line": line, "trains_running": 0, "trains": [], "note": "no train observed"}

    rows, scheduled = [], []
    for date, group in trains.groupby("service_date"):
        schedule = ScheduleIndex.across_bundles(eras_for(str(date), bundles), group["trip_id"])
        for train in group.itertuples():
            estimate = _service_day_seconds(train.observed_at, str(date)) - float(train.delay_s)
            call = next_call(schedule, str(train.trip_id), str(train.stop_id), estimate)
            if call is None:
                continue
            stop_id, stop = call
            rows.append(train)
            scheduled.append((stop_id, stop))

    date = str(trains["service_date"].iloc[0])
    seen = model.provenance
    in_training = date in seen.get("train_dates", []) or date in seen.get("validation_dates", [])
    if not rows:
        # Every running train is at its last call: nothing to predict, which is an
        # answer, not an error.
        return {
            "line": line,
            "trains_running": len(trains),
            "trains_predicted": 0,
            "expected_more_than_5_min_late": 0,
            "trains": [],
            "this_date_was_in_training": in_training,
            "note": "no running train has a next stop to predict",
        }

    local = pd.Timestamp(at).tz_convert(SYDNEY)
    weekend = local.dayofweek >= 5
    peak = (
        AM_PEAK[0] <= local.hour < AM_PEAK[1] or PM_PEAK[0] <= local.hour < PM_PEAK[1]
    ) and not weekend
    features = pd.DataFrame(
        {
            "scheduled_arrival_s": [stop.scheduled_arrival_s for _, stop in scheduled],
            "stop_sequence": [stop.stop_sequence for _, stop in scheduled],
            "prev_stop_delay_s": [float(train.delay_s) for train in rows],
            "hour_local": local.hour,
            "day_of_week": local.dayofweek,
            "is_weekend": weekend,
            "is_peak": peak,
            "route_short_name": line,
            "stop_id": [stop_id for stop_id, _ in scheduled],
        }
    )
    listed = []
    predicted = model.predict(features)
    for train, (stop_id, stop), expected in zip(rows, scheduled, predicted, strict=True):
        width = model.half_width(line, float(train.delay_s))
        listed.append(
            {
                "next_station": stations.get(stop_id, stop_id),
                "scheduled": _clock(stop.scheduled_arrival_s),
                "late_now_minutes": round(float(train.delay_s) / 60, 1),
                "expected_delay_minutes": round(expected / 60, 1),
                "interval_90_minutes": [
                    round((expected - width) / 60, 1),
                    round((expected + width) / 60, 1),
                ],
            }
        )
    listed.sort(key=lambda item: item["expected_delay_minutes"], reverse=True)
    return {
        "line": line,
        "trains_running": len(trains),
        "trains_predicted": len(listed),
        "expected_more_than_5_min_late": sum(item["expected_delay_minutes"] > 5 for item in listed),
        "trains": listed[:MAX_TRAINS],
        "interval": "90% prediction interval, split conformal by line and how late the train already is",
        "this_date_was_in_training": in_training,
        "note": "model estimates of each train's delay at its next stop; state each with its interval",
    }
