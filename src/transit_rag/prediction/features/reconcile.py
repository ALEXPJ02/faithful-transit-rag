"""Collapse collected observations into one row per completed stop event.

Collection records what the feed *said* at each poll. This works out what
actually happened, which is a different thing and the reason the two steps are
separate modules: reconciliation can be rerun, rewritten and got wrong without
risking data that cannot be collected twice.

The core move is that GTFS-Realtime only reports a stop while the train has yet
to reach it. So the **last** observation naming a stop event is the closest
thing to an outcome — and ``stops_ahead`` on that final row records how close to
the event the prediction was made, which is the honest measure of how much to
trust it.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from transit_rag.prediction.features.schedule import ScheduleIndex
from transit_rag.realtime.parsing import SYDNEY

# Sydney Trains peaks, weekdays only. Deliberately a named constant rather than
# magic numbers inline: the peak flag is a model feature, so where its
# boundaries sit is a modelling decision that belongs in the write-up.
AM_PEAK = (6, 10)  # 06:00-09:59
PM_PEAK = (15, 19)  # 15:00-18:59

#: The training table's columns, in order. Documented in docs/07.
TRAINING_COLUMNS = [
    "service_date",
    "trip_id",
    "stop_id",
    "route_id",
    "route_short_name",
    "stop_sequence",
    "scheduled_arrival_s",
    "delay_s",
    "arrival_delay_s",
    "departure_delay_s",
    "prev_stop_delay_s",
    "stops_ahead_final",
    "observation_count",
    "observed_at_utc",
    "hour_local",
    "day_of_week",
    "is_weekend",
    "is_peak",
    "schedule_matched",
]

_OBSERVATION_COLUMNS = [
    "service_date",
    "trip_id",
    "stop_id",
    "route_id",
    "route_short_name",
    "stops_ahead",
    "arrival_delay_s",
    "departure_delay_s",
    "observed_at_utc",
]


def _empty_observations() -> pd.DataFrame:
    return pd.DataFrame(columns=[*_OBSERVATION_COLUMNS, "observation_count"])


def load_from_sqlite(db_path: Path) -> pd.DataFrame:
    """Read the always-on collector's upserted table.

    Already one row per stop event, so ``observation_count`` comes straight
    from the store rather than being recomputed.
    """
    if not db_path.exists():
        return _empty_observations()
    with sqlite3.connect(db_path) as connection:
        return pd.read_sql_query(
            "SELECT service_date, trip_id, stop_id, route_id, route_short_name, "
            "       stops_ahead, arrival_delay_s, departure_delay_s, "
            "       last_seen_utc AS observed_at_utc, observation_count "
            "FROM stop_observations",
            connection,
        )


def load_from_snapshots(snapshot_dir: Path) -> pd.DataFrame:
    """Read the scheduled collector's per-poll CSV snapshots.

    These do not deduplicate — the same stop event appears once per poll that
    saw it — so ``collapse_to_stop_events`` has to do that work here.
    """
    if not snapshot_dir.exists():
        return _empty_observations()
    files = sorted(p for p in snapshot_dir.glob("*/*.csv") if p.name != "_polls.csv")
    if not files:
        return _empty_observations()
    frames = [pd.read_csv(path, dtype={"stop_id": str, "trip_id": str}) for path in files]
    return pd.concat(frames, ignore_index=True)


def collapse_to_stop_events(observations: pd.DataFrame) -> pd.DataFrame:
    """One row per stop event, keeping the last observation of each.

    Counts how many observations backed each event on the way through: a stop
    event seen five times as the train approached is far better evidence than
    one glimpsed once from twenty stops out, and the model's evaluation should
    be able to tell them apart.
    """
    if observations.empty:
        return _empty_observations()

    frame = observations.copy()
    if "observation_count" not in frame.columns:
        frame["observation_count"] = 1

    key = ["service_date", "trip_id", "stop_id"]
    counts = frame.groupby(key, as_index=False)["observation_count"].sum()

    # Sort so the last row of each group is the latest observation, then take
    # it. `keep="last"` after a stable sort is the whole selection rule.
    frame = frame.sort_values([*key, "observed_at_utc"], kind="stable")
    latest = frame.drop_duplicates(subset=key, keep="last").drop(columns=["observation_count"])

    return latest.merge(counts, on=key, how="left").reset_index(drop=True)


def attach_schedule(stop_events: pd.DataFrame, index: ScheduleIndex) -> pd.DataFrame:
    """Add stop_sequence and scheduled arrival from the static bundle.

    Both are null for trips the bundle does not know — roughly 11% in practice,
    trips still running on a superseded timetable version. Those rows are kept:
    the delay itself is still a real observation, and dropping an entire class
    of trips would bias the training set toward whatever the timetable happened
    to be current for.
    """
    if stop_events.empty:
        frame = stop_events.copy()
        for column in ("stop_sequence", "scheduled_arrival_s"):
            frame[column] = pd.Series(dtype="Int64")
        frame["schedule_matched"] = pd.Series(dtype=bool)
        return frame

    sequences: list[int | None] = []
    arrivals: list[int | None] = []
    matched: list[bool] = []
    for trip_id, stop_id in zip(stop_events["trip_id"], stop_events["stop_id"], strict=True):
        scheduled = index.lookup(str(trip_id), str(stop_id))
        sequences.append(scheduled.stop_sequence if scheduled else None)
        arrivals.append(scheduled.scheduled_arrival_s if scheduled else None)
        matched.append(scheduled is not None)

    frame = stop_events.copy()
    frame["stop_sequence"] = pd.array(sequences, dtype="Int64")
    frame["scheduled_arrival_s"] = pd.array(arrivals, dtype="Int64")
    frame["schedule_matched"] = matched
    return frame


def add_previous_stop_delay(stop_events: pd.DataFrame) -> pd.DataFrame:
    """Delay at the preceding stop of the same trip.

    Ordering prefers the timetable's ``stop_sequence``, which is exact. Where
    the bundle does not know the trip it falls back to the order in which the
    stops were last observed — a train's later stops leave the feed later, so
    observation time recovers the visit order well enough to keep those trips
    rather than discarding the feature for 11% of the data.
    """
    if stop_events.empty:
        frame = stop_events.copy()
        frame["prev_stop_delay_s"] = pd.Series(dtype="Int64")
        return frame

    frame = stop_events.copy()
    # Unmatched trips sort last within their group but stay internally ordered
    # by observation time; a large sentinel keeps them from interleaving with
    # genuine sequence numbers.
    frame["_order"] = frame["stop_sequence"].astype("Float64").fillna(10**9)
    frame = frame.sort_values(
        ["service_date", "trip_id", "_order", "observed_at_utc"], kind="stable"
    )
    frame["prev_stop_delay_s"] = (
        frame.groupby(["service_date", "trip_id"], sort=False)["delay_s"].shift(1).astype("Int64")
    )
    return frame.drop(columns=["_order"]).reset_index(drop=True)


def add_time_features(stop_events: pd.DataFrame) -> pd.DataFrame:
    """Hour, day of week and peak flag, in Sydney local time.

    Taken from the observation instant rather than the scheduled time, so the
    features exist for every row including the trips the bundle cannot match.
    Local time is the point — a peak flag derived from UTC would put Sydney's
    morning peak in the middle of the night.
    """
    if stop_events.empty:
        frame = stop_events.copy()
        frame["hour_local"] = pd.Series(dtype="Int64")
        frame["day_of_week"] = pd.Series(dtype="Int64")
        frame["is_weekend"] = pd.Series(dtype=bool)
        frame["is_peak"] = pd.Series(dtype=bool)
        return frame

    frame = stop_events.copy()
    local = pd.to_datetime(frame["observed_at_utc"], utc=True, format="mixed").dt.tz_convert(SYDNEY)
    frame["hour_local"] = local.dt.hour.astype("Int64")
    frame["day_of_week"] = local.dt.dayofweek.astype("Int64")  # Monday = 0
    frame["is_weekend"] = frame["day_of_week"] >= 5

    in_am = frame["hour_local"].between(AM_PEAK[0], AM_PEAK[1] - 1)
    in_pm = frame["hour_local"].between(PM_PEAK[0], PM_PEAK[1] - 1)
    frame["is_peak"] = (in_am | in_pm) & ~frame["is_weekend"]
    return frame


def build_training_table(
    stop_events: pd.DataFrame, index: ScheduleIndex | None = None
) -> pd.DataFrame:
    """Full pipeline: stop events in, model-ready rows out."""
    frame = collapse_to_stop_events(stop_events)

    if frame.empty:
        return pd.DataFrame(columns=TRAINING_COLUMNS)

    # Prefer arrival: it is the event the model predicts. Departure stands in
    # where a stop reports only that, rather than dropping the row.
    frame["arrival_delay_s"] = pd.array(frame["arrival_delay_s"], dtype="Int64")
    frame["departure_delay_s"] = pd.array(frame["departure_delay_s"], dtype="Int64")
    frame["delay_s"] = frame["arrival_delay_s"].fillna(frame["departure_delay_s"])
    frame = frame[frame["delay_s"].notna()]

    if index is None:
        index = ScheduleIndex({}, set())
    frame = attach_schedule(frame, index)
    frame = add_previous_stop_delay(frame)
    frame = add_time_features(frame)

    frame = frame.rename(columns={"stops_ahead": "stops_ahead_final"})
    for column in TRAINING_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame[TRAINING_COLUMNS].reset_index(drop=True)
