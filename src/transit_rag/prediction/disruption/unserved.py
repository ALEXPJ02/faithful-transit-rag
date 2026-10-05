"""Services that never reached the riders: cancelled trips and skipped stops.

``docs/08`` §3.5's **label check** re-runs rule (b) counting these as late, on the
dates trip statuses were collected (from 2026-09-29, ``docs/01`` §5). The label
itself never counts them, because it must mean the same on every date.

**A cancelled trip is placed by its timetable.** TfNSW publishes a cancellation
with no stop updates, so the windows the trip should have served come from its
timetabled calls, in the bundle eras of its service date. A call counts as
unserved only when both hold:

- **the cancellation was in the feed at the call's time.** Cancellations are
  withdrawn: on 2026-09-29 four T4 trips were cancelled at dawn and running as
  replacements by the afternoon. A call before the cancellation was first seen,
  or after it was last seen (plus one poll, :data:`STATUS_HOLDS`), is not
  counted;
- **the trip was not observed in that window.** 26 of the 36 cancelled trips to
  2026-10-04 reported stop updates first. Where a trip was seen running, what was
  seen stands.

**A skipped stop is placed the same way.** The trip ran, but not for the riders
at that stop: the call it skipped counts as unserved in its window, under the
same time condition. The feed lists only the skips still ahead of the train, so
the store keeps every stop a trip was reported skipping that day.

Times are GTFS times, counted from noon less twelve hours of the service date
(:func:`~transit_rag.prediction.features.schedule.service_day_origin`).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from transit_rag.ingestion.alerts import DEFAULT_LINES
from transit_rag.prediction.collection.bundles import eras_for
from transit_rag.prediction.disruption.labels import (
    SERVICE_KEY,
    WINDOW,
    LabelRule,
    latest_by_service,
)
from transit_rag.prediction.features.schedule import ScheduleIndex, service_day_origin

#: Trip statuses are polled every two minutes, so a status last seen at *t* may
#: have held until just before the next poll.
STATUS_HOLDS = pd.Timedelta(minutes=2)

#: The first service date trip statuses were collected for (``docs/01`` §5).
STATUSES_FROM = "2026-09-29"

CANCELLED = "cancelled"
SKIPPED = "skipped stop"


def load_trip_statuses(db_path: Path, lines: Iterable[str] = DEFAULT_LINES) -> pd.DataFrame:
    """Cancelled trips and trips with skipped stops on ``lines``, read-only.

    One row per trip and status, with when the feed first and last carried it.
    """
    wanted = tuple(lines)
    placeholders = ",".join("?" for _ in wanted)
    uri = f"file:{db_path}?mode=ro"
    if not db_path.exists():
        raise FileNotFoundError(f"{db_path} does not exist")
    connection = sqlite3.connect(uri, uri=True)
    try:
        frame = pd.read_sql_query(
            f"""
            SELECT service_date, trip_id, trip_relationship, route_short_name AS line,
                   skipped_stop_ids, first_seen_utc, last_seen_utc
            FROM trip_statuses
            WHERE route_short_name IN ({placeholders})
              AND (trip_relationship = 'CANCELED' OR skipped_stop_ids != '')
            """,
            connection,
            params=wanted,
        )
    finally:
        connection.close()
    for column in ("first_seen_utc", "last_seen_utc"):
        frame[column] = pd.to_datetime(frame[column], utc=True, format="ISO8601")
    return frame


@dataclass(frozen=True)
class Placement:
    """Where each failed service should have been, and what could not be placed."""

    #: One row per service and window it failed to serve: ``SERVICE_KEY`` and ``why``.
    calls: pd.DataFrame
    #: Trips the timetable does not describe, or skipped stops it does not list.
    unplaced: list[tuple[str, str, str]] = field(default_factory=list)
    #: Distinct trips with each status, placed or not.
    considered: dict[str, int] = field(default_factory=dict)

    def trips(self, why: str) -> int:
        """How many distinct trips placed at least one call for this reason."""
        chosen = self.calls[self.calls["why"] == why]
        return int(chosen.loc[:, ["service_date", "trip_id"]].drop_duplicates().shape[0])


def unserved_calls(
    statuses: pd.DataFrame,
    events: pd.DataFrame,
    bundles: Sequence[Path],
    rule: LabelRule | None = None,
) -> Placement:
    """Each unserved call's line x window, by the rules in the module docstring."""
    rule = rule or LabelRule()
    columns = [*SERVICE_KEY, "why"]
    rows: list[tuple[str, pd.Timestamp, str, str, str]] = []
    unplaced: list[tuple[str, str, str]] = []
    whys = statuses["trip_relationship"].map(lambda r: CANCELLED if r == "CANCELED" else SKIPPED)
    considered = {
        why: int(statuses.loc[whys == why, ["service_date", "trip_id"]].drop_duplicates().shape[0])
        for why in (CANCELLED, SKIPPED)
    }
    for service_date, group in statuses.groupby("service_date", sort=True):
        date = str(service_date)
        schedule = ScheduleIndex.across_bundles(eras_for(date, bundles), group["trip_id"])
        origin = pd.Timestamp(service_day_origin(date))
        for status in group.itertuples(index=False):
            calls = schedule.stops_of(str(status.trip_id))
            why = CANCELLED if status.trip_relationship == "CANCELED" else SKIPPED
            if why == SKIPPED:
                skipped = set(str(status.skipped_stop_ids).split(","))
                calls = [(stop, scheduled) for stop, scheduled in calls if stop in skipped]
            if not calls:
                unplaced.append((date, str(status.trip_id), why))
                continue
            held_until = status.last_seen_utc + STATUS_HOLDS
            for _, scheduled in calls:
                seconds = scheduled.scheduled_arrival_s
                if seconds is None:
                    seconds = scheduled.scheduled_departure_s
                if seconds is None:
                    continue
                due = origin + pd.Timedelta(seconds=seconds)
                if status.first_seen_utc <= due <= held_until:
                    rows.append(
                        (str(status.line), due.floor(WINDOW), date, str(status.trip_id), why)
                    )
    calls = pd.DataFrame(rows, columns=columns)
    seen = latest_by_service(events, rule).loc[:, list(SERVICE_KEY)].drop_duplicates()
    calls["window_start_utc"] = calls["window_start_utc"].astype(seen["window_start_utc"].dtype)
    # A trip seen running in a window was not cancelled there, whatever it became.
    calls = calls.merge(seen, on=list(SERVICE_KEY), how="left", indicator=True)
    running = (calls["_merge"] == "both") & (calls["why"] == CANCELLED)
    calls = calls.loc[~running, columns].drop_duplicates(list(SERVICE_KEY))
    return Placement(calls=calls.reset_index(drop=True), unplaced=unplaced, considered=considered)


def compare_labels(
    labels: pd.DataFrame, checked: pd.DataFrame, since: str = STATUSES_FROM
) -> pd.DataFrame:
    """Each window from ``since``: its label and counts, and the same with the check.

    The check's columns carry a ``_check`` suffix. Before ``since`` no trip status
    was collected, so the check could not differ there and is not compared.
    """
    keys = ["line", "window_start_utc"]
    compared = labels.merge(
        checked.loc[:, [*keys, "n_services", "n_late", "rule_b", "disrupted"]],
        on=keys,
        suffixes=("", "_check"),
        validate="one_to_one",
    )
    return compared[compared["service_date"] >= since].reset_index(drop=True)
