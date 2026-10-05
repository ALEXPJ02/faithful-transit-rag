"""What the reasons stage may know about a disruption: the feed, never the alert.

RQ1 Objective 3 explains a disruption from evidence. The incident's own alert
carries its cause, and that cause is the answer being scored (``docs/08``
§3.5). So the situation the reasons stage is given is built from the delay feed
alone: which line, when, how many services were late in the half hour before,
and where. Retrieval and the model both read :meth:`Situation.describe`, so
neither can see more than this.

Where delays were is said in station names, because that is how past alerts
say it ("urgent train repairs at Chatswood") and how a rider would ask.
"""

from __future__ import annotations

import csv
import io
import zipfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from transit_rag.prediction.disruption.labels import LabelRule
from transit_rag.realtime.parsing import SYDNEY

#: How far back the situation looks: one prediction horizon (``docs/08`` §3.1).
DEFAULT_LOOKBACK = timedelta(minutes=30)

#: How many stations the description names. Enough to place a disruption,
#: few enough that one busy interchange does not drown the rest.
TOP_STATIONS = 5


def station_names(bundle: Path) -> dict[str, str]:
    """Each stop id's station name, from a static bundle's ``stops.txt``.

    Platforms roll up to their parent: ``Central Station Platform 16`` is
    ``Central``. A stop with no parent keeps its own name, less the
    ``Station`` suffix.
    """
    with zipfile.ZipFile(bundle) as archive, archive.open("stops.txt") as raw:
        rows = list(csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig")))
    by_id = {row["stop_id"]: row for row in rows}

    def name(row: Mapping[str, str]) -> str:
        parent = by_id.get(row.get("parent_station") or "")
        text = (parent or row)["stop_name"]
        return text.removesuffix(" Station").strip()

    return {stop_id: name(row) for stop_id, row in by_id.items()}


@dataclass(frozen=True)
class Situation:
    """The delay feed on some lines in the minutes before a moment."""

    lines: tuple[str, ...]
    at: datetime
    lookback: timedelta
    services: int
    late: int
    max_delay_s: float | None
    #: (station, services more than five minutes late there), most first.
    delayed_stations: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        if self.at.tzinfo is None:
            raise ValueError("a situation's time must be timezone-aware")

    def describe(self) -> str:
        """The text retrieval embeds and the model reads. Nothing else is shown."""
        local = self.at.astimezone(SYDNEY)
        when = f"{local:%H:%M} on {local:%A} {local.day} {local:%B %Y} (Sydney time)"
        minutes = int(self.lookback.total_seconds() // 60)
        lines = " and ".join(self.lines)
        parts = [f"{lines}, {when}."]
        if self.services == 0:
            parts.append(f"No {lines} service was observed in the {minutes} minutes before.")
            return " ".join(parts)
        parts.append(
            f"In the {minutes} minutes before, {self.services} services were observed and "
            f"{self.late} were more than five minutes late."
        )
        if self.max_delay_s is not None and self.max_delay_s > 0:
            parts.append(
                f"The most delayed was {round(self.max_delay_s / 60)} minutes behind timetable."
            )
        if self.delayed_stations:
            named = ", ".join(f"{station} ({count})" for station, count in self.delayed_stations)
            parts.append(f"Late services were seen at {named}.")
        return " ".join(parts)


def situation_at(
    events: pd.DataFrame,
    lines: Iterable[str],
    at: datetime,
    stations: Mapping[str, str],
    *,
    lookback: timedelta = DEFAULT_LOOKBACK,
    rule: LabelRule | None = None,
) -> Situation:
    """What the feed showed on ``lines`` in the ``lookback`` before ``at``.

    ``events`` is :func:`~transit_rag.prediction.disruption.labels.load_stop_events`'
    output. The same reliability and plausibility bounds as the label apply, so
    the reasons stage never reads a delay the label would not trust. A
    service's state is its latest observation in the lookback, as in rule (b).
    """
    rule = rule or LabelRule()
    tracked = tuple(lines)
    start, end = pd.Timestamp(at - lookback), pd.Timestamp(at)
    frame = events[
        events["line"].isin(tracked)
        & (events["observed_at"] >= start)
        & (events["observed_at"] < end)
        & events["delay_s"].notna()
        & (events["stops_ahead"] <= rule.max_stops_ahead)
        & (events["delay_s"].abs() <= rule.max_plausible_delay_s)
    ]
    service = ["service_date", "trip_id"]
    latest = frame.sort_values("observed_at", kind="stable").drop_duplicates(service, keep="last")
    late_rows = frame[frame["delay_s"] > rule.late_threshold_s]
    by_station = (
        late_rows.assign(station=late_rows["stop_id"].map(stations).fillna(late_rows["stop_id"]))
        .drop_duplicates(["station", *service])
        .groupby("station")
        .size()
        .sort_values(ascending=False, kind="stable")
    )
    return Situation(
        lines=tracked,
        at=at,
        lookback=lookback,
        services=len(latest),
        late=int((latest["delay_s"] > rule.late_threshold_s).sum()),
        # From each service's latest observation, like ``late``: a train that has
        # recovered is not still far behind (found by Cursor Bugbot on #21).
        max_delay_s=float(latest["delay_s"].max()) if len(latest) else None,
        delayed_stations=tuple(
            (str(station), int(count)) for station, count in by_station.head(TOP_STATIONS).items()
        ),
    )
