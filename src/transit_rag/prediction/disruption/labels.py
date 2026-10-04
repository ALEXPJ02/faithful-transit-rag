"""Was a line disrupted in a 15-minute window? The ground truth for RQ2 detection.

``docs/08`` §3.2 defines it, decided 2026-09-28 and pending the supervisor's
confirmation. A line x window is disrupted if either rule holds:

(a) an *incident* names the line while the feed carries it. An incident is an
    alert the incident rule calls a disruption, with its republications merged
    (:mod:`transit_rag.ingestion.alerts`).
(b) at least a quarter of the line's services observed in the window are more
    than five minutes late.

**Rule (a) is timed by feed presence, not by ``active_period``.** No unplanned
alert has ever carried an end time. Timing by the publisher's claim labelled
every window from 2026-09-15 to 09-24 as disrupted.

**Rule (b) adapts TfNSW's on-time tolerance.** TfNSW judges a train once, at its
destination. A window has to be judged before most of its trains get there, so
a service counts as late in a window if the *latest* delay observed for it in
that window is more than 300 s.

Every threshold is a field of :class:`LabelRule`. A change the supervisor asks
for is then a re-run, and the thresholds are fixed before any test date is
scored.

**The population.** A window gets a label only when both rules could have
fired on it:

- *Alert collection was running.* Alerts were first collected on 2026-09-15.
  Before that, rule (a) cannot be known, so those windows are not built at all.
  A label that meant "rule (b) only" on the early dates would change meaning
  partway through the chronological split.
- *At least one service of the line was observed.* With no train, there is
  nothing to be late and nothing to predict.

Outside the population the label is ``pd.NA``, never ``False``.

**Rule (b) needs enough services to mean anything.** "A quarter of the line's
services" is degenerate on a thin window: at 02:00, one late train out of one
is 100%. :attr:`LabelRule.min_services` is the smallest count at which rule (b)
may fire. The default is five, the smallest count at which one late train
cannot reach a quarter by itself. This is an a priori bound and was not tuned to
the data. On the data to 2026-10-04, no minimum would add 28 rule (b) windows.
Twenty-six of them fall between 01:00 and 04:30 Sydney time. The other two are
T4 at 11:00 and 12:00 on 2026-09-22, during and just after the Bondi Junction
repairs, when T4 ran three and then four services.

Below the minimum, rule (b) is ``False``, but the window stays in the
population, so rule (a) can still mark it. An incident thins the service, and
the 11:00 window above is exactly the one a population filter would have
dropped. *Proposed 2026-10-05, pending confirmation alongside the rest of §3.2.*
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from transit_rag.ingestion.alerts import DEFAULT_LINES, Incident
from transit_rag.prediction.features.quality import (
    CLOSE_OBSERVATION_STOPS_AHEAD,
    MAX_PLAUSIBLE_DELAY_S,
)
from transit_rag.realtime.parsing import SERVICE_DAY_START_HOUR, SYDNEY

#: Agreed with the supervisor. Sydney's UTC offsets are whole hours, so
#: flooring a UTC instant to the quarter-hour gives the same boundaries as
#: flooring Sydney time, on both sides of a daylight-saving change.
WINDOW = pd.Timedelta(minutes=15)

#: "More than 5 minutes late": TfNSW's own on-time tolerance (Audit Office of
#: NSW, 2017). Strictly greater, so a train exactly 300 s late is on time.
LATE_THRESHOLD_S = 300

#: "At least a quarter of the line's services": a modelling choice, decided
#: with the definition.
LATE_SHARE = 0.25

#: The smallest window in which one late train cannot reach LATE_SHARE alone.
MIN_SERVICES = 5

#: The label table's columns, in order.
LABEL_COLUMNS: tuple[str, ...] = (
    "line",
    "window_start_utc",
    "window_start_local",
    "service_date",
    "n_services",
    "n_late",
    "share_late",
    "rule_a",
    "rule_b",
    "disrupted",
    "incident_ids",
)


@dataclass(frozen=True)
class LabelRule:
    """The thresholds of ``docs/08`` §3.2. The defaults are the decided values.

    The two quality bounds are the delay model's own
    (:mod:`transit_rag.prediction.features.quality`). A delay that the model is
    not allowed to learn from should not decide a label either.
    """

    late_threshold_s: int = LATE_THRESHOLD_S
    late_share: float = LATE_SHARE
    min_services: int = MIN_SERVICES
    max_stops_ahead: int = CLOSE_OBSERVATION_STOPS_AHEAD
    max_plausible_delay_s: int = MAX_PLAUSIBLE_DELAY_S

    def __post_init__(self) -> None:
        if not 0 < self.late_share <= 1:
            raise ValueError(f"late_share must be in (0, 1], got {self.late_share}")
        if self.min_services < 1:
            raise ValueError(f"min_services must be at least 1, got {self.min_services}")
        if self.late_threshold_s < 0:
            raise ValueError(f"late_threshold_s must not be negative, got {self.late_threshold_s}")

    def describe(self) -> str:
        return (
            f"an incident in the feed, or at least {self.late_share:.0%} of at least "
            f"{self.min_services} observed services more than {self.late_threshold_s} s late"
        )


@dataclass(frozen=True)
class Coverage:
    """The complete windows in which both feeds were being collected.

    A partial window at either end would be labelled from a fraction of its
    fifteen minutes. The first alert poll can land mid-window, and the
    snapshot is usually taken mid-window, so both ends round inwards.
    """

    start: pd.Timestamp
    end: pd.Timestamp
    #: The longest stretch with no successful alert poll. Rule (a) is blind
    #: during an outage, and the audit says so rather than assuming there was
    #: none.
    longest_alert_gap: pd.Timedelta

    @property
    def windows(self) -> int:
        return max(0, int((self.end - self.start) / WINDOW))


def _connect_read_only(db_path: Path) -> sqlite3.Connection:
    """Open a snapshot read-only: it is the only copy of data that cannot be re-collected."""
    if not db_path.exists():
        raise FileNotFoundError(f"{db_path} does not exist; pull a snapshot from the collector")
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)


def _utc(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, utc=True, format="ISO8601")


def load_stop_events(db_path: Path, lines: Iterable[str] = DEFAULT_LINES) -> pd.DataFrame:
    """Every collected stop event on ``lines``, one row per event.

    ``delay_s`` is the arrival delay, with the departure delay where a stop
    reports only that. ``reconcile.build_training_table`` makes the same
    choice, so the label and the delay model read the same number.
    ``observed_at`` is the last poll that named the event, which is just before
    the train reached the stop.
    """
    wanted = tuple(lines)
    placeholders = ",".join("?" for _ in wanted)
    connection = _connect_read_only(db_path)
    try:
        frame = pd.read_sql_query(
            f"""
            SELECT route_short_name AS line, service_date, trip_id, stop_id, stops_ahead,
                   COALESCE(arrival_delay_s, departure_delay_s) AS delay_s,
                   last_seen_utc AS observed_at
            FROM stop_observations
            WHERE route_short_name IN ({placeholders})
            """,
            connection,
            params=wanted,
        )
    finally:
        connection.close()
    frame["observed_at"] = _utc(frame["observed_at"])
    frame["delay_s"] = frame["delay_s"].astype("Float64")
    return frame


def load_coverage(db_path: Path) -> Coverage:
    """The complete windows from the first alert poll to the last trip-update poll."""
    connection = _connect_read_only(db_path)
    try:
        alert_polls = [
            row[0]
            for row in connection.execute(
                "SELECT poll_time_utc FROM alert_poll_log WHERE status = 'ok' ORDER BY 1"
            )
        ]
        last_poll = connection.execute(
            "SELECT MAX(poll_time_utc) FROM poll_log WHERE status = 'ok'"
        ).fetchone()[0]
    finally:
        connection.close()

    if not alert_polls or last_poll is None:
        raise ValueError(
            "the snapshot has no successful alert poll or no successful trip-update poll, "
            "so no window can be labelled"
        )
    polls = _utc(pd.Series(alert_polls))
    gaps = polls.diff().dropna()
    return Coverage(
        start=polls.iloc[0].ceil(WINDOW),
        end=pd.Timestamp(last_poll).tz_convert("UTC").floor(WINDOW),
        longest_alert_gap=gaps.max() if not gaps.empty else pd.Timedelta(0),
    )


def service_dates(window_starts: pd.Series) -> pd.Series:
    """The Sydney service date each window belongs to.

    This is the same rule as ``realtime.parsing.service_day``: the day turns
    over at 03:00 Sydney *wall-clock* time. Measured on the snapshot of
    2026-10-05, every stop event before 03:00 belongs to the previous service
    date, and none after it does. Subtracting three hours must happen on the
    wall clock. A tz-aware pandas subtraction is absolute, and on a
    daylight-saving day it would move 03:30 back across the boundary.
    """
    wall = window_starts.dt.tz_convert(SYDNEY).dt.tz_localize(None)
    return (wall - pd.Timedelta(hours=SERVICE_DAY_START_HOUR)).dt.strftime("%Y-%m-%d")


def window_grid(coverage: Coverage, lines: Iterable[str] = DEFAULT_LINES) -> pd.DataFrame:
    """Every line x window in the coverage, whether or not anything was observed."""
    starts = pd.date_range(coverage.start, coverage.end, freq=WINDOW, inclusive="left")
    return pd.DataFrame(
        [(line, start) for line in lines for start in starts],
        columns=["line", "window_start_utc"],
    )


def services_by_window(events: pd.DataFrame, rule: LabelRule) -> pd.DataFrame:
    """Rule (b)'s inputs: observed and late services per line x window.

    A service is one trip on one service date. Its state in a window is its
    **latest** observation there. A train 400 s late at 10:01 that recovers to
    200 s by 10:10 was not late in the 10:00 window by the time the window
    ended.
    """
    reliable = events[
        events["delay_s"].notna()
        & (events["stops_ahead"] <= rule.max_stops_ahead)
        & (events["delay_s"].abs() <= rule.max_plausible_delay_s)
    ]
    frame = reliable.assign(window_start_utc=reliable["observed_at"].dt.floor(WINDOW))
    latest = frame.sort_values("observed_at", kind="stable").drop_duplicates(
        ["line", "window_start_utc", "service_date", "trip_id"], keep="last"
    )
    latest = latest.assign(late=(latest["delay_s"] > rule.late_threshold_s).astype(int))
    counts = latest.groupby(["line", "window_start_utc"], as_index=False).agg(
        n_services=("trip_id", "size"), n_late=("late", "sum")
    )
    return counts


def late_needed(n_services: pd.Series, late_share: float) -> pd.Series:
    """How many late services put a window at ``late_share``: ``ceil(share x n)``.

    Counted rather than compared as a ratio. ``0.3 * 10`` is
    ``3.0000000000000004`` in floating point, so ``n_late >= share * n`` would
    let three late services out of ten miss a 30% threshold. The default of
    a quarter happens to be exact in binary; a threshold the supervisor might
    ask for need not be.
    """
    return np.ceil(late_share * n_services.astype(float) - 1e-9).astype(int)


@dataclass(frozen=True)
class IncidentSpan:
    """Where one incident puts rule (a) on one line."""

    incident_id: str
    line: str
    start: pd.Timestamp
    end: pd.Timestamp

    @property
    def window_starts(self) -> pd.DatetimeIndex:
        return pd.date_range(self.start.floor(WINDOW), self.end.floor(WINDOW), freq=WINDOW)


def incident_spans(
    incidents: Sequence[Incident], lines: Iterable[str] = DEFAULT_LINES
) -> list[IncidentSpan]:
    """Each incident's feed presence on each tracked line it names.

    The span is per line, not per incident. Edgecliff named T1 in only one
    alert, which the feed carried for a single poll, and named T4 for six and a
    half hours. Spanning the whole incident on every line it ever named would
    mark T1 disrupted for an afternoon on the strength of one poll. Within a
    line the span runs from the first alert to the last, so the gap a
    republication leaves between two alerts is still covered.
    """
    tracked = set(lines)
    spans: list[IncidentSpan] = []
    for incident in incidents:
        for line in incident.lines:
            if line not in tracked:
                continue
            naming = [alert for alert in incident.alerts if line in alert.lines]
            spans.append(
                IncidentSpan(
                    incident_id=incident.incident_id,
                    line=line,
                    start=pd.Timestamp(min(alert.first_seen for alert in naming)),
                    end=pd.Timestamp(max(alert.last_seen for alert in naming)),
                )
            )
    return spans


def label_windows(
    events: pd.DataFrame,
    incidents: Sequence[Incident],
    coverage: Coverage,
    rule: LabelRule | None = None,
    lines: Iterable[str] = DEFAULT_LINES,
) -> pd.DataFrame:
    """The label for every line x window in the coverage. See the module docstring."""
    rule = rule or LabelRule()
    tracked = tuple(lines)
    frame = window_grid(coverage, tracked).merge(
        services_by_window(events, rule), on=["line", "window_start_utc"], how="left"
    )
    frame["n_services"] = frame["n_services"].fillna(0).astype(int)
    frame["n_late"] = frame["n_late"].fillna(0).astype(int)
    observed = frame["n_services"] > 0
    frame["share_late"] = (frame["n_late"] / frame["n_services"]).where(observed)

    frame["rule_b"] = (frame["n_services"] >= rule.min_services) & (
        frame["n_late"] >= late_needed(frame["n_services"], rule.late_share)
    )

    flagged: dict[tuple[str, pd.Timestamp], set[str]] = {}
    for span in incident_spans(incidents, tracked):
        for start in span.window_starts:
            flagged.setdefault((span.line, start), set()).add(span.incident_id)
    keys = list(zip(frame["line"], frame["window_start_utc"], strict=True))
    frame["incident_ids"] = [",".join(sorted(flagged.get(key, ()))) for key in keys]
    frame["rule_a"] = frame["incident_ids"] != ""

    disrupted = (frame["rule_a"] | frame["rule_b"]).astype("boolean")
    frame["disrupted"] = disrupted.where(observed, pd.NA)

    frame["window_start_local"] = (
        frame["window_start_utc"].dt.tz_convert(SYDNEY).map(lambda t: t.isoformat())
    )
    frame["service_date"] = service_dates(frame["window_start_utc"])
    return frame.loc[:, list(LABEL_COLUMNS)]


def label_snapshot(
    db_path: Path,
    incidents: Sequence[Incident],
    rule: LabelRule | None = None,
    lines: Iterable[str] = DEFAULT_LINES,
) -> tuple[pd.DataFrame, Coverage]:
    """Load a snapshot and label it. The incidents come from the same snapshot."""
    tracked = tuple(lines)
    coverage = load_coverage(db_path)
    events = load_stop_events(db_path, tracked)
    return label_windows(events, incidents, coverage, rule, tracked), coverage
