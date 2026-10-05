"""Lead time: how much earlier than the operator a detector raised the alarm (``docs/08`` §3.4).

For each incident on each tracked line it names, take the first window end at
which the detector flagged that line. The search runs from ``lookback`` before the
incident's first alert on the line to the end of the window holding its last
sighting there. Lead time is the alert's first sighting minus that flag, in
minutes, so **positive
means the detector was earlier**. An incident with no flag in that span is missed.

**The lookback is 60 minutes, fixed a priori.** The detector's claim reaches 30
minutes ahead, and the alert's own time is uncertain by one 30-minute alert poll.
A flag earlier than both together is not this incident's warning. It is a
parameter, like every threshold here.

**Resolution.** An alert's first sighting is known only to within one alert poll,
so a lead time is good to about 30 minutes. The median is the headline, and no
lead time is quoted finer than the poll allows.

**Only the operator's incidents have a lead time.** Rule (b)'s lateness has no
alert to be early against, so a detector is also, and separately, scored on the
targets it catches (``detection.py``).
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd

from transit_rag.ingestion.alerts import DEFAULT_LINES, Incident
from transit_rag.prediction.disruption.labels import WINDOW

LOOKBACK = timedelta(minutes=60)


@dataclass(frozen=True)
class IncidentLead:
    """One incident on one line: when the operator said so, and when the detector did."""

    incident_id: str
    line: str
    alert_first_seen: datetime
    first_flag: datetime | None
    #: Alert time minus flag time. Positive is earlier than the operator. None if missed.
    lead_minutes: float | None


def lead_times(
    table: pd.DataFrame,
    flagged: pd.Series,
    incidents: Sequence[Incident],
    *,
    lookback: timedelta = LOOKBACK,
    lines: Iterable[str] = DEFAULT_LINES,
) -> list[IncidentLead]:
    """Lead time for every incident whose first alert falls within ``table``'s windows.

    ``table`` holds detection rows (``line``, ``window_start_utc``), and ``flagged``
    says, row by row, whether the detector flagged it. A row's flag is made at its
    window's end, which is the moment its prediction is about.
    """
    if table.empty:
        return []
    ends = table["window_start_utc"] + WINDOW
    first_window, last_end = table["window_start_utc"].min(), ends.max()
    tracked = set(lines)
    leads = []
    for incident in incidents:
        for line in incident.lines:
            if line not in tracked:
                continue
            naming = [alert for alert in incident.alerts if line in alert.lines]
            alert_time = min(alert.first_seen for alert in naming)
            last_seen = max(alert.last_seen for alert in naming)
            if not first_window <= pd.Timestamp(alert_time) < last_end:
                continue
            candidates = (
                (table["line"] == line)
                & flagged.astype(bool)
                & (ends >= pd.Timestamp(alert_time - lookback))
                & (ends <= pd.Timestamp(last_seen) + WINDOW)
            )
            if not candidates.any():
                leads.append(IncidentLead(incident.incident_id, line, alert_time, None, None))
                continue
            flag = ends[candidates].min().to_pydatetime()
            minutes = (alert_time - flag).total_seconds() / 60
            leads.append(IncidentLead(incident.incident_id, line, alert_time, flag, minutes))
    return leads


@dataclass(frozen=True)
class LeadSummary:
    """Lead times over the incidents in the rows scored."""

    incidents: int
    detected: int
    before_alert: int
    median_lead_minutes: float | None

    def describe(self) -> str:
        if not self.incidents:
            return "no operator incident in these rows"
        median = (
            "n/a" if self.median_lead_minutes is None else f"{self.median_lead_minutes:+.0f} min"
        )
        return (
            f"{self.detected}/{self.incidents} incident(s) flagged, {self.before_alert} before "
            f"the operator's alert; median lead {median} (good to ~30 min)"
        )


def summarise(leads: Sequence[IncidentLead]) -> LeadSummary:
    found = [lead.lead_minutes for lead in leads if lead.lead_minutes is not None]
    return LeadSummary(
        incidents=len(leads),
        detected=len(found),
        before_alert=sum(minutes > 0 for minutes in found),
        median_lead_minutes=statistics.median(found) if found else None,
    )
