"""The feed as it stood at a moment, read from a frozen snapshot.

``docs/08`` §3.5: live-condition questions are scored against frozen snapshots,
never the live API, so a re-run a week later reproduces the numbers. Every tool
the orchestrator calls reads the world through a :class:`SnapshotFeed` and a
moment. Nothing the snapshot recorded after that moment is visible.

What "in the feed at T" means for an alert is the causal rule the detector uses
(``prediction/disruption/features.py``): the alert was in the last alert poll at
or before *T*. One residual is stated, not fixed (``docs/10`` §5): the snapshot
keeps an alert's *final* text, so an alert republished with new wording after *T*
shows that wording. The texts are short, and mostly written at first posting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

from transit_rag.agent.situation import Situation, situation_at, station_names
from transit_rag.ingestion.alerts import AlertRecord, load_alerts
from transit_rag.prediction.disruption.labels import load_alert_polls, load_stop_events


@dataclass
class SnapshotFeed:
    """A snapshot's stop events, alerts and alert polls, queried as of a moment."""

    db_path: Path
    events: pd.DataFrame
    alerts: list[AlertRecord]
    alert_polls: pd.Series
    stations: dict[str, str] = field(default_factory=dict)

    @classmethod
    def open(cls, db_path: Path, bundle: Path) -> SnapshotFeed:
        """Read a snapshot (read-only) and the station names from a static bundle."""
        return cls(
            db_path=db_path,
            events=load_stop_events(db_path),
            alerts=load_alerts(db_path),
            alert_polls=load_alert_polls(db_path),
            stations=station_names(bundle),
        )

    def last_alert_poll(self, at: datetime) -> pd.Timestamp | None:
        """The last successful alert poll at or before ``at``."""
        polls = self.alert_polls[self.alert_polls <= pd.Timestamp(at)]
        return None if polls.empty else polls.max()

    def alerts_at(self, at: datetime, line: str) -> list[AlertRecord]:
        """Alerts naming ``line`` that the last alert poll at or before ``at`` carried."""
        poll = self.last_alert_poll(at)
        if poll is None:
            return []
        return [
            alert
            for alert in self.alerts
            if line in alert.lines and alert.first_seen <= poll <= alert.last_seen
        ]

    def situation(self, line: str, at: datetime) -> Situation:
        """What the trip-update feed showed on ``line`` in the half hour before ``at``."""
        return situation_at(self.events, (line,), at, self.stations)
