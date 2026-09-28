"""Collected service alerts into incidents: which alerts are disruptions, and
which alerts are the same disruption.

Both questions sit under the disruption label (``docs/08`` §3.2 rule (a)) and
the reasons corpus (RQ1 Objective 3), so they are answered once, here.

**Which alerts are disruptions.** Cause is not enough. Measured on the 22
unplanned T1/T4 alerts to 2026-09-28, "any cause other than MAINTENANCE"
admits a 1:47am cancellation "due to trackwork" and a trackwork bus stand
notice, both published as ``UNKNOWN_CAUSE``, and a police operation that closed
*streets*; and ``UNKNOWN_CAUSE`` also carries the Edgecliff closure, a genuine
incident. The header does not settle it either ("Station Update - ..." on
both kinds). The description does, so the rule reads it: planned-work wording
excludes, service-impact wording includes, anything else is a notice.

The rule was fitted by reading those 22 alerts. Alerts first seen from
:data:`RULE_FROZEN_ON` were never read while writing it, so they are the check
on it. A case it gets wrong is corrected in ``data/alert_overrides.csv`` with a
written reason, never by quietly editing the markers.

**Which alerts are the same disruption.** TfNSW republishes an incident under a
new ``entity.id`` whenever its scope or text changes -- the id is a
content-derived UUID -- and the old one leaves the feed. Counted as separate
incidents, twins inflate the sample and, worse, leak: explaining the later one,
the earlier one passes a "seen before" filter carrying the same cause.
Republications show up as a chain in which each alert appears within one alert
poll of the previous one disappearing, on a shared line. The cause may change
along the chain -- Edgecliff went from ``UNKNOWN_CAUSE`` to ``POLICE_ACTIVITY``
-- so the cause must match *or* one side must be unknown.
"""

from __future__ import annotations

import csv
import hashlib
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from transit_rag.ingestion.chunks import UnattributableChunkError

SYDNEY = ZoneInfo("Australia/Sydney")

DEFAULT_LINES: tuple[str, ...] = ("T1", "T4")
DEFAULT_OVERRIDES = Path("data/alert_overrides.csv")

#: Longer than this in the feed is a standing notice, not an incident
#: (``docs/08`` §3.2). Incidents so far run 0-243 minutes; the shortest
#: standing notice runs 1,969.
MAX_INCIDENT_PRESENCE = timedelta(hours=24)

#: One alert poll (15 x 120 s) plus slack for poll jitter. Republications in
#: the data arrive 30-31 minutes after their predecessor was last seen.
MAX_REPUBLISH_GAP = timedelta(minutes=35)

#: Alerts first seen on or after this date were not read when the markers
#: below were chosen, so they are where the rule gets checked.
RULE_FROZEN_ON = "2026-09-29"

#: Wording that marks scheduled work. Deliberately not "replacement bus": an
#: unplanned incident can say "buses are replacing trains" too.
PLANNED_MARKERS: tuple[str, ...] = ("trackwork", "planned work", "installation works")

#: Wording that says train running is affected.
IMPACT_MARKERS: tuple[str, ...] = (
    "extra travel",
    "not running",
    "no trains",
    "delay",
    "running late",
    "run late",
    "resumed",
    "suspended",
    "cancelled",
    "reduced service",
    "fewer services",
    "buses replace",
    "buses are replacing",
)

UNKNOWN_CAUSE = "UNKNOWN_CAUSE"

#: ``docs/08`` §3.2's four groups. MAINTENANCE is absent on purpose: planned
#: work is excluded before grouping, never classified.
CAUSE_GROUPS: dict[str, str] = {
    "TECHNICAL_PROBLEM": "technical",
    "ACCIDENT": "network_incident",
    "POLICE_ACTIVITY": "network_incident",
    "MEDICAL_EMERGENCY": "network_incident",
    "WEATHER": "weather_external",
    "STRIKE": "weather_external",
    "DEMONSTRATION": "weather_external",
    "CONSTRUCTION": "weather_external",
    "HOLIDAY": "weather_external",
    "OTHER_CAUSE": "other_unknown",
    UNKNOWN_CAUSE: "other_unknown",
}


@dataclass(frozen=True)
class AlertRecord:
    """One collected alert, reduced to the lines this project tracks."""

    alert_id: str
    cause: str
    effect: str
    header_text: str
    description_text: str
    first_seen: datetime
    last_seen: datetime
    lines: tuple[str, ...]

    @property
    def presence(self) -> timedelta:
        """How long the feed carried it, at the 30-minute poll resolution."""
        return self.last_seen - self.first_seen

    @property
    def short_id(self) -> str:
        return self.alert_id[:8]


@dataclass(frozen=True)
class Verdict:
    """Whether an alert is a disruption, and the rule that decided it."""

    is_incident: bool
    reason: str


@dataclass(frozen=True)
class Override:
    is_incident: bool
    reason: str


@dataclass(frozen=True)
class Incident:
    """One disruption, however many times TfNSW republished it."""

    alerts: tuple[AlertRecord, ...]
    incident_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.alerts:
            raise ValueError("an incident needs at least one alert")
        # Stable across rebuilds for the same membership, and independent of
        # the order the alerts were passed in.
        members = ",".join(sorted(alert.alert_id for alert in self.alerts))
        digest = hashlib.sha256(members.encode("utf-8")).hexdigest()[:10]
        object.__setattr__(self, "incident_id", f"inc-{digest}")

    @property
    def cause(self) -> str:
        """The first cause any of its alerts names, in publication order.

        Unknown only when none ever names one. Edgecliff was first posted as
        unknown and then as police activity; the correction is the answer.
        """
        for alert in sorted(self.alerts, key=lambda a: a.first_seen):
            if alert.cause != UNKNOWN_CAUSE:
                return alert.cause
        return UNKNOWN_CAUSE

    @property
    def cause_group(self) -> str:
        return CAUSE_GROUPS.get(self.cause, "other_unknown")

    @property
    def first_seen(self) -> datetime:
        return min(alert.first_seen for alert in self.alerts)

    @property
    def last_seen(self) -> datetime:
        return max(alert.last_seen for alert in self.alerts)

    @property
    def lines(self) -> tuple[str, ...]:
        return tuple(sorted({line for alert in self.alerts for line in alert.lines}))

    @property
    def alert_ids(self) -> tuple[str, ...]:
        return tuple(a.alert_id for a in sorted(self.alerts, key=lambda a: a.first_seen))


def _parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value)


def load_alerts(db_path: Path, lines: Iterable[str] = DEFAULT_LINES) -> list[AlertRecord]:
    """Every collected alert that names one of ``lines``, oldest first.

    The database is opened **read-only**. These snapshots are the only copy of
    data that cannot be re-collected, and nothing in ingestion has any business
    writing to them.
    """
    if not db_path.exists():
        raise FileNotFoundError(f"{db_path} does not exist; pull a snapshot from the collector")
    wanted = tuple(lines)
    placeholders = ",".join("?" for _ in wanted)
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            f"""
            SELECT a.alert_id, a.cause, a.effect, a.header_text, a.description_text,
                   a.first_seen_utc, a.last_seen_utc,
                   GROUP_CONCAT(DISTINCT s.route_short_name)
            FROM service_alerts AS a
            JOIN alert_scopes AS s ON s.alert_id = a.alert_id
            WHERE s.route_short_name IN ({placeholders})
            GROUP BY a.alert_id
            ORDER BY a.first_seen_utc, a.alert_id
            """,
            wanted,
        ).fetchall()
    finally:
        connection.close()

    return [
        AlertRecord(
            alert_id=str(alert_id),
            cause=str(cause or UNKNOWN_CAUSE),
            effect=str(effect or ""),
            header_text=str(header or ""),
            description_text=str(description or ""),
            first_seen=_parse_utc(first_seen),
            last_seen=_parse_utc(last_seen),
            lines=tuple(sorted(str(line_names).split(","))),
        )
        for (
            alert_id,
            cause,
            effect,
            header,
            description,
            first_seen,
            last_seen,
            line_names,
        ) in rows
    ]


def load_overrides(path: Path = DEFAULT_OVERRIDES) -> dict[str, Override]:
    """Manual corrections to the rule, keyed by alert id.

    A missing file means no corrections. A malformed row is an error rather
    than a skipped line: an override that silently fails to apply is a ground
    truth nobody decided.
    """
    if not path.exists():
        return {}
    overrides: dict[str, Override] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for number, row in enumerate(csv.DictReader(handle), start=2):
            decision = (row.get("is_incident") or "").strip().lower()
            reason = (row.get("reason") or "").strip()
            alert_id = (row.get("alert_id") or "").strip()
            if decision not in {"yes", "no"} or not alert_id or not reason:
                raise ValueError(
                    f"{path}:{number}: each override needs alert_id, is_incident "
                    f"(yes/no) and a reason; got {row}"
                )
            overrides[alert_id] = Override(is_incident=decision == "yes", reason=reason)
    return overrides


def _first_marker(text: str, markers: Sequence[str]) -> str | None:
    for marker in markers:
        if marker in text:
            return marker
    return None


def classify(alert: AlertRecord, override: Override | None = None) -> Verdict:
    """Is this alert a disruption? In order: override, planned cause, standing
    notice, planned-work wording, service-impact wording, otherwise a notice."""
    if override is not None:
        return Verdict(override.is_incident, f"override: {override.reason}")
    if alert.cause == "MAINTENANCE":
        return Verdict(False, "planned maintenance")
    if alert.presence > MAX_INCIDENT_PRESENCE:
        hours = alert.presence.total_seconds() / 3600
        return Verdict(False, f"standing notice ({hours:.0f} h in the feed)")

    text = f"{alert.header_text}\n{alert.description_text}".lower()
    planned = _first_marker(text, PLANNED_MARKERS)
    if planned:
        return Verdict(False, f"planned work: {planned!r}")
    impact = _first_marker(text, IMPACT_MARKERS)
    if impact:
        return Verdict(True, f"service impact: {impact!r}")
    return Verdict(False, "no effect on train running stated")


def _same_disruption(earlier: AlertRecord, later: AlertRecord) -> bool:
    """Whether ``later`` is plausibly a republication of ``earlier``."""
    if not set(earlier.lines) & set(later.lines):
        return False
    gap = later.first_seen - earlier.last_seen
    if gap > MAX_REPUBLISH_GAP:
        return False
    return earlier.cause == later.cause or UNKNOWN_CAUSE in (earlier.cause, later.cause)


def group_incidents(alerts: Sequence[AlertRecord]) -> list[Incident]:
    """Merge republication chains into incidents, oldest first.

    Transitive by construction (union-find), because a chain can be longer
    than two: Chatswood on 2026-09-25 is three alerts, each first seen on the
    poll after the previous one left the feed.
    """
    ordered = sorted(alerts, key=lambda a: (a.first_seen, a.alert_id))
    parent = list(range(len(ordered)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for i, earlier in enumerate(ordered):
        for j in range(i + 1, len(ordered)):
            later = ordered[j]
            if later.first_seen - earlier.last_seen > MAX_REPUBLISH_GAP:
                continue
            if _same_disruption(earlier, later):
                parent[find(j)] = find(i)

    groups: dict[int, list[AlertRecord]] = {}
    for index, alert in enumerate(ordered):
        groups.setdefault(find(index), []).append(alert)
    incidents = [Incident(alerts=tuple(members)) for members in groups.values()]
    return sorted(incidents, key=lambda incident: incident.first_seen)


@dataclass(frozen=True)
class AuditRow:
    """One alert's line in the audit: what was decided, why, and where it went."""

    alert: AlertRecord
    verdict: Verdict
    incident_id: str | None
    held_out: bool


def audit(
    alerts: Sequence[AlertRecord],
    overrides: dict[str, Override] | None = None,
) -> tuple[list[AuditRow], list[Incident]]:
    """Classify every alert, group the incidents, and say which rows check the rule."""
    overrides = overrides or {}
    verdicts = {alert.alert_id: classify(alert, overrides.get(alert.alert_id)) for alert in alerts}
    incidents = group_incidents([a for a in alerts if verdicts[a.alert_id].is_incident])
    membership = {
        alert_id: incident.incident_id for incident in incidents for alert_id in incident.alert_ids
    }
    rows = [
        AuditRow(
            alert=alert,
            verdict=verdicts[alert.alert_id],
            incident_id=membership.get(alert.alert_id),
            held_out=sydney_date(alert.first_seen) >= RULE_FROZEN_ON,
        )
        for alert in alerts
    ]
    return rows, incidents


def sydney_date(instant: datetime) -> str:
    return instant.astimezone(SYDNEY).date().isoformat()


def sydney_time(instant: datetime) -> str:
    return instant.astimezone(SYDNEY).strftime("%Y-%m-%d %H:%M")


#: How incident passages identify themselves in the index, beside the PDFs'
#: document keys and titles.
ALERT_DOCUMENT_KEY = "tfnsw_alerts"
ALERT_DOCUMENT_TITLE = "TfNSW service alerts"


@dataclass(frozen=True)
class IncidentPassage:
    """One incident as a retrievable passage, citable by the alerts behind it.

    One passage per *incident*, not per alert, so a republished disruption
    cannot fill two of the five retrieved slots, and excluding it from its own
    explanation is one id rather than a hunt for its twins.

    The ``Chunk`` contract carries over: a passage that cannot be cited does
    not exist. An alert has no page, so the locator is the alert ids.
    """

    incident_id: str
    alert_ids: tuple[str, ...]
    lines: tuple[str, ...]
    cause: str
    cause_group: str
    first_seen: datetime
    last_seen: datetime
    text: str

    def __post_init__(self) -> None:
        if not self.incident_id or not self.alert_ids:
            raise UnattributableChunkError(
                f"incident passage {self.incident_id!r} names no alerts to cite"
            )
        if not self.text.strip():
            raise UnattributableChunkError(f"incident passage {self.incident_id!r} is empty")
        if self.first_seen.tzinfo is None:
            raise UnattributableChunkError(
                f"incident passage {self.incident_id!r} has a naive first-seen time; the "
                f"leakage filter compares instants and cannot use it"
            )

    @property
    def chunk_id(self) -> str:
        return self.incident_id

    @property
    def locator(self) -> str:
        return "alerts " + ", ".join(alert_id[:8] for alert_id in self.alert_ids)

    @property
    def citation(self) -> str:
        noun = "alert" if len(self.alert_ids) == 1 else "alerts"
        ids = ", ".join(alert_id[:8] for alert_id in self.alert_ids)
        return f"TfNSW {noun} {ids} (first seen {sydney_time(self.first_seen)} Sydney time)"

    def metadata(self) -> dict[str, Any]:
        """Vector-store metadata: the citation, and what the leakage filter reads."""
        return {
            "document_key": ALERT_DOCUMENT_KEY,
            "document_title": ALERT_DOCUMENT_TITLE,
            "locator": self.locator,
            "citation": self.citation,
            "incident_id": self.incident_id,
            "alert_ids": ",".join(self.alert_ids),
            "lines": ",".join(self.lines),
            "cause": self.cause,
            "cause_group": self.cause_group,
            "first_seen_ts": int(self.first_seen.timestamp()),
            "last_seen_ts": int(self.last_seen.timestamp()),
        }


def _passage_text(incident: Incident) -> str:
    """What gets embedded: when, where, the stated cause, and TfNSW's words.

    Repeated wording across a republication chain is kept once -- Martin
    Place's two alerts carry byte-identical descriptions -- so a chain does not
    outweigh a single alert just by repeating itself.
    """
    lines = " and ".join(incident.lines)
    cause = incident.cause.replace("_", " ").lower()
    parts = [f"{lines}, first seen {sydney_time(incident.first_seen)}. Cause: {cause}."]
    seen: set[str] = set()
    for alert in sorted(incident.alerts, key=lambda a: a.first_seen):
        for piece in (alert.header_text.strip(), alert.description_text.strip()):
            if piece and piece not in seen:
                seen.add(piece)
                parts.append(piece)
    return "\n".join(parts)


def incident_passages(incidents: Sequence[Incident]) -> list[IncidentPassage]:
    return [
        IncidentPassage(
            incident_id=incident.incident_id,
            alert_ids=incident.alert_ids,
            lines=incident.lines,
            cause=incident.cause,
            cause_group=incident.cause_group,
            first_seen=incident.first_seen,
            last_seen=incident.last_seen,
            text=_passage_text(incident),
        )
        for incident in incidents
    ]


def incident_corpus_hash(passages: Sequence[IncidentPassage]) -> str:
    """One hash for "exactly these incident passages".

    Covers the text as well as the ids, so a changed rule, override or
    snapshot that alters any passage reads as a different corpus -- the alert
    counterpart of the PDF content pins.
    """
    material = "\n".join(
        sorted(
            f"{passage.incident_id}:{hashlib.sha256(passage.text.encode('utf-8')).hexdigest()}"
            for passage in passages
        )
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()
