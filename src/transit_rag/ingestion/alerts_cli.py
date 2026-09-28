"""``transit-alerts`` -- which collected alerts are disruptions, and which are one.

    transit-alerts audit --db data/delay_observations_20260929.db
    transit-alerts audit --db ... --held-out    # only the alerts that check the rule

Prints one line per T1/T4 alert -- the verdict, the rule that decided it, and
the incident it joined -- then the incidents themselves. The point is that
every exclusion is visible and has a reason, so the ground truth under the
disruption label and the reasons corpus can be read and argued with rather
than trusted (``docs/08`` §3.2). Reads the snapshot; writes nothing.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from transit_rag.ingestion.alerts import (
    CAUSE_GROUPS,
    DEFAULT_OVERRIDES,
    RULE_FROZEN_ON,
    AuditRow,
    Incident,
    audit,
    load_alerts,
    load_overrides,
    sydney_date,
    sydney_time,
)


def _alert_lines(rows: list[AuditRow]) -> list[str]:
    lines = [
        f"  {'first seen (Sydney)':<19}  {'id':<8}  {'mins':>6}  {'cause':<17}  "
        f"{'lines':<5}  {'verdict':<8}  {'reason':<38}  incident"
    ]
    for row in rows:
        alert = row.alert
        minutes = round(alert.presence.total_seconds() / 60)
        verdict = "incident" if row.verdict.is_incident else "excluded"
        marker = " *" if row.held_out else ""
        lines.append(
            f"  {sydney_time(alert.first_seen):<19}  {alert.short_id:<8}  {minutes:>6}  "
            f"{alert.cause[:17]:<17}  {','.join(alert.lines):<5}  {verdict:<8}  "
            f"{row.verdict.reason[:38]:<38}  {row.incident_id or '—'}{marker}"
        )
    return lines


def _incident_lines(incidents: list[Incident]) -> list[str]:
    lines = []
    for incident in incidents:
        first = min(incident.alerts, key=lambda a: a.first_seen)
        count = len(incident.alerts)
        lines.append(
            f"  {incident.incident_id}  {sydney_time(incident.first_seen)}  "
            f"{','.join(incident.lines):<5}  {incident.cause} -> {incident.cause_group}  "
            f"({count} alert{'' if count == 1 else 's'})  {first.header_text[:40]!r}"
        )
    return lines


def command_audit(args: argparse.Namespace) -> int:
    alerts = load_alerts(args.db)
    overrides = load_overrides(args.overrides)
    rows, incidents = audit(alerts, overrides)

    unknown = sorted(set(overrides) - {alert.alert_id for alert in alerts})
    held_out = [row for row in rows if row.held_out]
    shown = held_out if args.held_out else rows

    print(f"Alerts naming T1 or T4 in {args.db}: {len(rows)}")
    print(
        f"  {len(held_out)} first seen from {RULE_FROZEN_ON} (marked *): the rule was "
        "not written from these, so they are its check"
    )
    if overrides:
        print(f"  {len(overrides)} override(s) from {args.overrides}")
    for alert_id in unknown:
        print(f"  warning: override for {alert_id} names no alert in this snapshot")
    print()
    print("\n".join(_alert_lines(shown)))

    dates = {sydney_date(incident.first_seen) for incident in incidents}
    print(f"\nIncidents: {len(incidents)} on {len(dates)} Sydney date(s)")
    print("\n".join(_incident_lines(incidents)))

    groups = Counter(incident.cause_group for incident in incidents)
    every_group = sorted(set(CAUSE_GROUPS.values()))
    print("\nCause groups: " + ", ".join(f"{g} {groups.get(g, 0)}" for g in every_group))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="transit-alerts", description=__doc__.split("\n")[0])
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit_parser = subparsers.add_parser("audit", help="classify and group the collected alerts")
    audit_parser.add_argument(
        "--db", type=Path, required=True, help="a collection snapshot (read-only)"
    )
    audit_parser.add_argument("--overrides", type=Path, default=DEFAULT_OVERRIDES)
    audit_parser.add_argument(
        "--held-out",
        action="store_true",
        help=f"show only alerts first seen from {RULE_FROZEN_ON}, which check the rule",
    )
    audit_parser.set_defaults(handler=command_audit)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result: int = args.handler(args)
        return result
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
