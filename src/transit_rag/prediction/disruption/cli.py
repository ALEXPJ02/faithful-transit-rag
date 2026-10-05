"""``transit-label`` -- label every T1/T4 15-minute window disrupted or not.

    transit-label --db data/delay_observations_20261005.db               # audit; writes nothing
    transit-label --db ... --out data/disruption_labels.csv             # also write the labels
    transit-label --db ... --min-services 1                             # the rule with no minimum
    transit-label --db ... --cancellation-check                         # docs/08 §3.5's label check

Prints how many windows each rule marks, where they agree, and which incidents
are behind rule (a). The ground truth RQ2 scores against can then be read and
argued with, rather than trusted (``docs/08`` §3.2, ``docs/11``). It reads the
snapshot and writes only to ``--out``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from transit_rag.ingestion.alerts import (
    DEFAULT_OVERRIDES,
    Incident,
    audit,
    load_alerts,
    load_overrides,
    sydney_time,
)
from transit_rag.prediction.collection.bundles import discover
from transit_rag.prediction.disruption.labels import (
    LATE_SHARE,
    LATE_THRESHOLD_S,
    MIN_SERVICES,
    WINDOW,
    Coverage,
    LabelRule,
    incident_spans,
    label_windows,
    late_needed,
    load_coverage,
    load_stop_events,
)
from transit_rag.prediction.disruption.unserved import (
    CANCELLED,
    SKIPPED,
    STATUSES_FROM,
    Placement,
    compare_labels,
    load_trip_statuses,
    unserved_calls,
)
from transit_rag.realtime.parsing import SYDNEY

#: Longer than two alert polls without a successful one, and rule (a) was
#: blind long enough to miss an incident outright.
ALERT_GAP_WARNING = pd.Timedelta(hours=1)


def _local(instant: pd.Timestamp) -> str:
    return instant.tz_convert(SYDNEY).strftime("%Y-%m-%d %H:%M")


def _line_table(labels: pd.DataFrame) -> list[str]:
    rows = [
        f"  {'line':<5}{'windows':>9}{'labelled':>10}{'disrupted':>17}"
        f"{'rule (a)':>10}{'rule (b)':>10}{'both':>7}"
    ]
    for line, group in labels.groupby("line", sort=True):
        labelled = group[group["disrupted"].notna()]
        disrupted = int(labelled["disrupted"].astype(bool).sum())
        share = disrupted / len(labelled) if len(labelled) else float("nan")
        rule_a = int(labelled["rule_a"].sum())
        rule_b = int(labelled["rule_b"].sum())
        both = int((labelled["rule_a"] & labelled["rule_b"]).sum())
        rows.append(
            f"  {line!s:<5}{len(group):>9,}{len(labelled):>10,}"
            f"{f'{disrupted} ({share:.1%})':>17}{rule_a:>10}{rule_b:>10}{both:>7}"
        )
    return rows


def _incident_lines(incidents: list[Incident], labels: pd.DataFrame) -> list[str]:
    windows: dict[tuple[str, str], int] = {}
    for line, ids in zip(labels["line"], labels["incident_ids"], strict=True):
        for incident_id in filter(None, ids.split(",")):
            windows[(incident_id, line)] = windows.get((incident_id, line), 0) + 1
    by_incident = {incident.incident_id: incident for incident in incidents}
    rows = []
    for span in incident_spans(incidents):
        incident = by_incident[span.incident_id]
        rows.append(
            f"  {span.incident_id}  {span.line}  first seen {sydney_time(span.start)}  "
            f"{windows.get((span.incident_id, span.line), 0):>3} window(s)  {incident.cause_group}"
        )
    return rows


def _per_date(labels: pd.DataFrame) -> list[str]:
    labelled = labels[labels["disrupted"].notna()].assign(
        disrupted=lambda frame: frame["disrupted"].astype(bool)
    )
    table = labelled.pivot_table(
        index="service_date", columns="line", values="disrupted", aggfunc="sum", fill_value=0
    )
    lines = [f"  {'service date':<14}" + "".join(f"{line!s:>6}" for line in table.columns)]
    for date, row in table.iterrows():
        lines.append(f"  {date!s:<14}" + "".join(f"{int(value):>6}" for value in row))
    return lines


def _write(labels: pd.DataFrame, path: Path) -> None:
    out = labels.copy()
    out["window_start_utc"] = out["window_start_utc"].map(lambda t: t.isoformat())
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def command_label(args: argparse.Namespace) -> int:
    rule = LabelRule(
        late_threshold_s=args.late_threshold_s,
        late_share=args.late_share,
        min_services=args.min_services,
    )
    _, incidents = audit(load_alerts(args.db), load_overrides(args.overrides))
    coverage = load_coverage(args.db)
    events = load_stop_events(args.db)
    labels = label_windows(events, incidents, coverage, rule)

    print(f"Disruption labels from {args.db}")
    print(f"  Rule: {rule.describe()}")
    print(
        f"  Windows of {int(WINDOW.total_seconds() // 60)} min, {_local(coverage.start)} to "
        f"{_local(coverage.end)} Sydney time ({coverage.windows:,} per line)"
    )
    _print_coverage_warning(coverage)
    print()
    print("\n".join(_line_table(labels)))

    unlabelled = int(labels["disrupted"].isna().sum())
    lost = int((labels["rule_a"] & labels["disrupted"].isna()).sum())
    print()
    print(f"  Not labelled: {unlabelled:,} window(s) with no observed service.")
    print(f"  Rule (a) windows among them: {lost} -- incident windows the population loses.")
    if rule.min_services > 1:
        literal = label_windows(
            events, incidents, coverage, LabelRule(rule.late_threshold_s, rule.late_share, 1)
        )
        extra = int((literal["rule_b"] & ~labels["rule_b"]).sum())
        print(
            f"  With no minimum, rule (b) would also fire on {extra} window(s) with fewer "
            f"than {rule.min_services} services."
        )

    print(f"\nIncidents behind rule (a): {len(incidents)}")
    print("\n".join(_incident_lines(incidents, labels)))
    print("\nDisrupted windows per service date:")
    print("\n".join(_per_date(labels)))

    if args.cancellation_check:
        placement = unserved_calls(load_trip_statuses(args.db), events, discover(), rule)
        checked = label_windows(events, incidents, coverage, rule, unserved=placement.calls)
        print()
        print("\n".join(_check_lines(placement, compare_labels(labels, checked), rule)))

    if args.out is not None:
        _write(labels, args.out)
        print(f"\nwrote {len(labels):,} rows to {args.out}")
    return 0


def _check_lines(placement: Placement, windows: pd.DataFrame, rule: LabelRule) -> list[str]:
    """The label check: what counting unserved services as late changes (``docs/11`` §7)."""
    calls = placement.calls["why"].value_counts()
    changed = windows[
        (windows["n_services"] != windows["n_services_check"])
        | (windows["n_late"] != windows["n_late_check"])
    ]
    was = windows["disrupted"].fillna(False).astype(bool)
    now = windows["disrupted_check"].fillna(False).astype(bool)
    entered = windows["disrupted"].isna() & windows["disrupted_check"].notna()
    lines = [
        "Label check (docs/08 §3.5): rule (b) counting cancelled and skipped-stop services "
        f"as late, on service dates from {STATUSES_FROM}",
        f"  Cancelled: {placement.considered.get(CANCELLED, 0)} trip(s), "
        f"{placement.trips(CANCELLED)} unserved in {int(calls.get(CANCELLED, 0))} window(s) "
        "while the cancellation stood.",
        f"  Skipped stops: {placement.considered.get(SKIPPED, 0)} trip(s), "
        f"{placement.trips(SKIPPED)} skipping a call in {int(calls.get(SKIPPED, 0))} window(s) "
        f"while the skip stood. Not in any timetable: {len(placement.unplaced)}.",
        f"  Windows whose counts change: {len(changed)} "
        f"({int((windows['n_services_check'] - windows['n_services']).sum())} service(s) added, "
        f"{int((windows['n_late_check'] - windows['n_late']).sum())} more counted late).",
        f"  Rule (b) windows: {int(windows['rule_b'].sum())} -> "
        f"{int(windows['rule_b_check'].sum())}. Disrupted windows: {int(was.sum())} -> "
        f"{int(now.sum())}. Newly labelled: {int(entered.sum())}.",
    ]
    flipped = windows[was != now]
    for row in flipped.itertuples(index=False):
        lines.append(
            f"  {row.line} {_local(row.window_start_utc)}: {row.n_late}/{row.n_services} late "
            f"-> {row.n_late_check}/{row.n_services_check}, now disrupted"
        )
    if flipped.empty:
        lines.append("  No window changes its label.")
        near = changed[
            ~changed["rule_b_check"] & (changed["n_services_check"] >= rule.min_services)
        ]
        if not near.empty:
            short = late_needed(near["n_services_check"], rule.late_share) - near["n_late_check"]
            row = near.loc[short.idxmin()]
            lines.append(
                f"  Nearest: {row['line']} {_local(row['window_start_utc'])}, "
                f"{row['n_late_check']} of {row['n_services_check']} late with the check, "
                f"{int(short.min())} short of rule (b)."
            )
    return lines


def _print_coverage_warning(coverage: Coverage) -> None:
    gap = coverage.longest_alert_gap
    if gap > ALERT_GAP_WARNING:
        print(
            f"  WARNING: alerts were not polled for {gap} at one point. Rule (a) could not "
            "see an incident then, so its windows may be falsely negative."
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="transit-label",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--db", type=Path, required=True, help="a collection snapshot (read-only)")
    parser.add_argument("--overrides", type=Path, default=DEFAULT_OVERRIDES)
    parser.add_argument("--out", type=Path, default=None, help="write the labels as CSV")
    parser.add_argument(
        "--late-threshold-s",
        type=int,
        default=LATE_THRESHOLD_S,
        help="a service is late above this many seconds (default: TfNSW's 300)",
    )
    parser.add_argument(
        "--late-share",
        type=float,
        default=LATE_SHARE,
        help="share of services late that marks a window (default: a quarter)",
    )
    parser.add_argument(
        "--min-services",
        type=int,
        default=MIN_SERVICES,
        help="fewest observed services rule (b) may fire on (default: 5)",
    )
    parser.add_argument(
        "--cancellation-check",
        action="store_true",
        help="also re-run rule (b) counting cancelled and skipped-stop services as late",
    )
    parser.set_defaults(handler=command_label)
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
