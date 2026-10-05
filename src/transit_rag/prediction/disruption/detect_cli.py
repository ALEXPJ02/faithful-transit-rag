"""``transit-detect`` -- the detection table, and the baseline every detector must beat.

    transit-detect --db data/delay_observations_20261005.db              # baseline and detectors
    transit-detect --db ... --baseline-only                             # persistence alone
    transit-detect --db ... --out data/detection_table.csv              # also write the table

Builds one row per T1/T4 window: what was known at its end, and whether the line
was disrupted in the next 30 minutes (``docs/12``). It splits the rows
chronologically by service date, 70/15/15, as the delay model does. It scores
naive persistence, then XGBoost and a random forest chosen on validation, with
and without the alert feature (``docs/08`` §3.5). The number of positive targets
in each split is printed first, because at a few per split no comparison means
anything. It reads the snapshot and writes only to ``--out``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from transit_rag.ingestion.alerts import DEFAULT_OVERRIDES, audit, load_alerts, load_overrides
from transit_rag.prediction.disruption.detection import (
    DetectionScores,
    persistence,
    score_detector,
)
from transit_rag.prediction.disruption.features import FEATURE_COLUMNS, detection_table
from transit_rag.prediction.disruption.labels import (
    MIN_SERVICES,
    LabelRule,
    label_windows,
    load_alert_polls,
    load_coverage,
    load_stop_events,
)
from transit_rag.prediction.disruption.models import fit_detector
from transit_rag.prediction.features.quality import Split, time_based_split


def build(db: Path, overrides: Path, rule: LabelRule) -> pd.DataFrame:
    """The detection table for one snapshot, labels and alerts from the same file."""
    alerts = load_alerts(db)
    _, incidents = audit(alerts, load_overrides(overrides))
    events = load_stop_events(db)
    labels = label_windows(events, incidents, load_coverage(db), rule)
    return detection_table(labels, events, alerts, load_alert_polls(db), rule)


def _split_lines(split: Split) -> list[str]:
    rows = [f"  {'':<11}{'rows':>7}{'known':>7}{'positive':>10}  dates"]
    for name in ("train", "validation", "test"):
        part: pd.DataFrame = getattr(split, name)
        known = part["target"].notna()
        positives = int(part.loc[known, "target"].astype(bool).sum())
        dates = sorted(part["service_date"].unique())
        span = f"{dates[0]} to {dates[-1]} ({len(dates)})" if dates else "none"
        rows.append(f"  {name:<11}{len(part):>7,}{int(known.sum()):>7,}{positives:>10}  {span}")
    return rows


def _score_line(name: str, scores: DetectionScores) -> str:
    def pct(value: float) -> str:
        return f"{value:.0%}" if value == value else "n/a"

    ap = (
        f"{scores.average_precision:.3f}"
        if scores.average_precision == scores.average_precision
        else "n/a"
    )
    return (
        f"  {name:<11}{scores.n:>6,}{scores.positives:>6}{ap:>8}{pct(scores.precision):>7}"
        f"{pct(scores.recall):>7}{pct(scores.f1):>6}{scores.false_alarms_per_day:>9.1f}"
        f"{pct(scores.recall_rule_a):>8}{pct(scores.recall_rule_b):>8}"
    )


def command_detect(args: argparse.Namespace) -> int:
    rule = LabelRule(min_services=args.min_services)
    table = build(args.db, args.overrides, rule)
    split = time_based_split(table)

    known = table["target"].notna()
    positives = int(table.loc[known, "target"].astype(bool).sum())
    print(f"Detection table from {args.db}")
    print(
        f"  {len(table):,} line-windows; the next 30 minutes are known for {int(known.sum()):,}, "
        f"of which {positives} ({positives / max(int(known.sum()), 1):.1%}) are disrupted"
    )
    print(f"  {int(table['alert_in_feed'].sum())} windows end with a disruption alert in the feed")
    print()
    print("Split by service date, 70/15/15:")
    print("\n".join(_split_lines(split)))

    print()
    print("Persistence -- the next 30 minutes are disrupted if this window is:")
    print(
        f"  {'':<11}{'rows':>6}{'pos':>6}{'AP':>8}{'prec':>7}{'recall':>7}{'F1':>6}"
        f"{'FA/day':>9}{'rec (a)':>8}{'rec (b)':>8}"
    )
    for name in ("validation", "test"):
        part = getattr(split, name)
        print(_score_line(name, score_detector(part, persistence(part), threshold=0.5)))

    if not args.baseline_only:
        _print_detectors(split)

    if args.out is not None:
        out = table.copy()
        out["window_start_utc"] = out["window_start_utc"].map(lambda t: t.isoformat())
        args.out.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.out, index=False)
        print(f"\nwrote {len(out):,} rows to {args.out}")
    return 0


#: The ablation ``docs/08`` §3.5 names: the same detectors, blind to the operator.
WITHOUT_ALERT: tuple[str, ...] = tuple(c for c in FEATURE_COLUMNS if c != "alert_in_feed")


def _print_detectors(split: Split) -> None:
    print()
    print("Detectors -- chosen on validation by average precision; test scored once:")
    print(
        f"  {'':<11}{'rows':>6}{'pos':>6}{'AP':>8}{'prec':>7}{'recall':>7}{'F1':>6}"
        f"{'FA/day':>9}{'rec (a)':>8}{'rec (b)':>8}"
    )
    for label, name, columns in (
        ("xgboost", "xgboost", FEATURE_COLUMNS),
        ("forest", "random forest", FEATURE_COLUMNS),
        ("xgb -alert", "xgboost", WITHOUT_ALERT),
        ("rf -alert", "random forest", WITHOUT_ALERT),
    ):
        fitted = fit_detector(name, split.train, split.validation, columns)
        print(f"  {label} (threshold {fitted.threshold:.3f}, {fitted.params})")
        for part_name in ("validation", "test"):
            part = getattr(split, part_name)
            scores = score_detector(part, fitted.score(part), fitted.threshold)
            print(_score_line(part_name, scores))
        if label == "xgboost":
            top = fitted.importance().head(6)
            print("    importance: " + ", ".join(f"{k} {v:.0%}" for k, v in top.items()))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="transit-detect",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--db", type=Path, required=True, help="a collection snapshot (read-only)")
    parser.add_argument("--overrides", type=Path, default=DEFAULT_OVERRIDES)
    parser.add_argument("--out", type=Path, default=None, help="write the detection table as CSV")
    parser.add_argument(
        "--min-services",
        type=int,
        default=MIN_SERVICES,
        help="fewest observed services the label's rule (b) may fire on (default: 5)",
    )
    parser.add_argument(
        "--baseline-only", action="store_true", help="score persistence and fit nothing"
    )
    parser.set_defaults(handler=command_detect)
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
