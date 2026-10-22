"""``transit-detect`` -- the detection table, and the baseline every detector must beat.

    transit-detect --db data/delay_observations_20261005.db              # baseline and detectors
    transit-detect --db ... --baseline-only                             # persistence alone
    transit-detect --db ... --out data/detection_table.csv              # also write the table
    transit-detect --db ... --compare                                   # add docs/18's models

Builds one row per T1/T4 window: what was known at its end, and whether the line
was disrupted in the next 30 minutes (``docs/12``). It splits the rows
chronologically by service date, 70/15/15, as the delay model does. It scores
naive persistence, then XGBoost and a random forest chosen on validation, with
and without the alert feature (``docs/08`` §3.5). The number of positive targets
in each split is printed first, because at a few per split no comparison means
anything. It reads the snapshot and writes only to ``--out``.

``--compare`` adds LightGBM, CatBoost and logistic regression, after the scored
detectors and in the same format, then ranks every system against persistence
(``docs/18``). Without it the output is exactly what the scored run printed. It
refuses ``--save`` and ``--out``: the comparison writes nothing.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple

import pandas as pd

from transit_rag.evaluation.lead_time import lead_times, summarise
from transit_rag.evaluation.stats import Interval, bootstrap_by_date, paired_by_date
from transit_rag.ingestion.alerts import (
    DEFAULT_OVERRIDES,
    Incident,
    audit,
    load_alerts,
    load_overrides,
)
from transit_rag.prediction.disruption.detection import (
    DetectionScores,
    average_precision,
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
from transit_rag.prediction.disruption.models import FittedDetector, fit_detector, save_detector
from transit_rag.prediction.features.quality import Split, time_based_split


def build(db: Path, overrides: Path, rule: LabelRule) -> tuple[pd.DataFrame, list[Incident]]:
    """The detection table for one snapshot, and its incidents, from the same file."""
    alerts = load_alerts(db)
    _, incidents = audit(alerts, load_overrides(overrides))
    events = load_stop_events(db)
    labels = label_windows(events, incidents, load_coverage(db), rule)
    table = detection_table(labels, events, alerts, load_alert_polls(db), rule)
    return table, incidents


def _extras(
    part: pd.DataFrame,
    score: pd.Series,
    threshold: float,
    incidents: list[Incident],
    interval: Interval,
) -> list[str]:
    """The AP's date-resampled 95% interval, and lead time against the operator's alerts."""
    leads = summarise(lead_times(part, score >= threshold, incidents))
    return [f"      AP {interval.describe()}", f"      lead time: {leads.describe()}"]


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
    table, incidents = build(args.db, args.overrides, rule)
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
        score = persistence(part)
        print(_score_line(name, score_detector(part, score, threshold=0.5)))
        print("\n".join(_extras(part, score, 0.5, incidents, _ap_interval(part, score))))

    if not args.baseline_only:
        chosen = _print_detectors(split, incidents, compare=args.compare)
        if args.save is not None:
            save_detector(
                chosen,
                args.save,
                trained_on={
                    "snapshot": str(args.db),
                    "min_services": args.min_services,
                    **{
                        f"{name}_dates": sorted(getattr(split, name)["service_date"].unique())
                        for name in ("train", "validation", "test")
                    },
                    "saved_at_utc": datetime.now(UTC).isoformat(),
                },
            )
            print(f"\nsaved the XGBoost detector to {args.save}")

    if args.out is not None:
        out = table.copy()
        out["window_start_utc"] = out["window_start_utc"].map(lambda t: t.isoformat())
        args.out.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.out, index=False)
        print(f"\nwrote {len(out):,} rows to {args.out}")
    return 0


#: The ablation ``docs/08`` §3.5 names: the same detectors, blind to the operator.
WITHOUT_ALERT: tuple[str, ...] = tuple(c for c in FEATURE_COLUMNS if c != "alert_in_feed")


#: What the scored run fitted, in the order it printed them: (label, model, features).
SCORED_RUNS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("xgboost", "xgboost", FEATURE_COLUMNS),
    ("forest", "random forest", FEATURE_COLUMNS),
    ("xgb -alert", "xgboost", WITHOUT_ALERT),
    ("rf -alert", "random forest", WITHOUT_ALERT),
)
#: ``docs/18``'s post-hoc comparison, each with and without the alert feature too.
ADDED_RUNS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("lightgbm", "lightgbm", FEATURE_COLUMNS),
    ("catboost", "catboost", FEATURE_COLUMNS),
    ("logistic", "logistic regression", FEATURE_COLUMNS),
    ("lgbm -alert", "lightgbm", WITHOUT_ALERT),
    ("cat -alert", "catboost", WITHOUT_ALERT),
    ("logit -alert", "logistic regression", WITHOUT_ALERT),
)


class _Run(NamedTuple):
    """A fitted detector and what it scored, so the comparison computes nothing twice."""

    detector: FittedDetector
    validation_score: pd.Series
    test_score: pd.Series
    test_interval: Interval


def _print_detectors(
    split: Split, incidents: list[Incident], *, compare: bool = False
) -> FittedDetector:
    """Fit and score each detector; return the full-feature XGBoost one, the headline."""
    runs: dict[str, _Run] = {}
    print()
    print("Detectors -- chosen on validation by average precision; test scored once:")
    print(
        f"  {'':<11}{'rows':>6}{'pos':>6}{'AP':>8}{'prec':>7}{'recall':>7}{'F1':>6}"
        f"{'FA/day':>9}{'rec (a)':>8}{'rec (b)':>8}"
    )
    for label, name, columns in SCORED_RUNS:
        runs[label] = _fit_and_print(label, name, columns, split, incidents)
        if label == "xgboost":
            top = runs[label].detector.importance().head(6)
            print("    importance: " + ", ".join(f"{k} {v:.0%}" for k, v in top.items()))

    if compare:
        print()
        print("Added for comparison (docs/18) -- the same rule, after the scored run:")
        for label, name, columns in ADDED_RUNS:
            runs[label] = _fit_and_print(label, name, columns, split, incidents)
        _print_comparison(split, runs)
    return runs["xgboost"].detector


def _fit_and_print(
    label: str, name: str, columns: tuple[str, ...], split: Split, incidents: list[Incident]
) -> _Run:
    detector = fit_detector(name, split.train, split.validation, columns)
    print(f"  {label} (threshold {detector.threshold:.3f}, {detector.params})")
    scores: dict[str, pd.Series] = {}
    intervals: dict[str, Interval] = {}
    for part_name in ("validation", "test"):
        part = getattr(split, part_name)
        scores[part_name] = detector.score(part)
        intervals[part_name] = _ap_interval(part, scores[part_name])
        print(_score_line(part_name, score_detector(part, scores[part_name], detector.threshold)))
        print(
            "\n".join(
                _extras(
                    part, scores[part_name], detector.threshold, incidents, intervals[part_name]
                )
            )
        )
    return _Run(detector, scores["validation"], scores["test"], intervals["test"])


def _ap_of(rows: pd.DataFrame, column: str) -> float:
    return average_precision(
        rows["target"].to_numpy(dtype=bool), rows[column].to_numpy(dtype=float)
    )


def _ap_interval(part: pd.DataFrame, score: pd.Series) -> Interval:
    """AP on the rows whose target is known, and its date-resampled 95% interval."""
    known = part["target"].notna()
    scored = part.loc[known].assign(_score=score[known])
    return bootstrap_by_date(scored, lambda rows: _ap_of(rows, "_score"))


def paired_gap(part: pd.DataFrame, score: pd.Series, baseline: pd.Series) -> Interval:
    """AP(score) minus AP(baseline), and its 95% interval from resampling whole dates.

    Paired: each resample scores both systems on the same drawn dates, so a hard
    day lowers both. Over five test dates the two systems' own intervals overlap
    almost entirely, and comparing them would hide a gap that holds on every draw.
    """
    known = part["target"].notna()
    scored = part.loc[known].assign(_score=score[known], _baseline=baseline[known])
    return paired_by_date(scored, _ap_of, "_score", "_baseline")


def _print_comparison(split: Split, runs: dict[str, _Run]) -> None:
    """Every system on test, beside persistence, with the gap paired by date."""
    test, validation = split.test, split.validation
    baseline = persistence(test)
    positives = int(test.loc[test["target"].notna(), "target"].astype(bool).sum())
    dates = test["service_date"].nunique()

    print()
    print(
        f"Every system on test ({positives} positives over {dates} dates). The gap to "
        "persistence resamples the same dates for both:"
    )
    print(
        f"  {'':<13}{'val AP':>7}{'test AP':>8}  {'95% interval':<16}"
        f"{'AP - persistence':>17}  {'95% interval':<18}{'F1':>5}{'FA/day':>8}"
    )
    held = score_detector(validation, persistence(validation), threshold=0.5)
    scores = score_detector(test, baseline, threshold=0.5)
    print(
        f"  {'persistence':<13}{held.average_precision:>7.3f}{scores.average_precision:>8.3f}"
        f"  {_ap_interval(test, baseline).bounds():<16}{'--':>17}  {'':<18}"
        f"{scores.f1:>5.0%}{scores.false_alarms_per_day:>8.1f}"
    )
    partial: list[str] = []
    for label, run in runs.items():
        scores = score_detector(test, run.test_score, run.detector.threshold)
        gap = paired_gap(test, run.test_score, baseline)
        if 0 < gap.defined < 1:
            partial.append(f"{label} ({gap.defined:.0%})")
        print(
            f"  {label:<13}{run.detector.validation_ap:>7.3f}{scores.average_precision:>8.3f}"
            f"  {run.test_interval.bounds():<16}{gap.point:>+17.3f}  {gap.bounds('+.3f'):<18}"
            f"{scores.f1:>5.0%}{scores.false_alarms_per_day:>8.1f}"
        )
    if partial:
        print("  gap defined in fewer than every resample: " + ", ".join(partial))


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
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--baseline-only", action="store_true", help="score persistence and fit nothing"
    )
    mode.add_argument(
        "--compare",
        action="store_true",
        help="also fit LightGBM, CatBoost and logistic regression, and rank every system (docs/18)",
    )
    parser.add_argument(
        "--save", type=Path, default=None, help="write the XGBoost detector, for the agent"
    )
    parser.set_defaults(handler=command_detect)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.compare and (args.save is not None or args.out is not None):
        parser.error("--compare writes nothing (docs/18); run --save or --out without it")
    try:
        result: int = args.handler(args)
        return result
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
