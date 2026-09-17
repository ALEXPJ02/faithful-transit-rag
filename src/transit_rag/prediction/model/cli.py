"""``transit-train`` -- fit the delay model and score it against naive persistence.

    transit-train --report-only        # train and print, write nothing
    transit-train                      # also write models/delay_model.joblib + metrics
    transit-train --curve              # MAE against training days, in one sweep
    transit-train --max-dates 14       # truncate to the first 14 service dates
    transit-train --all-rows           # sensitivity check: skip the reliability filter
    transit-train --keep-implausible   # audit what the plausibility bound excludes

**Rows the feed could not have meant.** GTFS-Realtime sometimes republishes the
previous day's run under today's ``start_date``, which reconciles to a delay of
roughly 24 hours. Those rows are dropped before the split -- see
``quality.MAX_PLAUSIBLE_DELAY_S`` -- so every partition and the naive baseline
see the same data. On the 2026-09-16 table it is 7 rows of 204,628, it moves
test MAE, RMSE and MASE not at all, and it takes validation MAE from 32.1 s to
14.5 s, because all seven landed in validation.

**How the comparison is kept fair.** The model trains on every available row --
XGBoost handles a missing ``prev_stop_delay_s`` natively -- but both predictors
are *scored* only on rows where the baseline has an input, i.e. not the first
stop of a trip. Scoring the model on rows the baseline was never offered would
flatter it by roughly the share of first stops (~6%).

**The test split is scored once**, after hyperparameter selection has finished on
validation. Nothing in this module fits or selects against it.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from transit_rag.config import PROJECT_ROOT
from transit_rag.prediction.features.quality import (
    CLOSE_OBSERVATION_STOPS_AHEAD,
    MAX_PLAUSIBLE_DELAY_S,
    time_based_split,
)
from transit_rag.prediction.model import baseline, metrics
from transit_rag.prediction.model.dataset import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    build_dataset,
    category_dtypes,
    filter_plausible,
    filter_reliable,
    load_training_table,
)
from transit_rag.prediction.model.train import FittedModel, train

log = logging.getLogger("transit_rag.train")

DEFAULT_TABLE = PROJECT_ROOT / "data" / "training_table.csv"
DEFAULT_MODEL = PROJECT_ROOT / "models" / "delay_model.joblib"
DEFAULT_METRICS = PROJECT_ROOT / "models" / "delay_model_metrics.json"

#: Prefix sizes for the learning curve. Anything larger than the dataset is
#: skipped, so the same list works on 7 dates now and ~32 in October.
CURVE_DATE_COUNTS: tuple[int, ...] = (7, 14, 21, 28, 35)


def truncate_to_dates(table: pd.DataFrame, max_dates: int) -> pd.DataFrame:
    """Keep only the earliest ``max_dates`` service dates.

    Chronological, never a sample: the point of the learning curve is to answer
    "what would this model have looked like with less collection", and taking a
    random subset would answer a different and easier question.
    """
    dates = sorted(table["service_date"].unique())[:max_dates]
    return table[table["service_date"].isin(set(dates))].reset_index(drop=True)


def _fit_and_score(table: pd.DataFrame) -> tuple[FittedModel, dict[str, object]]:
    """Split, fit on train, select on validation, score once on test."""
    split = time_based_split(table)
    if split.validation.empty or split.test.empty:
        raise SystemExit(
            "The split has an empty validation or test partition -- there are too few "
            "service dates to evaluate anything. Collect more days before training."
        )

    # Levels fixed on the full table, before splitting, so train/validation/test
    # encode every stop id identically. Per-split inference would give the same
    # stop different codes in different partitions.
    levels = category_dtypes(table)
    train_set = build_dataset(split.train, levels)
    validation_set = build_dataset(split.validation, levels)

    log.info("fitting %d candidates on %s rows", 4, f"{len(train_set):,}")
    fitted = train(train_set, validation_set)
    log.info(
        "selected depth=%s lr=%s (validation MAE %.1fs, %d trees)",
        fitted.params["max_depth"],
        fitted.params["learning_rate"],
        fitted.validation_mae_s,
        fitted.best_iteration,
    )

    # Fair-comparison mask: rows the baseline can actually predict.
    test = split.test
    comparable = test[baseline.scoreable(test)].reset_index(drop=True)
    test_set = build_dataset(comparable, levels)

    model_prediction = fitted.predict(test_set.features)
    baseline_prediction = baseline.predict(comparable)

    model_scores = metrics.score(test_set.target, model_prediction)
    baseline_scores = metrics.score(test_set.target, baseline_prediction)

    summary: dict[str, object] = {
        "split": {
            "train_rows": len(split.train),
            "validation_rows": len(split.validation),
            "test_rows": len(split.test),
            "test_rows_compared": len(comparable),
            "train_dates": sorted(split.train["service_date"].unique()),
            "validation_dates": sorted(split.validation["service_date"].unique()),
            "test_dates": sorted(split.test["service_date"].unique()),
        },
        "hyperparameters": {key: value for key, value in fitted.params.items() if key != "n_jobs"},
        "search": fitted.search,
        "test": {
            "model": _scores_dict(model_scores),
            "baseline": _scores_dict(baseline_scores),
            "mase": metrics.mase(model_scores, baseline_scores),
        },
        "segments": {
            column: metrics.by_segment(
                comparable[column], test_set.target, model_prediction
            ).to_dict("records")
            for column in ("is_peak", "route_short_name", "hour_local")
        },
        "_frames": {
            "comparable": comparable,
            "truth": test_set.target,
            "model_prediction": model_prediction,
            "baseline_prediction": baseline_prediction,
            "model_scores": model_scores,
            "baseline_scores": baseline_scores,
        },
    }
    summary["_levels"] = levels
    return fitted, summary


def _scores_dict(scores: metrics.Scores) -> dict[str, object]:
    return {
        "n": scores.n,
        "mae_s": scores.mae_s,
        "mae_min": scores.mae_min,
        "rmse_s": scores.rmse_s,
        "rmse_min": scores.rmse_min,
        "mape_nonzero_pct": scores.mape_nonzero,
        "mape_coverage": scores.mape_coverage,
    }


def _print_report(fitted: FittedModel, summary: dict[str, object]) -> None:
    frames = summary["_frames"]
    assert isinstance(frames, dict)
    model_scores = frames["model_scores"]
    baseline_scores = frames["baseline_scores"]
    split = summary["split"]
    assert isinstance(split, dict)

    print()
    print(
        f"Train {split['train_rows']:,} rows / Validation {split['validation_rows']:,} "
        f"/ Test {split['test_rows']:,}"
    )
    print(
        f"Scored on {split['test_rows_compared']:,} test rows -- those where naive "
        "persistence also has a prediction."
    )
    print()
    print(f"  {'':<22}{'rows':>8}{'MAE':>21}{'RMSE':>10}{'MAPE*':>8}")
    print(baseline_scores.line("naive persistence"))
    print(model_scores.line("XGBoost"))
    print()

    ratio = metrics.mase(model_scores, baseline_scores)
    verdict = "beats" if ratio < 1 else "does NOT beat"
    improvement = (1 - ratio) * 100
    print(f"  MASE {ratio:.3f} -- the model {verdict} naive persistence ({improvement:+.1f}% MAE).")
    print(
        f"  *MAPE covers only the {model_scores.mape_coverage:.0%} of rows with a non-zero "
        "delay; it is undefined on the rest. See metrics.py."
    )
    print()
    print(metrics.residual_summary(frames["truth"], frames["model_prediction"]))

    importance = fitted.feature_importance()
    if not importance.empty:
        print()
        print("  Feature importance (gain):")
        for row in importance.itertuples():
            print(f"    {row.feature:<24}{row.share:>7.1%}")

    segments = summary["segments"]
    assert isinstance(segments, dict)
    for column in ("is_peak", "route_short_name"):
        print()
        print(f"  MAE by {column}:")
        for record in segments[column]:
            print(f"    {record['segment']!s:<24}{record['mae_s']:>8.1f}s  (n={record['n']:,})")


def _run_curve(table: pd.DataFrame) -> None:
    """MAE against number of training days, from prefixes of one dataset.

    This is the evidence for "was the collection window long enough": a curve
    that has flattened says yes, one still falling says the model was still
    learning when collection stopped.
    """
    available = len(table["service_date"].unique())
    print()
    print(f"{'dates':>6}{'rows':>10}{'model MAE':>12}{'baseline MAE':>14}{'MASE':>8}")
    for count in CURVE_DATE_COUNTS:
        if count > available:
            continue
        subset = truncate_to_dates(table, count)
        try:
            _, summary = _fit_and_score(subset)
        except SystemExit:
            print(f"{count:>6}{len(subset):>10}   too few dates to split")
            continue
        test = summary["test"]
        assert isinstance(test, dict)
        model = test["model"]
        base = test["baseline"]
        assert isinstance(model, dict) and isinstance(base, dict)
        print(
            f"{count:>6}{len(subset):>10}{model['mae_s']:>11.1f}s{base['mae_s']:>13.1f}s"
            f"{test['mase']:>8.3f}"
        )
    print()
    print("A curve still falling at the largest prefix means more collection would help.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--table", type=Path, default=DEFAULT_TABLE, help="Reconciled training table"
    )
    parser.add_argument(
        "--max-dates",
        type=int,
        default=None,
        help="Use only the earliest N service dates (chronological, never sampled)",
    )
    parser.add_argument(
        "--all-rows",
        action="store_true",
        help=f"Skip the stops_ahead_final <= {CLOSE_OBSERVATION_STOPS_AHEAD} reliability filter",
    )
    parser.add_argument(
        "--keep-implausible",
        action="store_true",
        help=(
            f"Keep rows with |delay| over {MAX_PLAUSIBLE_DELAY_S}s. Only for auditing what the "
            f"bound excludes -- these are feed artifacts, not slow trains"
        ),
    )
    parser.add_argument(
        "--curve", action="store_true", help="Sweep training-set size and report MAE against days"
    )
    parser.add_argument("--report-only", action="store_true", help="Print, write nothing")
    parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--metrics-out", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s"
    )

    if not args.table.exists():
        log.error(
            "No training table at %s. Build one with `transit-reconcile --split` after "
            "pulling the collector database.",
            args.table,
        )
        sys.exit(1)

    table = load_training_table(args.table)
    log.info("%s rows from %s", f"{len(table):,}", args.table)

    if not args.all_rows:
        before = len(table)
        table = filter_reliable(table)
        log.info(
            "%s rows within %d stop(s) of the event (%.0f%% of %s) -- the reliable outcomes",
            f"{len(table):,}",
            CLOSE_OBSERVATION_STOPS_AHEAD,
            100 * len(table) / before,
            f"{before:,}",
        )

    # Before the split, so train, validation and test are filtered identically
    # and the naive baseline is scored on the same rows the model is. Applied
    # even under --all-rows: that flag exists to test sensitivity to the
    # reliability *heuristic*, not to train on a 24-hour delay nobody believes.
    if not args.keep_implausible:
        before = len(table)
        table = filter_plausible(table)
        removed = before - len(table)
        if removed:
            log.info(
                "dropped %s row(s) with |delay| over %ds (%.4f%%) -- see quality.MAX_PLAUSIBLE_DELAY_S",
                f"{removed:,}",
                MAX_PLAUSIBLE_DELAY_S,
                100 * removed / before,
            )

    if args.max_dates is not None:
        table = truncate_to_dates(table, args.max_dates)
        log.info(
            "truncated to the earliest %d service dates: %s rows", args.max_dates, f"{len(table):,}"
        )

    if args.curve:
        _run_curve(table)
        return

    fitted, summary = _fit_and_score(table)
    _print_report(fitted, summary)

    if args.report_only:
        return

    import joblib

    levels = summary["_levels"]
    assert isinstance(levels, dict)

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": fitted.booster,
            "feature_columns": list(FEATURE_COLUMNS),
            "categorical_columns": list(CATEGORICAL_COLUMNS),
            # Without these the artefact is unusable: inference must map a stop
            # id to the same code training used, and the levels are not
            # recoverable from the booster.
            "category_levels": {column: list(dtype.categories) for column, dtype in levels.items()},
        },
        args.model_out,
    )
    log.info("wrote %s", args.model_out)

    # The metrics file is the contract with the evaluation harness: the
    # faithfulness threshold reads MAE from here rather than hardcoding a
    # figure, so retraining updates the metric without touching the eval plan.
    serialisable = {key: value for key, value in summary.items() if not key.startswith("_")}
    serialisable["trained_at_utc"] = datetime.now(UTC).isoformat()
    serialisable["table"] = str(args.table)
    serialisable["reliability_filtered"] = not args.all_rows
    serialisable["plausibility_bound_s"] = None if args.keep_implausible else MAX_PLAUSIBLE_DELAY_S
    args.metrics_out.write_text(json.dumps(serialisable, indent=2, default=str))
    log.info("wrote %s", args.metrics_out)


if __name__ == "__main__":
    main()
