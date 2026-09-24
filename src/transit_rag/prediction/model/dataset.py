"""The feature contract: what the model may see, and what it must never see.

This module exists to be tested. Everything else in the package can be wrong in
ways that show up as a bad score; a leaked feature is wrong in a way that shows
up as an *excellent* score, and would be believed.

The rule is a single question asked of every column: **would this value be known
at the moment the question is asked?** A rider asking "will my 17:40 from Central
be late?" is asking before the train arrives, so anything derived from the
arrival itself is unavailable, and anything describing how this project happened
to collect the row is not a fact about the world at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from transit_rag.prediction.features.quality import (
    CLOSE_OBSERVATION_STOPS_AHEAD,
    MAX_PLAUSIBLE_DELAY_S,
    MIN_SCHEDULE_COVERAGE,
)

#: What the model predicts: arrival delay in seconds, departure as a fallback.
#: Built in ``reconcile.build_training_table``.
TARGET = "delay_s"

#: Features, and the case for each being knowable before the train arrives.
#:
#: ``scheduled_arrival_s`` is the *timetabled* time, not the actual one -- see
#: the note on "scheduled-vs-actual delta" below.
FEATURE_COLUMNS: tuple[str, ...] = (
    "scheduled_arrival_s",  # timetable says the train is due at this second
    "stop_sequence",  # how far into its run the train is
    "prev_stop_delay_s",  # how late it already was at the last stop it reported
    "hour_local",  # Sydney local hour
    "day_of_week",  # Monday = 0
    "is_weekend",
    "is_peak",
    "route_short_name",  # T1 or T4
    "stop_id",  # some stations are chronically late
)

#: Columns treated as categories rather than numbers. ``stop_id`` is an integer
#: in the table but nothing about it is ordinal -- stop 201171 is not "less
#: than" stop 2770594 in any sense the model should exploit.
CATEGORICAL_COLUMNS: tuple[str, ...] = ("route_short_name", "stop_id")

#: Columns that must never become features, with the reason. Asserted in tests:
#: this is the guard that a good score is a real one.
FORBIDDEN_COLUMNS: dict[str, str] = {
    # --- the target, and the two columns it is built from ---
    "delay_s": "the target itself",
    "arrival_delay_s": "a component of the target (reconcile.py builds delay_s from it)",
    "departure_delay_s": "the other component of the target",
    # --- artefacts of how this project collected the row ---
    # None of these describe the world; they describe our polling. A rider's
    # question carries no information about how many times we happened to
    # observe the trip, or how close to the event our last look was.
    "stops_ahead_final": "how close our last observation was -- a property of collection",
    "observation_count": "how many times we saw the row -- a property of collection",
    "observed_at_utc": "when we polled, not when the train is due",
    "schedule_matched": "whether our bundle knew the trip -- a property of our join",
    # --- identifiers that cannot generalise ---
    # A trip_id embeds a timetable version and never recurs; splitting on it
    # would memorise individual trips and score well in-sample only.
    "trip_id": "unique per trip and encodes the timetable version; memorisation, not learning",
    "service_date": "the split key; using it would let the model learn each day directly",
    "route_id": "redundant with route_short_name at far higher cardinality",
}


class LeakageError(RuntimeError):
    """A forbidden column reached the feature matrix."""


@dataclass(frozen=True)
class Dataset:
    """A feature matrix, its target, and the service dates behind each row.

    ``service_date`` rides along rather than being a feature so that scores can
    be reported per day without the model ever seeing it.
    """

    features: pd.DataFrame
    target: pd.Series
    service_date: pd.Series

    def __len__(self) -> int:
        return len(self.target)


def load_training_table(path: Path) -> pd.DataFrame:
    """Read a reconciled training table.

    ``stop_id`` is read as a string: it is an identifier, and letting pandas
    infer int64 invites it to be used as a magnitude somewhere downstream.
    """
    return pd.read_csv(path, dtype={"stop_id": str, "trip_id": str})


def filter_reliable(
    table: pd.DataFrame, max_stops_ahead: int = CLOSE_OBSERVATION_STOPS_AHEAD
) -> pd.DataFrame:
    """Keep only stop events whose final observation was close to the event.

    Beyond a stop or so out, the recorded "delay" is a forecast the train had
    ample opportunity to revise, and training on it teaches the model to predict
    *the feed's guesses* rather than what happened. ``docs/07-training-table.md``
    calls the close rows the honest size of the dataset; measured on 3-9
    September 2026 they are 90% of it, so the filter is nearly free.
    """
    return table[table["stops_ahead_final"] <= max_stops_ahead].reset_index(drop=True)


def filter_plausible(table: pd.DataFrame, max_delay_s: int = MAX_PLAUSIBLE_DELAY_S) -> pd.DataFrame:
    """Drop stop events whose delay could not have happened.

    See :data:`~transit_rag.prediction.features.quality.MAX_PLAUSIBLE_DELAY_S`
    for why the bound is where it is. Two things about *how* it is applied
    matter more than the number:

    **``prev_stop_delay_s`` is bounded too, not just the target.** The naive
    baseline predicts straight from that column, so an implausible value there
    is an implausible *baseline* prediction -- the model would be compared
    against a strawman on those rows rather than beating a fair one. On the
    2026-09-16 table no row has a poisoned feature and a clean target, so this
    costs nothing today; it is here for the ghost trip whose first stop is
    corrupt and whose second is not.

    **It runs before the split, never after.** Filtering a partition after the
    boundary is drawn -- or worse, filtering only test -- changes what each
    partition means and is indistinguishable from choosing the rows that
    flatter the result.
    """
    within = table["delay_s"].abs() <= max_delay_s
    # NaN is left alone: absent is a different claim from implausible, and the
    # first stop of every trip has no previous-stop delay by construction.
    previous = table["prev_stop_delay_s"].abs() <= max_delay_s
    return table[within & (previous | table["prev_stop_delay_s"].isna())].reset_index(drop=True)


def schedule_coverage(table: pd.DataFrame) -> pd.Series:
    """Share of each service date's rows that joined to the static timetable."""
    return table.groupby("service_date")["scheduled_arrival_s"].apply(lambda s: s.notna().mean())


def filter_schedule_covered(
    table: pd.DataFrame, min_coverage: float = MIN_SCHEDULE_COVERAGE
) -> pd.DataFrame:
    """Drop whole service dates the static timetable can no longer describe.

    See :data:`~transit_rag.prediction.features.quality.MIN_SCHEDULE_COVERAGE`
    for why the threshold is where it is and why it is date-level. What matters
    about *how* it is applied is the same as for the plausibility bound:

    **It runs before the split.** A schedule-blind date is missing
    ``scheduled_arrival_s`` and ``stop_sequence``, so leaving one in changes
    what its partition *measures* rather than merely enlarging it. Not
    hypothetical: the 14-date run of 2026-09-16
    (``models/delay_model_14d_metrics.json``) took its whole validation split
    from 09-13 and 09-14 and half its test split from 09-15 -- all three blind
    -- so early stopping and the hyperparameter search were both decided on
    rows missing those two columns.

    Where the blind window lands depends on how many dates exist: on the
    22-date table of 2026-09-24 it falls wholly inside train and both other
    partitions are clean. That it happens to be harmless at one table size is
    the reason to filter rather than to trust the split boundaries.

    **It is not what makes the model look good.** The filter cannot reach the
    test split -- on the 2026-09-24 table both settings score the identical
    43,800 test rows against the identical baseline -- and excluding the blind
    dates moves MASE from 0.834 to 0.828. It buys 0.8%, not the headline.

    **Whole dates leave, not rows.** Keeping the ~0.1% of a blind date that
    happens to join would put a handful of unrepresentative rows in the split
    and leave the date's boundaries in place.
    """
    coverage = schedule_coverage(table)
    keep = set(coverage[coverage >= min_coverage].index)
    return table[table["service_date"].isin(keep)].reset_index(drop=True)


def category_dtypes(table: pd.DataFrame) -> dict[str, pd.CategoricalDtype]:
    """Fix the category levels once, from the whole table, before splitting.

    Letting each split call ``.astype("category")`` for itself gives every
    partition its own level set and its own integer codes, so a stop that
    appears in validation but not in train is either an unseen category (which
    XGBoost rejects outright) or -- worse -- silently the *same code* as a
    different stop. The levels have to be decided once and shared.

    The same dtypes are saved beside the model, because inference has to encode
    a stop id exactly as training did or the codes mean nothing.
    """
    return {
        column: pd.CategoricalDtype(categories=sorted(table[column].dropna().unique()))
        for column in CATEGORICAL_COLUMNS
    }


def build_features(
    table: pd.DataFrame, dtypes: dict[str, pd.CategoricalDtype] | None = None
) -> pd.DataFrame:
    """Project a training table onto the permitted feature columns.

    ``dtypes`` fixes the categorical levels; pass the result of
    :func:`category_dtypes` over the *full* dataset so every split encodes
    identically. Omitting it derives levels from this table alone, which is only
    safe when the table is the whole dataset.

    Raises :class:`LeakageError` if a forbidden column somehow survives, which
    can only happen if :data:`FEATURE_COLUMNS` and :data:`FORBIDDEN_COLUMNS`
    have been edited into disagreement.
    """
    missing = [column for column in FEATURE_COLUMNS if column not in table.columns]
    if missing:
        raise KeyError(f"training table is missing feature columns: {', '.join(missing)}")

    features = table.loc[:, list(FEATURE_COLUMNS)].copy()

    leaked = sorted(set(features.columns) & set(FORBIDDEN_COLUMNS))
    if leaked:
        reasons = "; ".join(f"{name} ({FORBIDDEN_COLUMNS[name]})" for name in leaked)
        raise LeakageError(f"forbidden columns reached the feature matrix: {reasons}")

    levels = dtypes if dtypes is not None else category_dtypes(table)
    for column in CATEGORICAL_COLUMNS:
        features[column] = features[column].astype(levels[column])
    for column in ("is_weekend", "is_peak"):
        features[column] = features[column].astype(bool)

    return features


def build_dataset(
    table: pd.DataFrame, dtypes: dict[str, pd.CategoricalDtype] | None = None
) -> Dataset:
    """Training table in, model-ready dataset out."""
    return Dataset(
        features=build_features(table, dtypes),
        target=table[TARGET].astype(float).reset_index(drop=True),
        service_date=table["service_date"].reset_index(drop=True),
    )
