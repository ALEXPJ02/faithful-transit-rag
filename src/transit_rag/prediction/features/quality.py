"""Splitting the training table, and reporting whether it is fit to train on.

Both live here because they answer the same question from opposite ends: the
split decides what the model may learn from, and the report decides whether
there is enough of anything to be worth learning from at all.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

#: A stop event whose final observation was this close is treated as a
#: reliable outcome. Beyond it the "delay" is a forecast the train had ample
#: opportunity to revise, and should not be scored as though it were observed.
CLOSE_OBSERVATION_STOPS_AHEAD = 1

#: Beyond two hours a Sydney Trains service is operationally a cancellation or
#: a replacement, not a late train. This is a statement about what ``delay_s``
#: is allowed to *mean*, decided from how the network runs rather than from
#: the shape of the tail -- which matters, because a bound chosen by looking
#: at the data is a bound a reviewer is right to distrust.
#:
#: What it removes is a GTFS-Realtime artifact: the feed republishes the
#: previous day's run stamped with today's ``start_date``, so every predicted
#: arrival lands ~24 h past schedule. Measured on the 204,628-row table of
#: 2026-09-16, that is 7 rows from a single trip at 79,422-86,142 s (22.1-23.9
#: h). The largest *plausible* delay in the same table is 4,394 s (73 min) and
#: nothing at all falls between the two, so any bound from roughly 1.5 h to
#: 20 h removes exactly the same rows. The result does not depend on where in
#: that range this sits, which is the point.
MAX_PLAUSIBLE_DELAY_S = 7200

#: A service date needs at least this share of its rows joined to the static
#: timetable to be trained on. Below it the date is *schedule-blind*: TfNSW
#: superseded the timetable era those trips were planned under before the
#: archiver kept a copy, and the static API serves only the current era, so
#: the join can never be made (``docs/07-training-table.md``).
#:
#: It is a date-level filter, not a row-level one, because the loss is
#: date-level: an era covers whole service days. Dropping the individual null
#: rows would keep the 0.1% of 2026-09-11 that happens to join and leave a
#: date that is 99.9% absent sitting inside the chronological split.
#:
#: Like :data:`MAX_PLAUSIBLE_DELAY_S`, the number is not fitted to the data.
#: Coverage is bimodal -- measured on the 22-date table of 2026-09-24, dates
#: run either 0.0-0.2% or 96.0-99.6%, with nothing between -- so any threshold
#: from roughly 1% to 95% removes exactly the same five dates. The result does
#: not depend on where in that range this sits.
MIN_SCHEDULE_COVERAGE = 0.5


@dataclass(frozen=True)
class Split:
    """Chronological train/validation/test partition."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame

    def describe(self) -> str:
        total = len(self.train) + len(self.validation) + len(self.test)
        if total == 0:
            return "  empty split"
        lines = []
        for name, frame in (
            ("train", self.train),
            ("validation", self.validation),
            ("test", self.test),
        ):
            span = (
                f"{frame['service_date'].min()} to {frame['service_date'].max()}"
                if not frame.empty
                else "empty"
            )
            lines.append(f"  {name:<11} {len(frame):>8,} rows ({len(frame) / total:>5.1%})  {span}")
        return "\n".join(lines)


def time_based_split(
    table: pd.DataFrame, fractions: tuple[float, float, float] = (0.70, 0.15, 0.15)
) -> Split:
    """Split chronologically by service date, never at random.

    A random shuffle would put observations from the same afternoon on both
    sides of the boundary, and the model would score well by having already
    seen conditions it is being asked to predict. That is the standard way
    time-series results get quietly inflated, and it would invalidate the
    comparison against the naive-persistence baseline this project reports.

    Splitting on whole service dates rather than rows keeps a day intact: a day
    straddling the boundary leaks in exactly the same way, just less obviously.
    """
    if table.empty:
        empty = table.copy()
        return Split(empty, empty.copy(), empty.copy())

    dates = sorted(table["service_date"].unique())
    train_end = max(1, round(len(dates) * fractions[0]))
    validation_end = max(train_end, round(len(dates) * (fractions[0] + fractions[1])))

    train_dates = set(dates[:train_end])
    validation_dates = set(dates[train_end:validation_end])
    test_dates = set(dates[validation_end:])

    return Split(
        train=table[table["service_date"].isin(train_dates)].reset_index(drop=True),
        validation=table[table["service_date"].isin(validation_dates)].reset_index(drop=True),
        test=table[table["service_date"].isin(test_dates)].reset_index(drop=True),
    )


def report(table: pd.DataFrame) -> str:
    """A plain-language summary of whether this table can support a model."""
    if table.empty:
        return "Training table is empty — nothing has been reconciled."

    lines: list[str] = []
    dates = sorted(table["service_date"].unique())
    lines.append(f"Rows: {len(table):,}   service dates: {len(dates)} ({dates[0]} to {dates[-1]})")
    lines.append(f"Distinct trips: {table['trip_id'].nunique():,}")

    by_line = table["route_short_name"].value_counts()
    lines.append("Rows per line: " + ", ".join(f"{k}={v:,}" for k, v in by_line.items()))

    close = (table["stops_ahead_final"] <= CLOSE_OBSERVATION_STOPS_AHEAD).sum()
    lines.append(
        f"Observed within {CLOSE_OBSERVATION_STOPS_AHEAD} stop(s) of the event: "
        f"{close:,} ({close / len(table):.0%}) — these are the reliable outcomes"
    )

    repeated = (table["observation_count"] > 1).sum()
    lines.append(f"Backed by more than one observation: {repeated:,} ({repeated / len(table):.0%})")

    matched = table["schedule_matched"].sum()
    lines.append(
        f"Matched to the static timetable: {matched:,} ({matched / len(table):.0%}) — "
        "a falling rate means an aged bundle, or a window spanning a timetable "
        "republication with one era's bundle missing"
    )

    with_prev = table["prev_stop_delay_s"].notna().sum()
    lines.append(
        f"Have a previous-stop delay: {with_prev:,} ({with_prev / len(table):.0%}) — "
        "the first stop of each trip cannot have one"
    )

    delays = table["delay_s"].dropna()
    if not delays.empty:
        lines.append(
            f"Delay seconds — median {delays.median():.0f}, "
            f"p90 {delays.quantile(0.9):.0f}, max {delays.max():.0f}"
        )

    per_day = table.groupby("service_date").size()
    lines.append(
        f"Rows per service date — min {per_day.min():,}, median {per_day.median():,.0f}, max {per_day.max():,}"
    )
    thin = per_day[per_day < per_day.median() * 0.25]
    if len(thin):
        lines.append(
            f"WARNING: {len(thin)} day(s) far below the median — likely collection gaps: "
            + ", ".join(str(d) for d in thin.index[:5])
        )
    return "\n".join(lines)
