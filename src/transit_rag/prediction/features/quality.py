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
        "a falling rate means the bundle has aged; re-fetch it"
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
