"""Confidence intervals by resampling whole service dates (``docs/08`` §3.5).

Rows within a day are not independent. A disruption fills several windows, and
an incident's alerts are scored together, so resampling rows would treat one
bad afternoon as many independent outcomes and report an interval far
narrower than the evidence allows. The unit resampled is the **service date**.

**An undefined resample is counted, not dropped quietly.** With few test dates,
many resamples hold no positive at all, and average precision or macro-F1 is
then undefined. The interval is computed over the resamples where the metric is
defined, and :class:`Interval` reports what share that was. A 95% interval
resting on 40% of the resamples is a different claim from one resting on all of
them, and ``docs/08`` §3.2 already expects the reasons bootstrap to be
degenerate at this sample size.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

#: Enough resamples for stable 2.5/97.5 percentiles at this data size.
DEFAULT_RESAMPLES = 2000


@dataclass(frozen=True)
class Interval:
    """A metric on the rows given, and its date-resampled 95% interval."""

    point: float
    low: float
    high: float
    dates: int
    #: Share of resamples in which the metric was defined.
    defined: float

    def describe(self, digits: int = 3) -> str:
        if math.isnan(self.point):
            return "undefined"
        if math.isnan(self.low):
            return f"{self.point:.{digits}f} (no interval: {self.dates} date(s))"
        caveat = "" if self.defined == 1 else f", defined in {self.defined:.0%} of resamples"
        return (
            f"{self.point:.{digits}f} [{self.low:.{digits}f}, {self.high:.{digits}f}]"
            f" over {self.dates} dates{caveat}"
        )


def bootstrap_by_date(
    frame: pd.DataFrame,
    metric: Callable[[pd.DataFrame], float],
    *,
    date_column: str = "service_date",
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
) -> Interval:
    """``metric(frame)``, and its 95% interval from resampling whole dates with replacement.

    The seed is fixed so a re-run reproduces the interval exactly. A date drawn
    twice contributes its rows twice, as a bootstrap requires.
    """
    point = metric(frame)
    groups = {date: rows for date, rows in frame.groupby(date_column, sort=True)}
    dates = list(groups)
    if len(dates) < 2:
        return Interval(point, math.nan, math.nan, len(dates), 0.0)

    rng = np.random.default_rng(seed)
    values = []
    for _ in range(resamples):
        drawn = rng.choice(len(dates), size=len(dates), replace=True)
        sample = pd.concat([groups[dates[i]] for i in drawn], ignore_index=True)
        value = metric(sample)
        if not math.isnan(value):
            values.append(value)
    if not values:
        return Interval(point, math.nan, math.nan, len(dates), 0.0)
    low, high = np.percentile(values, [2.5, 97.5])
    return Interval(point, float(low), float(high), len(dates), len(values) / resamples)
