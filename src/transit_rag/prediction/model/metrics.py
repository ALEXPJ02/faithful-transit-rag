"""Scoring, and the honest handling of a metric that does not apply.

``docs/02-tech-stack.md`` specifies MAE, RMSE and MAPE. Two of those work.

**MAPE does not, and the write-up has to say so.** It divides by the true value,
and on this target the true value is *exactly zero* for 53.8% of rows (measured
on 3-9 September 2026): most trains are on time to the second, which is a fact
about Sydney Trains rather than a defect in the data. MAPE is therefore undefined
for more than half the dataset and unbounded near it, so reporting it as
specified would produce either an error or a meaningless number.

What is reported instead:

* **MAE and RMSE** as the headline, in seconds and minutes.
* **MAPE over the non-zero subset only**, explicitly labelled with the share of
  rows it covers, so the specified metric still appears but cannot mislead.
* **MASE** -- MAE scaled by the naive baseline's MAE -- as the principled
  substitute. It is scale-free like MAPE but defined at zero, and it reads
  directly against the comparison this project is built around: below 1.0 means
  better than naive persistence, above means worse.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SECONDS_PER_MINUTE = 60.0


@dataclass(frozen=True)
class Scores:
    """One model's error on one set of rows."""

    n: int
    mae_s: float
    rmse_s: float
    #: MAPE over rows whose true delay is non-zero. ``None`` when there are none.
    mape_nonzero: float | None
    #: Share of rows the MAPE figure covers.
    mape_coverage: float

    @property
    def mae_min(self) -> float:
        return self.mae_s / SECONDS_PER_MINUTE

    @property
    def rmse_min(self) -> float:
        return self.rmse_s / SECONDS_PER_MINUTE

    def line(self, label: str) -> str:
        mape = f"{self.mape_nonzero:>7.0f}%" if self.mape_nonzero is not None else f"{'n/a':>8}"
        return (
            f"  {label:<22}{self.n:>8,}"
            f"{self.mae_s:>9.1f}s ({self.mae_min:>4.1f}m)"
            f"{self.rmse_s:>9.1f}s"
            f"{mape}"
        )


def score(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> Scores:
    """MAE, RMSE and a guarded MAPE over aligned true/predicted values."""
    truth = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    if truth.shape != predicted.shape:
        raise ValueError(f"shape mismatch: {truth.shape} true vs {predicted.shape} predicted")
    if truth.size == 0:
        return Scores(
            n=0, mae_s=float("nan"), rmse_s=float("nan"), mape_nonzero=None, mape_coverage=0.0
        )

    error = predicted - truth
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))

    # Guarded, not silently dropped: the coverage figure travels with the value
    # so a reader can see how much of the data the percentage speaks for.
    nonzero = truth != 0
    mape = float(np.mean(np.abs(error[nonzero] / truth[nonzero])) * 100) if nonzero.any() else None

    return Scores(
        n=int(truth.size),
        mae_s=mae,
        rmse_s=rmse,
        mape_nonzero=mape,
        mape_coverage=float(nonzero.mean()),
    )


def mase(model: Scores, baseline: Scores) -> float:
    """Model MAE scaled by the baseline's. Below 1.0 beats naive persistence.

    Scale-free like MAPE but defined when the truth is zero, which is why it
    stands in for MAPE here.
    """
    if baseline.mae_s == 0:
        return float("inf") if model.mae_s > 0 else float("nan")
    return model.mae_s / baseline.mae_s


def by_segment(
    segment: pd.Series,
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> pd.DataFrame:
    """Scores broken down by a categorical column, worst MAE first.

    Not decoration. The faithfulness metric uses MAE as the basis for the error
    margin the agent must state, so a single global MAE is only defensible if
    error is reasonably uniform. Where it is not, the margin has to be segmented
    or the agent will overclaim in exactly the conditions riders care about.
    """
    frame = pd.DataFrame(
        {
            "segment": np.asarray(segment),
            "truth": np.asarray(y_true, dtype=float),
            "predicted": np.asarray(y_pred, dtype=float),
        }
    )
    rows = []
    for value, group in frame.groupby("segment", observed=True):
        scores = score(group["truth"], group["predicted"])
        rows.append({"segment": value, "n": scores.n, "mae_s": scores.mae_s})
    return pd.DataFrame(rows).sort_values("mae_s", ascending=False).reset_index(drop=True)


def residual_summary(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> str:
    """Percentiles of the signed error, for justifying MAE as the headline.

    Delay is heavily right-skewed, so a mean absolute error can look reassuring
    while a long tail does the damage. Reporting the spread lets the write-up
    argue for MAE rather than assume it.
    """
    error = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    if error.size == 0:
        return "  no rows"
    quantiles = np.percentile(error, [5, 25, 50, 75, 95])
    labels = ("p5", "p25", "median", "p75", "p95")
    body = "  ".join(
        f"{name} {value:>+7.0f}s" for name, value in zip(labels, quantiles, strict=True)
    )
    return f"  residual (predicted - actual):  {body}"
