"""The error margin the delay tool states: split conformal intervals, per bin.

RQ1 Objective 2 requires a prediction to carry an error margin, and the margin
has to hold up (``docs/08`` §4). The global MAE cannot be that margin.
Conditional error varies by an order of magnitude, and it is largest for trains
that are already late, which is exactly when a rider asks.

**Split conformal.** On the validation split, the absolute residual
``|y - ŷ|`` is the nonconformity score. Its ``⌈(n+1)(1-alpha)⌉``-th smallest value
``q̂`` makes ``ŷ ± q̂`` cover the truth with probability at least ``1 - alpha`` for a
new row exchangeable with the validation rows (Angelopoulos & Bates, 2021). It
is distribution-free, and it is fitted on validation and **never on test**, like
every other choice in this package.

**Mondrian.** The same is done separately within each bin, so the guarantee
holds per bin rather than only on average. A bin is a function of the features
alone, because the interval has to be known before the train arrives.

**The bins are line x how late the train already is**, not the plan's line x
hour band x peak. The plan's own argument is about trains already running late,
and its bins do not reach them. This was measured on the validation split of the
2026-10-05 table, cross-fitted between its two halves. |residual| varies nine-fold
across lateness bands (90th percentile 26 s on time, 236 s already more than
five minutes late), but only 1.7-fold across hour bands. Trains already more
than five minutes late were covered 61% and 49% of the time under the plan's
bins, against 89% and 75% under these, and the median interval narrowed from
about 44 s to 28 s. The choice was made on validation, as every other choice in
this package is. The test split played no part in it.

Rail delays are autocorrelated within a day, so exchangeability holds only
approximately. Binning mitigates this; it does not remove it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from transit_rag.prediction.features.reconcile import AM_PEAK, PM_PEAK

#: The 90% interval ``docs/08`` §2 states for O2.
DEFAULT_ALPHA = 0.10

#: Hour bands aligned with the peak definitions in ``reconcile.py``, so the
#: peak flag splits only the two bands where it can be true.
HOUR_BANDS: tuple[tuple[str, int, int], ...] = (
    ("night", 0, AM_PEAK[0]),
    ("morning", AM_PEAK[0], AM_PEAK[1]),
    ("midday", AM_PEAK[1], PM_PEAK[0]),
    ("afternoon", PM_PEAK[0], PM_PEAK[1]),
    ("evening", PM_PEAK[1], 24),
)

#: How late the train already was at its previous stop. This is known when the
#: question is asked, and it is where conditional error was measured to vary
#: most (``docs/08`` §4).
LATENESS_BANDS: tuple[tuple[str, float, float], ...] = (
    ("first stop", math.nan, math.nan),
    ("on time (<= 60 s)", -math.inf, 60),
    ("late 1-5 min", 60, 300),
    ("late > 5 min", 300, math.inf),
)

#: Chosen on validation (module docstring). The plan's bins were
#: ``("line", "hour_band", "peak")`` and stay available as a key.
DEFAULT_KEY: tuple[str, ...] = ("line", "lateness")


def hour_band(hours: pd.Series) -> pd.Series:
    """Name the band each local hour falls in."""
    labels = pd.Series("", index=hours.index, dtype=object)
    for name, start, end in HOUR_BANDS:
        labels[hours.between(start, end - 1)] = name
    return labels


def lateness_band(previous_delay_s: pd.Series) -> pd.Series:
    """Name the band each previous-stop delay falls in; a trip's first stop has none."""
    values = previous_delay_s.astype(float)
    labels = pd.Series(LATENESS_BANDS[0][0], index=values.index, dtype=object)
    for name, low, high in LATENESS_BANDS[1:]:
        labels[(values > low) & (values <= high)] = name
    return labels


def bin_columns(table: pd.DataFrame) -> pd.DataFrame:
    """Every column a bin key can be built from, one row per table row."""
    return pd.DataFrame(
        {
            "line": table["route_short_name"].astype(str).to_numpy(),
            "hour_band": hour_band(table["hour_local"]).to_numpy(),
            "peak": np.where(table["is_peak"].astype(bool), "peak", "off-peak"),
            "lateness": lateness_band(table["prev_stop_delay_s"]).to_numpy(),
        },
        index=table.index,
    )


def bin_keys(table: pd.DataFrame, key: Sequence[str] = DEFAULT_KEY) -> pd.Series:
    """One string per row naming its bin, for example ``"T1|morning|peak"``."""
    columns = bin_columns(table)
    unknown = sorted(set(key) - set(columns.columns))
    if unknown:
        raise KeyError(f"no bin column named {', '.join(unknown)}")
    return columns[list(key)].astype(str).agg("|".join, axis=1)


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """The ``⌈(n+1)(1-alpha)⌉``-th smallest score, or infinity when ``n`` is too small.

    With fewer than ``1/alpha - 1`` scores, no finite quantile carries the
    guarantee. An infinite half-width is the honest answer: it says the
    interval is unknown, rather than quoting a margin that has not been
    earned.
    """
    n = scores.size
    rank = math.ceil((n + 1) * (1 - alpha))
    if n == 0 or rank > n:
        return math.inf
    return float(np.sort(scores)[rank - 1])


@dataclass(frozen=True)
class ConformalIntervals:
    """Half-widths fitted on validation: one per bin, and a global fallback."""

    alpha: float
    key: tuple[str, ...]
    global_half_width: float
    half_widths: dict[str, float] = field(default_factory=dict)
    calibration_sizes: dict[str, int] = field(default_factory=dict)

    @classmethod
    def fit(
        cls,
        truth: pd.Series | np.ndarray,
        predicted: pd.Series | np.ndarray,
        keys: pd.Series,
        alpha: float = DEFAULT_ALPHA,
        key: Sequence[str] = DEFAULT_KEY,
    ) -> ConformalIntervals:
        """Calibrate on validation rows. Never pass the test split here."""
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        scores = np.abs(np.asarray(predicted, dtype=float) - np.asarray(truth, dtype=float))
        labels = np.asarray(keys, dtype=object)
        if scores.shape != labels.shape:
            raise ValueError(f"{scores.size} residuals but {labels.size} bin keys")
        half_widths: dict[str, float] = {}
        sizes: dict[str, int] = {}
        for label in sorted(set(labels)):
            in_bin = scores[labels == label]
            half_widths[str(label)] = conformal_quantile(in_bin, alpha)
            sizes[str(label)] = int(in_bin.size)
        return cls(
            alpha=alpha,
            key=tuple(key),
            global_half_width=conformal_quantile(scores, alpha),
            half_widths=half_widths,
            calibration_sizes=sizes,
        )

    def half_width(self, keys: pd.Series) -> pd.Series:
        """Each row's margin: its bin's, or the global one for a bin never calibrated.

        A bin absent from validation falls back to the global half-width,
        which carries only the marginal guarantee. :meth:`uncalibrated` names
        those bins, so the fallback is reported rather than silent.
        """
        return keys.map(self.half_widths).astype(float).fillna(self.global_half_width)

    def uncalibrated(self, keys: pd.Series) -> list[str]:
        return sorted(set(keys) - set(self.half_widths))

    def to_dict(self) -> dict[str, object]:
        """What the model artefact stores, so inference can state the same margin."""
        return {
            "alpha": self.alpha,
            "key": list(self.key),
            "global_half_width_s": self.global_half_width,
            "half_widths_s": self.half_widths,
            "calibration_sizes": self.calibration_sizes,
        }


def coverage_table(
    truth: pd.Series | np.ndarray,
    predicted: pd.Series | np.ndarray,
    half_widths: pd.Series | np.ndarray,
    segments: pd.Series,
) -> pd.DataFrame:
    """Share of rows inside ``ŷ ± half-width``, overall and per segment.

    This is the evidence ``docs/08`` §4 asks for. A marginal 90% can hide a
    segment at 60%, so the table is read per row, worst coverage first.
    """
    frame = pd.DataFrame(
        {
            "segment": np.asarray(segments, dtype=object),
            "inside": np.abs(np.asarray(predicted, dtype=float) - np.asarray(truth, dtype=float))
            <= np.asarray(half_widths, dtype=float),
            "half_width": np.asarray(half_widths, dtype=float),
        }
    )
    rows = [
        {
            "segment": segment,
            "n": len(group),
            "coverage": float(group["inside"].mean()),
            "median_half_width_s": float(group["half_width"].median()),
        }
        for segment, group in frame.groupby("segment", sort=True)
    ]
    rows.append(
        {
            "segment": "all",
            "n": len(frame),
            "coverage": float(frame["inside"].mean()) if len(frame) else math.nan,
            "median_half_width_s": float(frame["half_width"].median()) if len(frame) else math.nan,
        }
    )
    return pd.DataFrame(rows).sort_values("coverage", kind="stable").reset_index(drop=True)
