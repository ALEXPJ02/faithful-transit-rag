"""Tests for the conformal error margin (``docs/08`` §4).

The guarantee is statistical, so two tests check it empirically on synthetic
residuals with known spread. The rest pin the finite-sample rank and the band
edges, because an off-by-one there turns a 90% interval into an 89% one that
nobody would notice.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from transit_rag.prediction.model.conformal import (
    ConformalIntervals,
    bin_keys,
    conformal_quantile,
    coverage_table,
    hour_band,
    lateness_band,
)


class TestQuantile:
    def test_it_is_the_finite_sample_rank_not_the_plain_quantile(self) -> None:
        """⌈(n+1)(1-alpha)⌉ = ⌈90.9⌉ = 91st of 100, where np.quantile would interpolate to 90.1."""
        assert conformal_quantile(np.arange(1, 101, dtype=float), 0.1) == 91.0

    def test_too_few_scores_give_an_unknown_margin_not_a_small_one(self) -> None:
        assert conformal_quantile(np.arange(9, dtype=float), 0.1) == 8.0  # rank 9 of 9
        assert math.isinf(conformal_quantile(np.arange(8, dtype=float), 0.1))  # rank 9 of 8
        assert math.isinf(conformal_quantile(np.array([]), 0.1))


def _residuals(rng: np.random.Generator, scale: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    truth = rng.normal(0, scale, n)
    return truth, np.zeros(n)


class TestCoverage:
    def test_a_ninety_percent_interval_covers_about_ninety_percent(self) -> None:
        rng = np.random.default_rng(7)
        truth, predicted = _residuals(rng, 30, 2_000)
        keys = pd.Series(["all"] * truth.size)
        intervals = ConformalIntervals.fit(truth, predicted, keys, alpha=0.1, key=("line",))

        new_truth, new_predicted = _residuals(rng, 30, 50_000)
        width = intervals.half_width(pd.Series(["all"] * new_truth.size))
        covered = np.mean(np.abs(new_truth - new_predicted) <= width.to_numpy())
        assert 0.89 <= covered <= 0.92

    def test_per_bin_margins_hold_where_one_global_margin_does_not(self) -> None:
        """The case for Mondrian: one bin ten times noisier than the other."""
        rng = np.random.default_rng(11)
        quiet, noisy = _residuals(rng, 10, 3_000), _residuals(rng, 100, 3_000)
        truth = np.concatenate([quiet[0], noisy[0]])
        keys = pd.Series(["quiet"] * 3_000 + ["noisy"] * 3_000)
        intervals = ConformalIntervals.fit(truth, np.zeros(6_000), keys, key=("line",))
        assert intervals.half_widths["noisy"] > 8 * intervals.half_widths["quiet"]

        new_noisy = rng.normal(0, 100, 20_000)
        per_bin = np.mean(np.abs(new_noisy) <= intervals.half_widths["noisy"])
        global_only = np.mean(np.abs(new_noisy) <= intervals.global_half_width)
        assert per_bin >= 0.88
        assert global_only < 0.85


class TestHalfWidth:
    def test_a_bin_never_calibrated_falls_back_to_global_and_is_named(self) -> None:
        keys = pd.Series(["a"] * 50 + ["b"] * 50)
        intervals = ConformalIntervals.fit(
            np.arange(100.0), np.zeros(100), keys, alpha=0.1, key=("line",)
        )
        asked = pd.Series(["a", "unseen"])
        widths = intervals.half_width(asked)
        assert widths.iloc[0] == intervals.half_widths["a"]
        assert widths.iloc[1] == intervals.global_half_width
        assert intervals.uncalibrated(asked) == ["unseen"]

    def test_it_refuses_an_alpha_that_is_not_a_probability(self) -> None:
        with pytest.raises(ValueError, match="alpha"):
            ConformalIntervals.fit(np.ones(3), np.ones(3), pd.Series(["a"] * 3), alpha=1.0)

    def test_it_refuses_residuals_and_keys_of_different_lengths(self) -> None:
        with pytest.raises(ValueError, match="bin keys"):
            ConformalIntervals.fit(np.ones(3), np.ones(3), pd.Series(["a"] * 2))

    def test_the_artefact_form_carries_what_inference_needs(self) -> None:
        intervals = ConformalIntervals.fit(
            np.arange(20.0), np.zeros(20), pd.Series(["a"] * 20), key=("line",)
        )
        stored = intervals.to_dict()
        assert stored["key"] == ["line"]
        assert stored["half_widths_s"] == intervals.half_widths
        assert stored["calibration_sizes"] == {"a": 20}


class TestBins:
    def test_hour_bands_follow_the_peak_definitions(self) -> None:
        hours = pd.Series([0, 5, 6, 9, 10, 14, 15, 18, 19, 23])
        assert list(hour_band(hours)) == [
            "night",
            "night",
            "morning",
            "morning",
            "midday",
            "midday",
            "afternoon",
            "afternoon",
            "evening",
            "evening",
        ]

    def test_lateness_bands_and_the_first_stop(self) -> None:
        previous = pd.Series([np.nan, -30, 60, 61, 300, 301])
        assert list(lateness_band(previous)) == [
            "first stop",
            "on time (<= 60 s)",
            "on time (<= 60 s)",
            "late 1-5 min",
            "late 1-5 min",
            "late > 5 min",
        ]

    def test_keys_name_the_bin_from_the_table(self) -> None:
        table = pd.DataFrame(
            {
                "route_short_name": ["T1", "T4"],
                "hour_local": [8, 22],
                "is_peak": [True, False],
                "prev_stop_delay_s": [400.0, np.nan],
            }
        )
        assert list(bin_keys(table)) == ["T1|late > 5 min", "T4|first stop"]
        plan = ("line", "hour_band", "peak")
        assert list(bin_keys(table, plan)) == ["T1|morning|peak", "T4|evening|off-peak"]
        with pytest.raises(KeyError, match="weather"):
            bin_keys(table, ("weather",))


def test_the_coverage_table_puts_the_worst_segment_first() -> None:
    table = coverage_table(
        truth=np.array([0, 0, 10, 10]),
        predicted=np.zeros(4),
        half_widths=np.array([1, 1, 1, 20]),
        segments=pd.Series(["good", "good", "bad", "bad"]),
    )
    assert list(table["segment"]) == ["bad", "all", "good"]
    assert list(table["coverage"]) == [0.5, 0.75, 1.0]
    assert list(table["n"]) == [2, 4, 2]
