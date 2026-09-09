"""Tests for scoring, especially the metric that does not apply.

MAPE is specified in `docs/02-tech-stack.md` but undefined on this target: 53.8%
of true delays are exactly zero. These pin the guarded behaviour so a later
change cannot quietly reintroduce a divide-by-zero or, worse, a plausible-looking
number computed over a subset nobody disclosed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from transit_rag.prediction.model import metrics


class TestScore:
    def test_mae_and_rmse_on_a_hand_worked_example(self) -> None:
        truth = [0.0, 10.0, 20.0]
        predicted = [3.0, 10.0, 16.0]  # errors +3, 0, -4

        scores = metrics.score(truth, predicted)

        assert scores.n == 3
        assert scores.mae_s == pytest_approx(7 / 3)
        assert scores.rmse_s == pytest_approx((25 / 3) ** 0.5)

    def test_a_perfect_prediction_scores_zero(self) -> None:
        scores = metrics.score([0.0, 5.0, 90.0], [0.0, 5.0, 90.0])

        assert scores.mae_s == 0.0
        assert scores.rmse_s == 0.0

    def test_seconds_convert_to_minutes(self) -> None:
        scores = metrics.score([0.0], [120.0])

        assert scores.mae_s == 120.0
        assert scores.mae_min == 2.0

    def test_mismatched_shapes_are_rejected(self) -> None:
        try:
            metrics.score([1.0, 2.0], [1.0])
        except ValueError as exc:
            assert "shape mismatch" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("expected a ValueError")

    def test_empty_input_does_not_divide_by_zero(self) -> None:
        scores = metrics.score([], [])

        assert scores.n == 0
        assert np.isnan(scores.mae_s)


class TestMapeIsGuarded:
    """The specified metric, made safe rather than silently dropped."""

    def test_zero_truths_are_excluded_and_the_coverage_is_reported(self) -> None:
        truth = [0.0, 0.0, 100.0, 200.0]
        predicted = [10.0, 20.0, 150.0, 100.0]  # 50% and 50% on the non-zero pair

        scores = metrics.score(truth, predicted)

        assert scores.mape_coverage == 0.5
        assert scores.mape_nonzero == pytest_approx(50.0)

    def test_an_all_zero_target_yields_no_mape_rather_than_infinity(self) -> None:
        """More than half this project's rows are exactly zero, so this is the
        realistic degenerate case, not a contrived one."""
        scores = metrics.score([0.0, 0.0, 0.0], [5.0, 1.0, 0.0])

        assert scores.mape_nonzero is None
        assert scores.mape_coverage == 0.0
        assert np.isfinite(scores.mae_s)

    def test_the_printed_line_says_n_a_when_mape_is_undefined(self) -> None:
        scores = metrics.score([0.0, 0.0], [1.0, 2.0])

        assert "n/a" in scores.line("all on time")


class TestMase:
    def test_below_one_means_the_model_beats_the_baseline(self) -> None:
        model = metrics.score([0.0, 100.0], [0.0, 90.0])  # MAE 5
        baseline = metrics.score([0.0, 100.0], [0.0, 80.0])  # MAE 10

        assert metrics.mase(model, baseline) == pytest_approx(0.5)

    def test_above_one_means_it_loses(self) -> None:
        model = metrics.score([0.0, 100.0], [0.0, 80.0])
        baseline = metrics.score([0.0, 100.0], [0.0, 90.0])

        assert metrics.mase(model, baseline) > 1

    def test_a_perfect_baseline_does_not_divide_by_zero(self) -> None:
        perfect = metrics.score([10.0], [10.0])
        worse = metrics.score([10.0], [12.0])

        assert metrics.mase(worse, perfect) == float("inf")


class TestBySegment:
    def test_splits_scores_and_orders_worst_first(self) -> None:
        segment = pd.Series(["peak", "peak", "off", "off"])
        truth = [0.0, 0.0, 0.0, 0.0]
        predicted = [1.0, 1.0, 50.0, 50.0]

        frame = metrics.by_segment(segment, truth, predicted)

        assert list(frame["segment"]) == ["off", "peak"]
        assert list(frame["n"]) == [2, 2]
        assert frame.loc[0, "mae_s"] > frame.loc[1, "mae_s"]


class TestResidualSummary:
    def test_reports_spread_not_just_a_midpoint(self) -> None:
        """Delay is right-skewed, so MAE alone can hide a long tail."""
        truth = np.zeros(100)
        predicted = np.concatenate([np.zeros(95), np.full(5, 600.0)])

        line = metrics.residual_summary(truth, predicted)

        assert "p95" in line and "median" in line

    def test_empty_input_is_a_sentence_not_a_crash(self) -> None:
        assert "no rows" in metrics.residual_summary([], [])


def pytest_approx(value: float) -> object:
    """Local alias so the intent reads clearly in the assertions above."""
    import pytest

    return pytest.approx(value)
