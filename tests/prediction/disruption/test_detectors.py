"""Tests for the detectors RQ2 compares (``docs/12`` §6).

The models themselves are libraries, so these tests pin the parts of the method
this package owns: the encoding, choosing on validation, and refusing to choose
when validation cannot tell candidates apart.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from transit_rag.prediction.disruption.detection import average_precision
from transit_rag.prediction.disruption.features import FEATURE_COLUMNS
from transit_rag.prediction.disruption.models import (
    FOREST_GRID,
    XGB_GRID,
    feature_matrix,
    fit_detector,
    known,
)


def _table(n: int, seed: int, signal: float = 3.0) -> pd.DataFrame:
    """Windows whose disruption follows share_late, plus noise in everything else."""
    rng = np.random.default_rng(seed)
    share = rng.uniform(0, 0.6, n)
    target = rng.uniform(0, 1, n) < 1 / (1 + np.exp(-signal * (share - 0.35) * 10))
    frame = pd.DataFrame({column: rng.normal(0, 1, n) for column in FEATURE_COLUMNS})
    frame["line"] = rng.choice(["T1", "T4"], n)
    frame["share_late"] = share
    frame["is_weekend"] = rng.uniform(0, 1, n) < 0.3
    frame["is_peak"] = rng.uniform(0, 1, n) < 0.4
    frame["alert_in_feed"] = False
    frame["target"] = pd.array(target, dtype="boolean")
    frame.loc[frame.index[:5], "target"] = pd.NA
    return frame


class TestFeatureMatrix:
    def test_the_line_is_coded_in_a_fixed_order_whatever_comes_first(self) -> None:
        frame = pd.DataFrame({"line": ["T4", "T1"], "n_services": [3, 4]})
        assert list(feature_matrix(frame, ("line", "n_services"))["line"]) == [1.0, 0.0]

    def test_missing_stays_missing_and_flags_become_numbers(self) -> None:
        frame = pd.DataFrame({"share_late": [np.nan, 0.5], "is_peak": [True, False]})
        matrix = feature_matrix(frame, ("share_late", "is_peak"))
        assert np.isnan(matrix["share_late"].iloc[0])
        assert list(matrix["is_peak"]) == [1.0, 0.0]


def test_only_rows_with_a_known_target_are_fitted_or_scored() -> None:
    table = _table(20, seed=0)
    assert len(known(table)) == 15


@pytest.mark.parametrize("name", ["xgboost", "random forest"])
def test_a_detector_is_chosen_on_validation_and_finds_a_planted_signal(name: str) -> None:
    train, validation = _table(600, seed=1), _table(300, seed=2)
    fitted = fit_detector(name, train, validation)

    grid = XGB_GRID if name == "xgboost" else FOREST_GRID
    assert len(fitted.search) == len(grid)
    assert fitted.validation_ap == max(row["validation_ap"] for row in fitted.search)
    assert 0 < fitted.threshold < 1

    fresh = known(_table(300, seed=3))
    ap = average_precision(fresh["target"].to_numpy(dtype=bool), fitted.score(fresh).to_numpy())
    base_rate = fresh["target"].astype(bool).mean()
    assert ap > base_rate + 0.2
    assert fitted.importance().index[0] == "share_late"
    assert fitted.importance().sum() == pytest.approx(1.0)


def test_no_candidate_is_chosen_when_validation_has_nothing_to_find() -> None:
    validation = _table(50, seed=4)
    validation["target"] = pd.array([False] * 50, dtype="boolean")
    with pytest.raises(ValueError, match="no positive target"):
        fit_detector("random forest", _table(100, seed=5), validation)


def test_no_detector_is_fitted_when_train_holds_one_class() -> None:
    """XGBoost's own message for this is about inferred class labels; say what to do."""
    train = _table(100, seed=6)
    train["target"] = pd.array([True] * 100, dtype="boolean")
    with pytest.raises(ValueError, match="only one class"):
        fit_detector("xgboost", train, _table(50, seed=7))


def test_a_forest_refitted_gives_bit_identical_scores() -> None:
    """docs/08 §3.5: a re-run must reproduce. Parallel prediction did not, by 1e-16."""
    train, validation = _table(400, seed=8), _table(200, seed=9)
    first = fit_detector("random forest", train, validation)
    second = fit_detector("random forest", train, validation)
    fresh = known(_table(200, seed=10))
    assert np.array_equal(first.score(fresh).to_numpy(), second.score(fresh).to_numpy())
    assert first.threshold == second.threshold
