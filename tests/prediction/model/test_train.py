"""Training tests, including a deliberate leak.

The leak test is the point of this file. Declaring that a column is forbidden is
cheap; demonstrating that including it would have produced a spectacular and
entirely false score is what makes the guard meaningful.

Small on purpose: a few hundred synthetic rows, not the real 93,000.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from transit_rag.prediction.model import dataset as dataset_module
from transit_rag.prediction.model import metrics
from transit_rag.prediction.model.dataset import (
    FEATURE_COLUMNS,
    build_dataset,
    category_dtypes,
)
from transit_rag.prediction.model.train import (
    CATBOOST_GRID,
    LGBM_GRID,
    PARAM_GRID,
    FittedModel,
    train,
    train_catboost,
    train_lightgbm,
)

pytest.importorskip("xgboost", reason="install the 'prediction' extra")

TRAIN_DATES = 6


def synthetic_table(rows: int = 400, seed: int = 0) -> pd.DataFrame:
    """Delay is a noisy function of the previous stop's delay, as in real data.

    The noise term matters: it sets a floor below which no honest model can go,
    which is what makes the leak test's collapse unambiguous.
    """
    rng = np.random.default_rng(seed)
    previous = rng.normal(60, 40, rows).round()
    delay = (previous * 0.8 + rng.normal(0, 20, rows)).round()
    return pd.DataFrame(
        {
            "service_date": [f"2026-09-{3 + i % 8:02d}" for i in range(rows)],
            "trip_id": [f"trip-{i}" for i in range(rows)],
            "stop_id": [f"stop-{i % 5}" for i in range(rows)],
            "route_id": ["NSN_1a"] * rows,
            "route_short_name": rng.choice(["T1", "T4"], rows),
            "stop_sequence": rng.integers(1, 30, rows),
            "scheduled_arrival_s": rng.integers(20000, 80000, rows),
            "delay_s": delay,
            "arrival_delay_s": delay,  # reconcile.py copies this into the target
            "departure_delay_s": delay,
            "prev_stop_delay_s": previous,
            "stops_ahead_final": rng.integers(0, 2, rows),
            "observation_count": rng.integers(1, 6, rows),
            "observed_at_utc": ["2026-09-03T00:00:00+00:00"] * rows,
            "hour_local": rng.integers(0, 24, rows),
            "day_of_week": rng.integers(0, 7, rows),
            "is_weekend": rng.random(rows) > 0.7,
            "is_peak": rng.random(rows) > 0.6,
            "schedule_matched": [True] * rows,
        }
    )


def split_sets(table: pd.DataFrame) -> tuple[dataset_module.Dataset, dataset_module.Dataset]:
    """Fit set and held-out set, sharing one set of category levels."""
    levels = category_dtypes(table)
    dates = sorted(table["service_date"].unique())
    fit_rows = table[table["service_date"].isin(dates[:TRAIN_DATES])]
    held_rows = table[table["service_date"].isin(dates[TRAIN_DATES:])]
    return build_dataset(fit_rows, levels), build_dataset(held_rows, levels)


def held_out_mae(table: pd.DataFrame) -> float:
    """Train and score, using whatever FEATURE_COLUMNS currently says."""
    fit_set, held_set = split_sets(table)
    fitted = train(fit_set, held_set)
    return metrics.score(held_set.target, fitted.predict(held_set.features)).mae_s


class TestTrain:
    def test_selects_the_best_candidate_on_validation(self) -> None:
        fit_set, validation_set = split_sets(synthetic_table())

        fitted = train(fit_set, validation_set)

        assert fitted.validation_mae_s > 0
        assert len(fitted.search) == len(PARAM_GRID)
        # The winner is the best of what was tried, not merely the last.
        assert fitted.validation_mae_s == min(row["validation_mae_s"] for row in fitted.search)

    def test_records_every_candidate_so_the_search_can_be_written_up(self) -> None:
        fit_set, validation_set = split_sets(synthetic_table())

        fitted = train(fit_set, validation_set)

        for record in fitted.search:
            assert {"max_depth", "learning_rate", "validation_mae_s"} <= set(record)

    def test_feature_importance_covers_only_permitted_features(self) -> None:
        fit_set, validation_set = split_sets(synthetic_table())

        importance = train(fit_set, validation_set).feature_importance()

        assert set(importance["feature"]) <= set(FEATURE_COLUMNS)
        assert importance["share"].sum() == pytest.approx(1.0)


class TestTheAddedRegressors:
    """docs/18: LightGBM and CatBoost, chosen by the same rule as XGBoost."""

    def test_their_grids_are_the_ones_frozen_before_the_run(self) -> None:
        """Derived from PARAM_GRID, so they cannot drift from it; pinned as the record."""
        assert LGBM_GRID == (
            {"max_depth": 4, "learning_rate": 0.10, "min_child_weight": 5, "num_leaves": 16, "min_child_samples": 5},
            {"max_depth": 6, "learning_rate": 0.10, "min_child_weight": 5, "num_leaves": 64, "min_child_samples": 5},
            {"max_depth": 6, "learning_rate": 0.05, "min_child_weight": 20, "num_leaves": 64, "min_child_samples": 20},
            {"max_depth": 8, "learning_rate": 0.05, "min_child_weight": 20, "num_leaves": 256, "min_child_samples": 20},
        )  # fmt: skip
        assert CATBOOST_GRID == (
            {"depth": 4, "learning_rate": 0.10},
            {"depth": 6, "learning_rate": 0.10},
            {"depth": 6, "learning_rate": 0.05},
            {"depth": 8, "learning_rate": 0.05},
        )

    def test_catboost_takes_a_missing_stop_as_xgboost_does(self) -> None:
        """CatBoost raises on a missing category; reconciliation keeps unmatched trips."""
        pytest.importorskip("catboost", reason="install the 'prediction' extra")
        table = synthetic_table()
        table.loc[[3, 6], "stop_id"] = None  # one fitted on (2026-09-06), one held out (09-09)
        fit_set, validation_set = split_sets(table)
        assert fit_set.features["stop_id"].isna().any()
        assert validation_set.features["stop_id"].isna().any()

        predicted = train_catboost(fit_set, validation_set).predict(validation_set.features)

        assert np.isfinite(np.asarray(predicted, dtype=float)).all()

    @pytest.mark.parametrize(
        ("fit", "grid", "library"),
        [(train_lightgbm, LGBM_GRID, "lightgbm"), (train_catboost, CATBOOST_GRID, "catboost")],
    )
    def test_chosen_on_validation_from_a_grid_the_size_of_xgboost_s(
        self, fit: Any, grid: tuple[dict[str, Any], ...], library: str
    ) -> None:
        pytest.importorskip(library, reason="install the 'prediction' extra")
        fit_set, validation_set = split_sets(synthetic_table())

        fitted: FittedModel = fit(fit_set, validation_set)

        assert len(fitted.search) == len(grid) == len(PARAM_GRID)
        assert fitted.validation_mae_s == min(row["validation_mae_s"] for row in fitted.search)
        assert fitted.best_iteration >= 0
        # The rows it was chosen on use the shared stop levels, so prediction
        # must accept them; and it must beat predicting the training median.
        predicted = fitted.predict(validation_set.features)
        constant = np.full(len(validation_set), fit_set.target.median())
        assert metrics.score(validation_set.target, predicted).mae_s < (
            metrics.score(validation_set.target, constant).mae_s
        )

    def test_only_xgboost_reports_gain_importance(self) -> None:
        pytest.importorskip("lightgbm", reason="install the 'prediction' extra")
        fit_set, validation_set = split_sets(synthetic_table())
        with pytest.raises(TypeError, match="XGBoost booster"):
            train_lightgbm(fit_set, validation_set).feature_importance()


class TestTheLeakGuardActuallyMatters:
    """Proof that the forbidden list is doing work.

    ``delay_s`` here is a noisy function of ``prev_stop_delay_s``, so an honest
    model cannot beat the noise floor. Handing it ``arrival_delay_s`` -- which
    reconcile.py copies straight into the target -- collapses the error to
    nearly nothing. If this stops showing a dramatic gap, the guard has stopped
    guarding and a leaked score would be believed.
    """

    def test_leaking_the_target_collapses_the_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        table = synthetic_table()

        honest = held_out_mae(table)

        # The only change is the column list; same data, same split, same fit.
        monkeypatch.setattr(
            dataset_module, "FEATURE_COLUMNS", (*FEATURE_COLUMNS, "arrival_delay_s")
        )
        monkeypatch.setattr(
            dataset_module,
            "FORBIDDEN_COLUMNS",
            {k: v for k, v in dataset_module.FORBIDDEN_COLUMNS.items() if k != "arrival_delay_s"},
        )
        leaked = held_out_mae(table)

        assert honest > 5, f"the honest model should sit at the noise floor, got {honest:.1f}"
        assert leaked < honest / 3, (
            f"leaking the target should collapse MAE, but honest={honest:.1f} "
            f"leaked={leaked:.1f} -- the guard may no longer be meaningful"
        )
