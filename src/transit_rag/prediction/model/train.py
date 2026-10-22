"""Fitting the XGBoost regressor, and choosing its hyperparameters honestly.

XGBoost over an LSTM is settled (``docs/02-tech-stack.md`` s3, after Sarhani &
Voss 2024). What was never specified anywhere in the project is *how the
hyperparameters get chosen*, which for a research subject is a required methods
sentence rather than an implementation detail.

The procedure here: a small explicit grid, selected on the **validation** split
only, with early stopping also driven by validation. The test split is not
touched during fitting or selection -- it is scored once, at the end, by the
caller. The grid is deliberately small; with a collection window of weeks the
honest constraint is data, not search budget, and a large sweep over a thin
validation set mostly selects noise.

**LightGBM and CatBoost were added after the scored run**, at the supervisor's
request (``docs/18``). They are chosen the same way, on a grid of the same size
that mirrors :data:`PARAM_GRID`'s depths and learning rates, and they never
replace the XGBoost model the agent reads.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from transit_rag.prediction.model import metrics
from transit_rag.prediction.model.dataset import CATEGORICAL_COLUMNS, Dataset

log = logging.getLogger(__name__)

#: Fixed across every candidate. ``hist`` handles the categorical features
#: natively, which is why the feature matrix uses pandas category dtype rather
#: than one-hot expanding 308 stop ids into 308 columns.
BASE_PARAMS: dict[str, Any] = {
    "objective": "reg:absoluteerror",  # matches MAE, the reported headline
    "tree_method": "hist",
    "enable_categorical": True,
    "n_estimators": 2000,  # an upper bound; early stopping picks the real number
    "early_stopping_rounds": 50,
    "random_state": 42,
    "n_jobs": -1,
}

#: The search. Small on purpose -- see the module docstring.
PARAM_GRID: tuple[dict[str, Any], ...] = (
    {"max_depth": 4, "learning_rate": 0.10, "min_child_weight": 5},
    {"max_depth": 6, "learning_rate": 0.10, "min_child_weight": 5},
    {"max_depth": 6, "learning_rate": 0.05, "min_child_weight": 20},
    {"max_depth": 8, "learning_rate": 0.05, "min_child_weight": 20},
)

LGBM_BASE: dict[str, Any] = {
    "objective": "l1",  # MAE, as reg:absoluteerror is for XGBoost
    "n_estimators": 2000,
    "random_state": 42,
    "n_jobs": -1,
    # Both are needed for a refit to reproduce bit for bit (docs/08 §3.5).
    "deterministic": True,
    "force_row_wise": True,
    "verbose": -1,
}
#: :data:`PARAM_GRID`, translated as the detection grid is (``disruption/models.py``),
#: so the two cannot drift: ``num_leaves`` is 2^depth, every leaf a depth-wise tree
#: of that depth can hold, and ``min_child_weight`` is the least hessian a leaf may
#: hold in both libraries. Under an absolute-error loss both give every row a
#: hessian of 1, so here it is also a row count, and ``min_child_samples`` set to
#: the same number matches it.
LGBM_GRID: tuple[dict[str, Any], ...] = tuple(
    {
        **candidate,
        "num_leaves": 2 ** candidate["max_depth"],
        "min_child_samples": candidate["min_child_weight"],
    }
    for candidate in PARAM_GRID
)

CATBOOST_BASE: dict[str, Any] = {
    "loss_function": "MAE",
    "iterations": 2000,
    "early_stopping_rounds": 50,
    "random_seed": 42,
    # Otherwise every fit writes a catboost_info/ folder into the working directory.
    "allow_writing_files": False,
    "verbose": False,
}
#: CatBoost grows symmetric trees, where a minimum leaf size does not apply, so
#: only depth and learning rate carry over.
CATBOOST_GRID: tuple[dict[str, Any], ...] = tuple(
    {"depth": candidate["max_depth"], "learning_rate": candidate["learning_rate"]}
    for candidate in PARAM_GRID
)

#: What a missing line or stop becomes for CatBoost, which rejects a missing
#: category where XGBoost and LightGBM take it natively.
MISSING_CATEGORY = "<missing>"


@dataclass
class FittedModel:
    """A fitted regressor plus the record of how it was chosen."""

    booster: Any
    params: dict[str, Any]
    best_iteration: int
    validation_mae_s: float
    #: Every candidate tried, so the write-up can state the search, not just the winner.
    search: list[dict[str, Any]] = field(default_factory=list)

    def predict(self, features: pd.DataFrame) -> pd.Series:
        return pd.Series(self.booster.predict(features), index=features.index, dtype=float)

    def feature_importance(self) -> pd.DataFrame:
        """Gain-based importance, most important first.

        Worth reading rather than filing: if ``prev_stop_delay_s`` dominates
        overwhelmingly, the model is close to an elaborate restatement of naive
        persistence, and that weakens the case for including a prediction layer
        at all.
        """
        if not hasattr(self.booster, "get_booster"):
            raise TypeError("gain importance is read from an XGBoost booster")
        booster = self.booster.get_booster()
        gains = booster.get_score(importance_type="gain")
        frame = pd.DataFrame(
            {"feature": list(gains.keys()), "gain": list(gains.values())},
        )
        if frame.empty:
            return frame
        frame["share"] = frame["gain"] / frame["gain"].sum()
        return frame.sort_values("gain", ascending=False).reset_index(drop=True)


def train(train_set: Dataset, validation_set: Dataset) -> FittedModel:
    """Fit each candidate, keep the one with the best validation MAE.

    Selection and early stopping both look only at ``validation_set``. The test
    split must not be passed to this function.
    """
    from xgboost import XGBRegressor

    best: FittedModel | None = None
    search: list[dict[str, Any]] = []

    for candidate in PARAM_GRID:
        params = {**BASE_PARAMS, **candidate}
        model = XGBRegressor(**params)
        model.fit(
            train_set.features,
            train_set.target,
            eval_set=[(validation_set.features, validation_set.target)],
            verbose=False,
        )

        # XGBoost's own eval metric for reg:absoluteerror is MAE, so the last
        # recorded score at the best iteration is the validation MAE directly.
        results = model.evals_result()["validation_0"]
        metric_name = next(iter(results))
        validation_mae = float(results[metric_name][model.best_iteration])

        record = {
            **candidate,
            "best_iteration": model.best_iteration,
            "validation_mae_s": validation_mae,
        }
        search.append(record)
        log.info(
            "  depth=%s lr=%s min_child_weight=%s -> validation MAE %.1fs (%d trees)",
            candidate["max_depth"],
            candidate["learning_rate"],
            candidate["min_child_weight"],
            validation_mae,
            model.best_iteration,
        )

        if best is None or validation_mae < best.validation_mae_s:
            best = FittedModel(
                booster=model,
                params=params,
                best_iteration=int(model.best_iteration),
                validation_mae_s=validation_mae,
            )

    if best is None:  # pragma: no cover - PARAM_GRID is never empty
        raise RuntimeError("PARAM_GRID is empty; nothing to train")

    best.search = search
    return best


def _select(
    grid: tuple[dict[str, Any], ...],
    base: dict[str, Any],
    fit: Callable[[dict[str, Any]], tuple[Any, int]],
    validation_set: Dataset,
) -> FittedModel:
    """Fit each candidate and keep the best validation MAE.

    The MAE is taken from each candidate's predictions, so it is measured the same
    way for every library. :func:`train` reads XGBoost's own float32 eval metric
    instead, and stays as it is because the scored run used it. The comparison
    table re-measures XGBoost this way, so its column is one measure throughout.

    ``fit`` returns the model and its best iteration counted from 0, as XGBoost's is.
    """
    best: FittedModel | None = None
    search: list[dict[str, Any]] = []
    for candidate in grid:
        params = {**base, **candidate}
        model, best_iteration = fit(params)
        predicted = model.predict(validation_set.features)
        validation_mae = metrics.score(validation_set.target, predicted).mae_s
        search.append(
            {**candidate, "best_iteration": best_iteration, "validation_mae_s": validation_mae}
        )
        log.info(
            "  %s -> validation MAE %.1fs (%d trees)", candidate, validation_mae, best_iteration + 1
        )
        if best is None or validation_mae < best.validation_mae_s:
            best = FittedModel(model, params, best_iteration, validation_mae)
    if best is None:  # pragma: no cover - the grids are never empty
        raise RuntimeError("the grid is empty; nothing to train")
    best.search = search
    return best


def train_lightgbm(train_set: Dataset, validation_set: Dataset) -> FittedModel:
    """LightGBM by the same rule as :func:`train`. Never pass test.

    It reads the pandas category dtype natively, so it sees the same shared
    levels as XGBoost.
    """
    import lightgbm

    def fit(params: dict[str, Any]) -> tuple[Any, int]:
        model = lightgbm.LGBMRegressor(**params)
        model.fit(
            train_set.features,
            train_set.target,
            eval_X=validation_set.features,
            eval_y=validation_set.target,
            callbacks=[lightgbm.early_stopping(50, verbose=False)],
        )
        # LightGBM counts iterations from 1.
        return model, int(model.best_iteration_) - 1

    return _select(LGBM_GRID, LGBM_BASE, fit, validation_set)


def train_catboost(train_set: Dataset, validation_set: Dataset) -> FittedModel:
    """CatBoost by the same rule as :func:`train`. Never pass test.

    The categorical columns are named, so CatBoost encodes them by its own
    ordered target statistics rather than reading their codes as numbers.
    """
    from catboost import CatBoostRegressor

    def fit(params: dict[str, Any]) -> tuple[Any, int]:
        model = CatBoostRegressor(**params, cat_features=list(CATEGORICAL_COLUMNS))
        model.fit(
            _named_categories(train_set.features),
            train_set.target,
            eval_set=(_named_categories(validation_set.features), validation_set.target),
        )
        return _NamedCategories(model), int(model.get_best_iteration())

    return _select(CATBOOST_GRID, CATBOOST_BASE, fit, validation_set)


def _named_categories(features: pd.DataFrame) -> pd.DataFrame:
    """The categorical columns as strings, with a missing value named :data:`MISSING_CATEGORY`."""
    named = features.copy()
    for column in CATEGORICAL_COLUMNS:
        values = named[column].astype(object)
        named[column] = values.where(values.notna(), MISSING_CATEGORY)
    return named


class _NamedCategories:
    """A fitted CatBoost that names missing categories when it predicts, as it did to fit."""

    def __init__(self, model: Any) -> None:
        self.model = model

    def predict(self, features: pd.DataFrame) -> Any:
        return self.model.predict(_named_categories(features))
