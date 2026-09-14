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
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from transit_rag.prediction.model.dataset import Dataset

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
