"""The detectors RQ2 compares: XGBoost and a random forest on the same features.

``docs/08`` §3.3 names them, after Cottreau et al. (2025). Both see
:data:`~transit_rag.prediction.disruption.features.FEATURE_COLUMNS` and nothing
else, and both are chosen the way the delay model is (``model/train.py``):

- a small explicit grid, because with a few weeks of data the binding constraint
  is data, not search budget;
- selection by **average precision on validation**, the same headline the test
  is scored on;
- a threshold chosen on validation;
- test touched once, by the caller.

Positives are about 6% of windows. Neither model is reweighted to "fix" that.
Average precision does not need it, and reweighting changes what a score of 0.7
means without making the ranking any better.

**LightGBM, CatBoost and logistic regression were added after the scored run**, at
the supervisor's request (``docs/18``). They are a comparison disclosed beside the
result, not part of it, so :data:`SCORED_DETECTORS` stays what was scored. Each is
chosen by the same rule on a grid of the same size: the boosted grids mirror
XGBoost's depths and learning rate, so no boosting library gets more search than
another.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from transit_rag.prediction.disruption.detection import average_precision, best_f1_threshold
from transit_rag.prediction.disruption.features import FEATURE_COLUMNS

#: The lines in a fixed order, so the code a model sees for T4 never depends
#: on which line happens to come first in a split.
LINES: tuple[str, ...] = ("T1", "T4")

XGB_GRID: tuple[dict[str, Any], ...] = (
    {"max_depth": 3, "learning_rate": 0.05, "min_child_weight": 5},
    {"max_depth": 4, "learning_rate": 0.05, "min_child_weight": 10},
    {"max_depth": 6, "learning_rate": 0.05, "min_child_weight": 20},
)
XGB_BASE: dict[str, Any] = {
    "objective": "binary:logistic",
    "eval_metric": "aucpr",  # the selection metric, so early stopping agrees with it
    "tree_method": "hist",
    "n_estimators": 1000,
    "early_stopping_rounds": 50,
    "random_state": 42,
    "n_jobs": -1,
}

FOREST_GRID: tuple[dict[str, Any], ...] = (
    {"max_depth": 6, "min_samples_leaf": 5},
    {"max_depth": 10, "min_samples_leaf": 10},
    {"max_depth": None, "min_samples_leaf": 20},
)
FOREST_BASE: dict[str, Any] = {"n_estimators": 500, "random_state": 42, "n_jobs": -1}

#: XGBoost's candidates, translated:
#:
#: - ``num_leaves`` is 2^depth, every leaf a depth-wise tree of that depth can hold,
#:   so leaf-wise growth gets the same room and no more.
#: - ``min_child_weight`` means the same in both libraries: the least hessian a
#:   leaf may hold. Under log loss a row adds p(1 - p), at most 0.25, so it is not
#:   a row count. ``min_child_samples``, LightGBM's own row floor, is set to the
#:   same number, which those hessian floors already exceed, so it never binds.
LGBM_GRID: tuple[dict[str, Any], ...] = (
    {
        "max_depth": 3,
        "num_leaves": 8,
        "learning_rate": 0.05,
        "min_child_weight": 5,
        "min_child_samples": 5,
    },
    {
        "max_depth": 4,
        "num_leaves": 16,
        "learning_rate": 0.05,
        "min_child_weight": 10,
        "min_child_samples": 10,
    },
    {
        "max_depth": 6,
        "num_leaves": 64,
        "learning_rate": 0.05,
        "min_child_weight": 20,
        "min_child_samples": 20,
    },
)
LGBM_BASE: dict[str, Any] = {
    "objective": "binary",
    # The only metric, so early stopping agrees with selection. Passed as an extra
    # metric instead, binary log loss would still run and could stop the fit.
    "metric": "average_precision",
    "n_estimators": 1000,
    "random_state": 42,
    "n_jobs": -1,
    # Both are needed for a refit to reproduce bit for bit (docs/08 §3.5).
    "deterministic": True,
    "force_row_wise": True,
    "importance_type": "gain",
    "verbose": -1,
}

CATBOOST_GRID: tuple[dict[str, Any], ...] = (
    {"depth": 3, "learning_rate": 0.05},
    {"depth": 4, "learning_rate": 0.05},
    {"depth": 6, "learning_rate": 0.05},
)
CATBOOST_BASE: dict[str, Any] = {
    "loss_function": "Logloss",
    "eval_metric": "PRAUC",  # the selection metric, so early stopping agrees with it
    "iterations": 1000,
    "early_stopping_rounds": 50,
    "random_seed": 42,
    # Otherwise every fit writes a catboost_info/ folder into the working directory.
    "allow_writing_files": False,
    "verbose": False,
}

#: Inverse regularisation strength. Strong to weak, as the tree grids run shallow
#: to deep.
LOGISTIC_GRID: tuple[dict[str, Any], ...] = ({"C": 0.01}, {"C": 0.1}, {"C": 1.0})

#: What RQ2's scored run compared, and what ``transit-detect`` fits by default.
SCORED_DETECTORS: tuple[str, ...] = ("xgboost", "random forest")
#: The post-hoc comparison of ``docs/18``, fitted only under ``--compare``.
ADDED_DETECTORS: tuple[str, ...] = ("lightgbm", "catboost", "logistic regression")


def feature_matrix(table: pd.DataFrame, columns: Sequence[str] = FEATURE_COLUMNS) -> pd.DataFrame:
    """The permitted columns as numbers. The line becomes 0/1 in a fixed order.

    Missing values stay NaN. Both models split on missingness natively, and a
    thin window's missing delay is information, not a gap to fill.
    """
    matrix = table.loc[:, list(columns)].copy()
    if "line" in matrix:
        matrix["line"] = matrix["line"].map({line: code for code, line in enumerate(LINES)})
    return matrix.astype(float)


def known(table: pd.DataFrame) -> pd.DataFrame:
    """Rows whose target is known: the only rows anything is fitted or scored on."""
    return table[table["target"].notna()]


@dataclass
class FittedDetector:
    """A fitted detector, how it was chosen, and the threshold validation set."""

    name: str
    model: Any
    columns: tuple[str, ...]
    params: dict[str, Any]
    validation_ap: float
    threshold: float
    search: list[dict[str, Any]] = field(default_factory=list)

    def score(self, table: pd.DataFrame) -> pd.Series:
        """Probability of disruption in the next 30 minutes, one per row."""
        probabilities = self.model.predict_proba(feature_matrix(table, self.columns))[:, 1]
        return pd.Series(probabilities, index=table.index, dtype=float)

    def importance(self) -> pd.Series:
        """Each feature's share of importance, largest first. Tree models only.

        Each library defines importance its own way (XGBoost and LightGBM by gain,
        the forest by impurity, CatBoost by how far predictions move), so shares are
        comparable within a model, never across models.
        """
        if not hasattr(self.model, "feature_importances_"):
            raise TypeError(f"{self.name} has no tree importances to report")
        values = np.asarray(self.model.feature_importances_, dtype=float)
        total = values.sum() or 1.0
        return pd.Series(values / total, index=list(self.columns)).sort_values(ascending=False)


def _fit_xgboost(
    train: pd.DataFrame, validation: pd.DataFrame, columns: Sequence[str], params: dict[str, Any]
) -> Any:
    from xgboost import XGBClassifier

    model = XGBClassifier(**{**XGB_BASE, **params})
    model.fit(
        feature_matrix(train, columns),
        train["target"].astype(bool),
        eval_set=[(feature_matrix(validation, columns), validation["target"].astype(bool))],
        verbose=False,
    )
    return model


def _fit_forest(
    train: pd.DataFrame, validation: pd.DataFrame, columns: Sequence[str], params: dict[str, Any]
) -> Any:
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(**{**FOREST_BASE, **params})
    model.fit(feature_matrix(train, columns), train["target"].astype(bool))
    # Fitted in parallel, predicted on one thread. Parallel prediction sums the
    # trees' probabilities in whatever order threads finish, so a score could
    # differ in its last bit between runs, and docs/08 §3.5 needs re-runs to
    # reproduce. The fit itself is deterministic under its random_state.
    model.set_params(n_jobs=1)
    return model


def _fit_lightgbm(
    train: pd.DataFrame, validation: pd.DataFrame, columns: Sequence[str], params: dict[str, Any]
) -> Any:
    import lightgbm

    model = lightgbm.LGBMClassifier(**{**LGBM_BASE, **params})
    model.fit(
        feature_matrix(train, columns),
        train["target"].astype(bool),
        eval_X=feature_matrix(validation, columns),
        eval_y=validation["target"].astype(bool),
        callbacks=[lightgbm.early_stopping(50, verbose=False)],
    )
    return model


def _fit_catboost(
    train: pd.DataFrame, validation: pd.DataFrame, columns: Sequence[str], params: dict[str, Any]
) -> Any:
    from catboost import CatBoostClassifier

    model = CatBoostClassifier(**{**CATBOOST_BASE, **params})
    model.fit(
        feature_matrix(train, columns),
        train["target"].astype(bool),
        eval_set=(feature_matrix(validation, columns), validation["target"].astype(bool)),
    )
    return model


def _fit_logistic(
    train: pd.DataFrame, validation: pd.DataFrame, columns: Sequence[str], params: dict[str, Any]
) -> Any:
    """A linear reference: whether the tree ensembles earn their complexity.

    A linear model cannot split on a missing value, so missing values are filled
    with the training median and each gets an indicator column. The indicator
    carries what the trees read from NaN: a thin window is information, not a gap.
    Scaling puts the penalty on every feature equally.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        LogisticRegression(max_iter=2000, **params),
    )
    model.fit(feature_matrix(train, columns), train["target"].astype(bool))
    return model


_FITTERS = {
    "xgboost": (_fit_xgboost, XGB_GRID),
    "random forest": (_fit_forest, FOREST_GRID),
    "lightgbm": (_fit_lightgbm, LGBM_GRID),
    "catboost": (_fit_catboost, CATBOOST_GRID),
    "logistic regression": (_fit_logistic, LOGISTIC_GRID),
}


def fit_detector(
    name: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    columns: Sequence[str] = FEATURE_COLUMNS,
) -> FittedDetector:
    """Fit each candidate on train and keep the best validation AP. Never pass test."""
    fitter, grid = _FITTERS[name]
    train, validation = known(train), known(validation)
    if train["target"].astype(bool).nunique() < 2:
        raise ValueError(
            "train holds only one class of target, so there is nothing to learn to separate; "
            "collect more dates"
        )
    if not validation["target"].astype(bool).any():
        raise ValueError("validation holds no positive target, so no candidate can be chosen")

    best: FittedDetector | None = None
    search: list[dict[str, Any]] = []
    for params in grid:
        model = fitter(train, validation, columns, params)
        candidate = FittedDetector(name, model, tuple(columns), params, float("nan"), 0.5)
        scores = candidate.score(validation)
        candidate.validation_ap = average_precision(
            validation["target"].to_numpy(dtype=bool), scores.to_numpy()
        )
        search.append({**params, "validation_ap": candidate.validation_ap})
        if best is None or candidate.validation_ap > best.validation_ap:
            best = candidate

    assert best is not None  # the grids are never empty
    best.threshold = best_f1_threshold(validation, best.score(validation))
    best.search = search
    return best


def save_detector(detector: FittedDetector, path: Path, *, trained_on: dict[str, Any]) -> None:
    """Write a fitted detector, its threshold, and where its training came from.

    ``trained_on`` records the snapshot and the train and validation dates. The
    agent reads it, because a detector asked about a window it was fitted on
    would be reporting what it memorised, not what it detects.
    """
    import joblib

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "name": detector.name,
            "model": detector.model,
            "columns": list(detector.columns),
            "params": detector.params,
            "validation_ap": detector.validation_ap,
            "threshold": detector.threshold,
            "search": detector.search,
            "trained_on": trained_on,
        },
        path,
    )


def load_detector(path: Path) -> tuple[FittedDetector, dict[str, Any]]:
    """Read a detector written by :func:`save_detector`, and its provenance."""
    import joblib

    stored = joblib.load(path)
    detector = FittedDetector(
        name=stored["name"],
        model=stored["model"],
        columns=tuple(stored["columns"]),
        params=stored["params"],
        validation_ap=stored["validation_ap"],
        threshold=stored["threshold"],
        search=stored["search"],
    )
    return detector, dict(stored["trained_on"])
