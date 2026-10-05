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
        """Each feature's share of importance, largest first."""
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
    return model


_FITTERS = {"xgboost": (_fit_xgboost, XGB_GRID), "random forest": (_fit_forest, FOREST_GRID)}


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
