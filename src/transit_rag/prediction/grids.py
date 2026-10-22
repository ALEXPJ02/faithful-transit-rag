"""XGBoost's hyperparameter grids, translated for LightGBM and CatBoost (``docs/18`` §2).

Both comparisons, detection and delay, derive their LightGBM and CatBoost grids
from XGBoost's through these two functions, so no library is searched harder than
another, and the translation lives in one place. Review found it mistranslated
twice while it was copied by hand.

They match capacity on the parameters the libraries share. Each library keeps its
own design otherwise: LightGBM's rules for splitting a category, and CatBoost's
ordered target statistics, are part of what the comparison compares.
"""

from __future__ import annotations

from typing import Any


def for_lightgbm(xgboost_grid: tuple[dict[str, Any], ...]) -> tuple[dict[str, Any], ...]:
    """Each XGBoost candidate, with the two leaf settings LightGBM needs.

    - ``num_leaves`` is 2^depth, every leaf a depth-wise tree of that depth can
      hold, so leaf-wise growth gets the same room and no more.
    - ``min_child_weight`` already means the same in both libraries: the least
      hessian a leaf may hold. ``min_child_samples``, LightGBM's own row floor, is
      set to the same number. A row's hessian is at most 1, under absolute error
      and log loss alike, so a leaf that meets the hessian floor meets the row
      floor too, and the row floor never binds.
    """
    return tuple(
        {
            **candidate,
            "num_leaves": 2 ** candidate["max_depth"],
            "min_child_samples": candidate["min_child_weight"],
        }
        for candidate in xgboost_grid
    )


def for_catboost(xgboost_grid: tuple[dict[str, Any], ...]) -> tuple[dict[str, Any], ...]:
    """Each XGBoost candidate's depth and learning rate.

    CatBoost grows symmetric trees, where a minimum leaf size does not apply.
    """
    return tuple(
        {"depth": candidate["max_depth"], "learning_rate": candidate["learning_rate"]}
        for candidate in xgboost_grid
    )
