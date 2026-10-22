"""Tests for translating XGBoost's grids for LightGBM and CatBoost (``docs/18`` §2).

Both comparisons derive their grids here, so this is where the translation rule is
pinned. Review found it mistranslated twice while it was copied by hand.
"""

from __future__ import annotations

from transit_rag.prediction.grids import for_catboost, for_lightgbm

XGBOOST = (
    {"max_depth": 3, "learning_rate": 0.05, "min_child_weight": 5},
    {"max_depth": 8, "learning_rate": 0.10, "min_child_weight": 20},
)


def test_lightgbm_keeps_xgboost_s_settings_and_adds_its_leaf_settings() -> None:
    assert for_lightgbm(XGBOOST) == (
        {"max_depth": 3, "learning_rate": 0.05, "min_child_weight": 5, "num_leaves": 8, "min_child_samples": 5},
        {"max_depth": 8, "learning_rate": 0.10, "min_child_weight": 20, "num_leaves": 256, "min_child_samples": 20},
    )  # fmt: skip


def test_a_leaf_wise_tree_gets_every_leaf_a_depth_wise_tree_can_hold() -> None:
    """2^depth, not 2^depth - 1: a tree of depth d has 2^d leaves and 2^d - 1 splits."""
    assert [g["num_leaves"] for g in for_lightgbm(XGBOOST)] == [2**3, 2**8]


def test_catboost_takes_depth_and_learning_rate_alone() -> None:
    assert for_catboost(XGBOOST) == (
        {"depth": 3, "learning_rate": 0.05},
        {"depth": 8, "learning_rate": 0.10},
    )


def test_the_xgboost_grid_is_left_as_it_was() -> None:
    """The scored run's grid is frozen; translating it must not change it."""
    before = [dict(candidate) for candidate in XGBOOST]
    for_lightgbm(XGBOOST)
    for_catboost(XGBOOST)
    assert [dict(candidate) for candidate in XGBOOST] == before
