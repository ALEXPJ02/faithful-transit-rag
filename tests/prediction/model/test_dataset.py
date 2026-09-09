"""Tests for the feature contract.

These are the most important tests in the package. Every other failure shows up
as a bad score; a leaked feature shows up as an excellent one and gets believed.
"""

from __future__ import annotations

import pandas as pd
import pytest

from transit_rag.prediction.model.dataset import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    FORBIDDEN_COLUMNS,
    TARGET,
    LeakageError,
    build_dataset,
    build_features,
    category_dtypes,
    filter_reliable,
)


def make_table(rows: int = 6) -> pd.DataFrame:
    """A training table with every column reconcile.py produces."""
    return pd.DataFrame(
        {
            "service_date": [f"2026-09-{3 + i % 4:02d}" for i in range(rows)],
            "trip_id": [f"trip-{i}" for i in range(rows)],
            "stop_id": [f"20{i % 3}" for i in range(rows)],
            "route_id": ["NSN_1a"] * rows,
            "route_short_name": ["T1", "T4"] * (rows // 2),
            "stop_sequence": list(range(rows)),
            "scheduled_arrival_s": [21600 + 300 * i for i in range(rows)],
            "delay_s": [0, 60, 120, 0, 30, 240][:rows],
            "arrival_delay_s": [0, 60, 120, 0, 30, 240][:rows],
            "departure_delay_s": [0, 60, 120, 0, 30, 240][:rows],
            "prev_stop_delay_s": [None, 30, 90, 0, 15, 200][:rows],
            "stops_ahead_final": [0, 1, 2, 0, 1, 5][:rows],
            "observation_count": [4, 3, 2, 5, 3, 1][:rows],
            "observed_at_utc": ["2026-09-03T00:00:00+00:00"] * rows,
            "hour_local": [6, 7, 8, 16, 17, 23][:rows],
            "day_of_week": [0, 1, 2, 3, 4, 5][:rows],
            "is_weekend": [False, False, False, False, False, True][:rows],
            "is_peak": [True, True, True, True, True, False][:rows],
            "schedule_matched": [True] * rows,
        }
    )


class TestTheContractItself:
    """Properties of the column lists, independent of any data."""

    def test_no_column_is_both_permitted_and_forbidden(self) -> None:
        assert set(FEATURE_COLUMNS) & set(FORBIDDEN_COLUMNS) == set()

    def test_the_target_is_forbidden_as_a_feature(self) -> None:
        assert TARGET in FORBIDDEN_COLUMNS

    @pytest.mark.parametrize("column", ["arrival_delay_s", "departure_delay_s"])
    def test_the_targets_components_are_forbidden(self, column: str) -> None:
        """reconcile.py builds delay_s by coalescing these two, so either one is
        the answer wearing a different name."""
        assert column in FORBIDDEN_COLUMNS

    @pytest.mark.parametrize(
        "column", ["stops_ahead_final", "observation_count", "observed_at_utc", "schedule_matched"]
    )
    def test_collection_artefacts_are_forbidden(self, column: str) -> None:
        """These describe how this project polled, not anything a rider's
        question could carry."""
        assert column in FORBIDDEN_COLUMNS


class TestBuildFeatures:
    def test_produces_exactly_the_permitted_columns(self) -> None:
        features = build_features(make_table())

        assert list(features.columns) == list(FEATURE_COLUMNS)

    def test_no_forbidden_column_survives(self) -> None:
        features = build_features(make_table())

        assert set(features.columns) & set(FORBIDDEN_COLUMNS) == set()

    def test_a_missing_feature_column_is_a_clear_error(self) -> None:
        table = make_table().drop(columns=["prev_stop_delay_s"])

        with pytest.raises(KeyError, match="prev_stop_delay_s"):
            build_features(table)

    def test_leakage_is_raised_not_silently_dropped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """If the two lists are ever edited into disagreement, the build must
        fail loudly rather than quietly training on the answer."""
        monkeypatch.setattr(
            "transit_rag.prediction.model.dataset.FEATURE_COLUMNS",
            (*FEATURE_COLUMNS, "arrival_delay_s"),
        )

        with pytest.raises(LeakageError, match="arrival_delay_s"):
            build_features(make_table())

    def test_categoricals_are_categorical(self) -> None:
        """stop_id is an integer in the table but nothing about it is ordinal."""
        features = build_features(make_table())

        for column in CATEGORICAL_COLUMNS:
            assert isinstance(features[column].dtype, pd.CategoricalDtype)


class TestSharedCategoryLevels:
    """Levels must be fixed once over the whole dataset, before splitting.

    Per-split encoding gives the same stop different integer codes in different
    partitions, which XGBoost rejects outright — and would be worse if it did
    not, because the codes would silently mean different things.
    """

    def test_levels_come_from_the_whole_table(self) -> None:
        table = make_table()
        levels = category_dtypes(table)

        assert set(levels["stop_id"].categories) == set(table["stop_id"].unique())

    def test_a_split_missing_a_category_still_encodes_consistently(self) -> None:
        table = make_table()
        levels = category_dtypes(table)
        # A partition that happens to contain only one of the three stops.
        partition = table[table["stop_id"] == "200"]

        features = build_features(partition, levels)

        assert list(features["stop_id"].cat.categories) == list(levels["stop_id"].categories)

    def test_without_shared_levels_partitions_disagree(self) -> None:
        """The bug this guards against, stated as a test."""
        table = make_table()
        one = build_features(table[table["stop_id"] == "200"])
        another = build_features(table[table["stop_id"] == "201"])

        assert list(one["stop_id"].cat.categories) != list(another["stop_id"].cat.categories)


class TestFilterReliable:
    def test_keeps_only_observations_close_to_the_event(self) -> None:
        table = make_table()

        kept = filter_reliable(table, max_stops_ahead=1)

        assert set(kept["stops_ahead_final"]) <= {0, 1}
        assert len(kept) == 4


class TestBuildDataset:
    def test_carries_service_date_without_making_it_a_feature(self) -> None:
        """Scores are reported per day, but the model must not see the day."""
        dataset = build_dataset(make_table())

        assert "service_date" not in dataset.features.columns
        assert len(dataset.service_date) == len(dataset)

    def test_target_is_the_delay_column(self) -> None:
        table = make_table()

        dataset = build_dataset(table)

        assert list(dataset.target) == [float(v) for v in table[TARGET]]
