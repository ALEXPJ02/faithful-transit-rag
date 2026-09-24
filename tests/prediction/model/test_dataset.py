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
    filter_plausible,
    filter_reliable,
    filter_schedule_covered,
    schedule_coverage,
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


class TestFilterPlausible:
    """The ghost trip: the feed republishing yesterday's run as today's.

    These rows are not slow trains. A ~24 h "delay" is an artifact, and one of
    them carries more error than thousands of real rows combined -- on the
    2026-09-16 table seven of them held 55% of validation MAE.
    """

    def ghost_table(self) -> pd.DataFrame:
        table = make_table(6)
        table.loc[2, "delay_s"] = 86142  # 23.93 h -- the signature of a rollover
        table.loc[2, "prev_stop_delay_s"] = 86142
        return table

    def test_a_twenty_four_hour_delay_is_dropped(self) -> None:
        kept = filter_plausible(self.ghost_table())

        assert len(kept) == 5
        assert 86142 not in set(kept["delay_s"])

    def test_real_delays_are_kept(self) -> None:
        """The bound must not reach anywhere near a genuinely late train."""
        table = make_table(6)
        table.loc[1, "delay_s"] = 4394  # the largest real delay in the corpus, 73 min

        assert len(filter_plausible(table)) == 6

    def test_an_implausible_previous_stop_delay_is_dropped_too(self) -> None:
        """The baseline predicts from this column, so a junk value there is a
        junk *baseline*, and the model would be beating a strawman."""
        table = make_table(6)
        table.loc[3, "prev_stop_delay_s"] = 90000

        kept = filter_plausible(table)

        assert len(kept) == 5
        assert 90000 not in set(kept["prev_stop_delay_s"].dropna())

    def test_a_missing_previous_stop_delay_is_not_implausible(self) -> None:
        """Absent is a different claim from impossible -- and the first stop of
        every trip has no previous-stop delay by construction."""
        table = make_table(6)
        assert table["prev_stop_delay_s"].isna().any()

        assert len(filter_plausible(table)) == 6

    def test_a_large_negative_delay_is_dropped(self) -> None:
        """A train running 24 h early is the same artifact with the sign flipped."""
        table = make_table(6)
        table.loc[4, "delay_s"] = -86142

        assert len(filter_plausible(table)) == 5

    def test_the_bound_is_configurable_for_sensitivity_checks(self) -> None:
        table = make_table(6)
        table.loc[1, "delay_s"] = 4394

        assert len(filter_plausible(table, max_delay_s=1800)) == 5

    def test_it_returns_a_clean_index(self) -> None:
        """Downstream code aligns predictions to targets positionally."""
        kept = filter_plausible(self.ghost_table())

        assert list(kept.index) == list(range(len(kept)))

    def test_filtering_before_the_split_leaves_every_partition_clean(self) -> None:
        """The ordering invariant, and the reason the filter is not optional.

        Filter then split, and all three partitions are on the same footing.
        Split then filter -- or filter only test -- silently changes what each
        partition means, and is indistinguishable from keeping the rows that
        flatter the result.
        """
        from transit_rag.prediction.features.quality import time_based_split

        rows = 12
        table = make_table(6)
        table = pd.concat([table] * (rows // 6), ignore_index=True)
        table["service_date"] = [f"2026-09-{3 + i:02d}" for i in range(len(table))]
        table.loc[len(table) - 1, "delay_s"] = 86142  # lands in the test partition

        split = time_based_split(filter_plausible(table))

        for partition in (split.train, split.validation, split.test):
            assert (partition["delay_s"].abs() <= 7200).all()


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


class TestFilterScheduleCovered:
    """Dropping service dates the static timetable can no longer describe.

    The failure this guards against is not a crash. A schedule-blind date
    trains and scores perfectly happily while missing two of nine features --
    it just measures something other than what the write-up claims. The
    previous model drew its whole validation split from such a window.
    """

    def _dated(self, dates_to_coverage: dict[str, float], per_date: int = 10) -> pd.DataFrame:
        """A table where each date has a known schedule-join coverage.

        Built directly rather than via ``make_table``, which caps several
        columns at six rows; only the two columns the filter reads matter.
        """
        rows = []
        for date, coverage in dates_to_coverage.items():
            joined = round(coverage * per_date)
            for i in range(per_date):
                rows.append(
                    {
                        "service_date": date,
                        "scheduled_arrival_s": float(i) if i < joined else None,
                    }
                )
        return pd.DataFrame(rows)

    def test_drops_the_blind_dates_and_keeps_the_rest(self) -> None:
        table = self._dated({"2026-09-10": 0.9, "2026-09-11": 0.0, "2026-09-16": 1.0})

        kept = filter_schedule_covered(table)

        assert sorted(kept["service_date"].unique()) == ["2026-09-10", "2026-09-16"]

    def test_a_whole_date_leaves_even_when_a_few_rows_join(self) -> None:
        # 2026-09-11 really does join 0.1% of its rows. Keeping those would put
        # a handful of unrepresentative rows into the split and leave the
        # date's boundary in place, which is the thing being removed.
        table = self._dated({"2026-09-11": 0.1, "2026-09-16": 1.0}, per_date=10)

        kept = filter_schedule_covered(table)

        assert "2026-09-11" not in set(kept["service_date"])
        assert len(kept) == 10

    def test_threshold_sits_in_the_empty_band(self) -> None:
        # The measured distribution is bimodal, so any threshold between the
        # two modes removes the same dates. If that ever stops being true the
        # constant has become a fitted parameter and needs re-justifying.
        table = self._dated({"blind": 0.0, "covered": 0.96}, per_date=100)

        for threshold in (0.01, 0.5, 0.95):
            kept = filter_schedule_covered(table, min_coverage=threshold)
            assert sorted(kept["service_date"].unique()) == ["covered"], threshold

    def test_coverage_is_reported_per_date(self) -> None:
        table = self._dated({"2026-09-11": 0.0, "2026-09-16": 1.0}, per_date=10)

        coverage = schedule_coverage(table)

        assert coverage["2026-09-11"] == pytest.approx(0.0)
        assert coverage["2026-09-16"] == pytest.approx(1.0)

    def test_everything_covered_is_a_no_op(self) -> None:
        table = self._dated({"2026-09-16": 1.0, "2026-09-17": 0.9})

        assert len(filter_schedule_covered(table)) == len(table)

    def test_it_returns_a_clean_index(self) -> None:
        """Downstream code aligns predictions to targets positionally."""
        kept = filter_schedule_covered(
            self._dated({"2026-09-11": 0.0, "2026-09-16": 1.0}, per_date=10)
        )

        assert list(kept.index) == list(range(len(kept)))

    def test_a_date_exactly_at_the_threshold_is_kept(self) -> None:
        # The comparison is >=, so a date sitting on the boundary stays. The
        # real distribution is nowhere near it, but the direction should be a
        # decision rather than an accident of which operator was typed.
        table = self._dated({"exactly-half": 0.5, "just-under": 0.49}, per_date=100)

        kept = filter_schedule_covered(table, min_coverage=0.5)

        assert sorted(kept["service_date"].unique()) == ["exactly-half"]

    def test_every_other_column_survives(self) -> None:
        # The filter selects rows; it must not reshape the table. A dropped
        # column here would surface much later as a missing feature.
        table = self._dated({"2026-09-11": 0.0, "2026-09-16": 1.0})
        table["delay_s"] = 42.0
        table["route_short_name"] = "T1"

        kept = filter_schedule_covered(table)

        assert list(kept.columns) == list(table.columns)
        assert set(kept["delay_s"]) == {42.0}
