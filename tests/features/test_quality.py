"""Tests for the split and the data-quality report."""

from __future__ import annotations

import pandas as pd

from transit_rag.prediction.features.quality import Split, report, time_based_split


def table(service_dates: list[str], rows_per_date: int = 2) -> pd.DataFrame:
    records = []
    for date in service_dates:
        for i in range(rows_per_date):
            records.append(
                {
                    "service_date": date,
                    "trip_id": f"trip-{i}",
                    "stop_id": f"stop-{i}",
                    "route_short_name": "T1",
                    "delay_s": 60 * i,
                    "prev_stop_delay_s": None if i == 0 else 30,
                    "stops_ahead_final": i,
                    "observation_count": i + 1,
                    "schedule_matched": True,
                }
            )
    return pd.DataFrame(records)


class TestTimeBasedSplit:
    def test_splits_chronologically_not_randomly(self) -> None:
        """A random shuffle would put the same afternoon on both sides of the
        boundary, and the model would score well by having already seen the
        conditions it is asked to predict."""
        dates = [f"2026-09-{d:02d}" for d in range(1, 21)]

        split = time_based_split(table(dates))

        assert split.train["service_date"].max() < split.validation["service_date"].min()
        assert split.validation["service_date"].max() < split.test["service_date"].min()

    def test_a_service_date_never_straddles_the_boundary(self) -> None:
        """Splitting on rows rather than whole days leaks the same way, just
        less visibly."""
        dates = [f"2026-09-{d:02d}" for d in range(1, 11)]

        split = time_based_split(table(dates, rows_per_date=7))

        in_train = set(split.train["service_date"])
        in_validation = set(split.validation["service_date"])
        in_test = set(split.test["service_date"])
        assert not (in_train & in_validation)
        assert not (in_validation & in_test)
        assert not (in_train & in_test)

    def test_respects_the_requested_proportions(self) -> None:
        dates = [f"2026-09-{d:02d}" for d in range(1, 21)]

        split = time_based_split(table(dates))

        assert split.train["service_date"].nunique() == 14
        assert split.validation["service_date"].nunique() == 3
        assert split.test["service_date"].nunique() == 3

    def test_a_single_day_lands_entirely_in_train(self) -> None:
        """Early in collection there is only one day. It must not silently
        produce empty training data."""
        split = time_based_split(table(["2026-09-03"]))

        assert len(split.train) > 0
        assert split.validation.empty
        assert split.test.empty

    def test_empty_input_gives_three_empty_frames(self) -> None:
        split = time_based_split(pd.DataFrame(columns=["service_date"]))

        assert split.train.empty and split.validation.empty and split.test.empty

    def test_describe_does_not_divide_by_zero_when_empty(self) -> None:
        empty = pd.DataFrame(columns=["service_date"])
        assert "empty" in Split(empty, empty.copy(), empty.copy()).describe()


class TestReport:
    def test_names_the_reliable_subset(self) -> None:
        text = report(table([f"2026-09-{d:02d}" for d in range(1, 5)]))

        assert "reliable outcomes" in text
        assert "Matched to the static timetable" in text

    def test_warns_about_days_far_below_the_median(self) -> None:
        """A collection gap looks like a normal day with very few rows, which
        is easy to miss in a table of totals."""
        full = table([f"2026-09-{d:02d}" for d in range(1, 10)], rows_per_date=20)
        thin = table(["2026-09-10"], rows_per_date=1)

        text = report(pd.concat([full, thin], ignore_index=True))

        assert "WARNING" in text
        assert "2026-09-10" in text

    def test_an_empty_table_says_so_plainly(self) -> None:
        assert "empty" in report(pd.DataFrame()).lower()
