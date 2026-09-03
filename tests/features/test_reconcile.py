"""Tests for turning observations into a training table."""

from __future__ import annotations

import pandas as pd
import pytest

from transit_rag.prediction.features.reconcile import (
    TRAINING_COLUMNS,
    add_previous_stop_delay,
    add_time_features,
    attach_schedule,
    build_training_table,
    collapse_to_stop_events,
)
from transit_rag.prediction.features.schedule import ScheduledStop, ScheduleIndex


def observation(
    *,
    stop_id: str = "stop-1",
    trip_id: str = "trip-a",
    service_date: str = "2026-09-03",
    stops_ahead: int = 0,
    arrival: int | None = 60,
    departure: int | None = None,
    observed_at: str = "2026-09-03T02:00:00+00:00",
) -> dict[str, object]:
    return {
        "service_date": service_date,
        "trip_id": trip_id,
        "stop_id": stop_id,
        "route_id": "NSN_2i",
        "route_short_name": "T1",
        "stops_ahead": stops_ahead,
        "arrival_delay_s": arrival,
        "departure_delay_s": departure,
        "observed_at_utc": observed_at,
    }


def frame(*rows: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


class TestCollapseToStopEvents:
    def test_keeps_the_last_observation_of_each_stop_event(self) -> None:
        """The feed only reports a stop until the train reaches it, so the last
        observation is the closest thing to an outcome."""
        collapsed = collapse_to_stop_events(
            frame(
                observation(arrival=300, stops_ahead=3, observed_at="2026-09-03T02:00:00+00:00"),
                observation(arrival=120, stops_ahead=0, observed_at="2026-09-03T02:06:00+00:00"),
                observation(arrival=200, stops_ahead=1, observed_at="2026-09-03T02:03:00+00:00"),
            )
        )

        assert len(collapsed) == 1
        assert collapsed.iloc[0]["arrival_delay_s"] == 120
        assert collapsed.iloc[0]["stops_ahead"] == 0

    def test_counts_how_many_observations_backed_each_event(self) -> None:
        """A stop seen five times as the train approached is better evidence
        than one glimpsed once from far out, and evaluation should be able to
        tell them apart."""
        collapsed = collapse_to_stop_events(
            frame(
                observation(observed_at="2026-09-03T02:00:00+00:00"),
                observation(observed_at="2026-09-03T02:02:00+00:00"),
                observation(observed_at="2026-09-03T02:04:00+00:00"),
            )
        )

        assert collapsed.iloc[0]["observation_count"] == 3

    def test_the_same_stop_on_two_service_dates_stays_separate(self) -> None:
        """Trip ids repeat daily; without the date they would collapse."""
        collapsed = collapse_to_stop_events(
            frame(
                observation(service_date="2026-09-02"),
                observation(service_date="2026-09-03"),
            )
        )

        assert len(collapsed) == 2

    def test_preexisting_counts_are_summed_not_discarded(self) -> None:
        """The SQLite source already carries a count; merging it with snapshot
        rows must not throw that history away."""
        rows = frame(
            observation(observed_at="2026-09-03T02:00:00+00:00"),
            observation(observed_at="2026-09-03T02:02:00+00:00"),
        )
        rows["observation_count"] = [7, 1]

        collapsed = collapse_to_stop_events(rows)

        assert collapsed.iloc[0]["observation_count"] == 8

    def test_empty_input_is_an_empty_frame_not_an_error(self) -> None:
        assert collapse_to_stop_events(pd.DataFrame()).empty


class TestAttachSchedule:
    def _index(self) -> ScheduleIndex:
        return ScheduleIndex(
            {("trip-a", "stop-1"): ScheduledStop(1, 21600, 21630)},
            {"trip-a"},
        )

    def test_adds_sequence_and_scheduled_arrival(self) -> None:
        result = attach_schedule(frame(observation()), self._index())

        assert result.iloc[0]["stop_sequence"] == 1
        assert result.iloc[0]["scheduled_arrival_s"] == 21600
        assert bool(result.iloc[0]["schedule_matched"]) is True

    def test_unmatched_trips_are_kept_with_null_schedule(self) -> None:
        """~11% of trips run on a superseded timetable version and are absent
        from the bundle. Their delays are still real observations, and dropping
        them would bias the set toward whatever timetable was current."""
        result = attach_schedule(frame(observation(trip_id="trip-old")), self._index())

        assert len(result) == 1
        assert pd.isna(result.iloc[0]["stop_sequence"])
        assert bool(result.iloc[0]["schedule_matched"]) is False


class TestPreviousStopDelay:
    def test_uses_timetable_order_when_known(self) -> None:
        rows = frame(
            observation(stop_id="s3", observed_at="2026-09-03T02:06:00+00:00"),
            observation(stop_id="s1", observed_at="2026-09-03T02:00:00+00:00"),
            observation(stop_id="s2", observed_at="2026-09-03T02:03:00+00:00"),
        )
        rows["delay_s"] = [300, 100, 200]
        rows["stop_sequence"] = pd.array([3, 1, 2], dtype="Int64")

        result = add_previous_stop_delay(rows).set_index("stop_id")

        assert pd.isna(result.loc["s1", "prev_stop_delay_s"])
        assert result.loc["s2", "prev_stop_delay_s"] == 100
        assert result.loc["s3", "prev_stop_delay_s"] == 200

    def test_falls_back_to_observation_order_without_a_sequence(self) -> None:
        """A train's later stops leave the feed later, so observation time
        recovers visit order well enough to keep the unmatched trips."""
        rows = frame(
            observation(stop_id="s2", observed_at="2026-09-03T02:03:00+00:00"),
            observation(stop_id="s1", observed_at="2026-09-03T02:00:00+00:00"),
        )
        rows["delay_s"] = [200, 100]
        rows["stop_sequence"] = pd.array([None, None], dtype="Int64")

        result = add_previous_stop_delay(rows).set_index("stop_id")

        assert pd.isna(result.loc["s1", "prev_stop_delay_s"])
        assert result.loc["s2", "prev_stop_delay_s"] == 100

    def test_does_not_leak_across_trips(self) -> None:
        rows = frame(
            observation(trip_id="trip-a", stop_id="s1"),
            observation(trip_id="trip-b", stop_id="s1"),
        )
        rows["delay_s"] = [100, 200]
        rows["stop_sequence"] = pd.array([1, 1], dtype="Int64")

        result = add_previous_stop_delay(rows)

        assert result["prev_stop_delay_s"].isna().all()


class TestTimeFeatures:
    def test_uses_sydney_local_time(self) -> None:
        """A peak flag derived from UTC would put Sydney's morning peak in the
        middle of the night."""
        result = add_time_features(frame(observation(observed_at="2026-09-02T22:00:00+00:00")))

        assert result.iloc[0]["hour_local"] == 8  # 08:00 Sydney

    def test_flags_weekday_peak(self) -> None:
        result = add_time_features(frame(observation(observed_at="2026-09-02T22:00:00+00:00")))

        assert bool(result.iloc[0]["is_peak"]) is True
        assert bool(result.iloc[0]["is_weekend"]) is False

    def test_weekend_is_never_peak(self) -> None:
        """Saturday 08:00 Sydney — peak hours, but no commuter peak."""
        result = add_time_features(frame(observation(observed_at="2026-09-04T22:00:00+00:00")))

        assert bool(result.iloc[0]["is_weekend"]) is True
        assert bool(result.iloc[0]["is_peak"]) is False

    def test_off_peak_weekday_is_not_peak(self) -> None:
        result = add_time_features(frame(observation(observed_at="2026-09-03T02:00:00+00:00")))

        assert bool(result.iloc[0]["is_peak"]) is False


class TestBuildTrainingTable:
    def test_produces_the_documented_schema(self) -> None:
        table = build_training_table(frame(observation()))

        assert list(table.columns) == TRAINING_COLUMNS

    def test_departure_delay_stands_in_when_arrival_is_missing(self) -> None:
        table = build_training_table(frame(observation(arrival=None, departure=90)))

        assert table.iloc[0]["delay_s"] == 90

    def test_rows_with_no_delay_at_all_are_dropped(self) -> None:
        """Nothing to predict, and a fabricated zero in the target column is
        exactly the failure the collector was fixed to avoid."""
        table = build_training_table(frame(observation(arrival=None, departure=None)))

        assert table.empty

    def test_empty_input_yields_the_schema_with_no_rows(self) -> None:
        table = build_training_table(pd.DataFrame())

        assert list(table.columns) == TRAINING_COLUMNS
        assert table.empty

    def test_works_without_a_schedule_index(self) -> None:
        """No bundle is a thinner table, not a failure."""
        table = build_training_table(frame(observation()))

        assert len(table) == 1
        assert bool(table.iloc[0]["schedule_matched"]) is False


@pytest.mark.parametrize("column", TRAINING_COLUMNS)
def test_every_documented_column_is_produced(column: str) -> None:
    table = build_training_table(frame(observation()))
    assert column in table.columns
