"""Tests for the static-timetable index."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from transit_rag.prediction.features.schedule import ScheduleIndex, parse_gtfs_time

STOP_TIMES = (
    "trip_id,arrival_time,departure_time,stop_id,stop_sequence\n"
    "trip-a,06:00:00,06:00:30,stop-1,1\n"
    "trip-a,06:05:00,06:05:30,stop-2,2\n"
    "trip-a,06:11:00,06:11:30,stop-3,3\n"
    "trip-b,25:10:00,25:10:30,stop-9,1\n"
    "trip-unwanted,07:00:00,07:00:30,stop-4,1\n"
)


@pytest.fixture
def bundle(tmp_path: Path) -> Path:
    path = tmp_path / "gtfs.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("stop_times.txt", STOP_TIMES)
    return path


class TestParseGtfsTime:
    def test_parses_a_normal_time(self) -> None:
        assert parse_gtfs_time("06:05:30") == 6 * 3600 + 5 * 60 + 30

    def test_hours_past_midnight_are_legal(self) -> None:
        """GTFS expresses 1:10am on a service day that began the previous
        morning as 25:10:00. Parsing it as a clock time would reject every
        after-midnight service."""
        assert parse_gtfs_time("25:10:00") == 25 * 3600 + 10 * 60

    @pytest.mark.parametrize("value", ["", "nonsense", "06:05", "aa:bb:cc"])
    def test_malformed_values_are_none_not_exceptions(self, value: str) -> None:
        assert parse_gtfs_time(value) is None


class TestScheduleIndex:
    def test_indexes_only_the_requested_trips(self, bundle: Path) -> None:
        """stop_times.txt covers ~83,000 trips; a run only asks about the few
        thousand it observed, and loading the rest is wasted memory."""
        index = ScheduleIndex.for_trips(bundle, ["trip-a"])

        assert index.knows_trip("trip-a")
        assert not index.knows_trip("trip-unwanted")
        assert len(index) == 3

    def test_looks_up_sequence_and_scheduled_times(self, bundle: Path) -> None:
        index = ScheduleIndex.for_trips(bundle, ["trip-a"])

        scheduled = index.lookup("trip-a", "stop-2")

        assert scheduled is not None
        assert scheduled.stop_sequence == 2
        assert scheduled.scheduled_arrival_s == 6 * 3600 + 5 * 60

    def test_unknown_trip_or_stop_returns_none(self, bundle: Path) -> None:
        index = ScheduleIndex.for_trips(bundle, ["trip-a"])

        assert index.lookup("trip-zzz", "stop-1") is None
        assert index.lookup("trip-a", "stop-999") is None

    def test_coverage_reports_the_match_rate(self, bundle: Path) -> None:
        """A falling rate is the signal that the bundle has aged behind the
        timetable — trips keep the version they were planned under."""
        index = ScheduleIndex.for_trips(bundle, ["trip-a", "trip-b", "trip-missing"])

        assert index.coverage(["trip-a", "trip-b", "trip-missing"]) == (2, 3)

    def test_a_bundle_without_stop_times_is_an_error(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.zip"
        with zipfile.ZipFile(empty, "w") as archive:
            archive.writestr("routes.txt", "route_id\n")

        with pytest.raises(FileNotFoundError, match=r"stop_times\.txt"):
            ScheduleIndex.for_trips(empty, ["trip-a"])


class TestAcrossBundles:
    """Merging timetable eras.

    Measured over 3-9 September 2026, TfNSW rolled the timetable over
    mid-window: each bundle matched about half the collected trips and neither
    matched the whole window. These cover the merge that recovers the rest.
    """

    @pytest.fixture
    def later_era(self, tmp_path: Path) -> Path:
        """A bundle from after a republication — different trips entirely."""
        path = tmp_path / "gtfs_later.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(
                "stop_times.txt",
                "trip_id,arrival_time,departure_time,stop_id,stop_sequence\n"
                "trip-c,08:00:00,08:00:30,stop-7,1\n"
                "trip-c,08:09:00,08:09:30,stop-8,2\n",
            )
        return path

    def test_neither_bundle_alone_covers_a_window_that_spans_a_rollover(
        self, bundle: Path, later_era: Path
    ) -> None:
        wanted = ["trip-a", "trip-c"]

        assert ScheduleIndex.for_trips(bundle, wanted).coverage(wanted) == (1, 2)
        assert ScheduleIndex.for_trips(later_era, wanted).coverage(wanted) == (1, 2)

    def test_merging_the_eras_covers_all_of_it(self, bundle: Path, later_era: Path) -> None:
        wanted = ["trip-a", "trip-c"]

        index = ScheduleIndex.across_bundles([bundle, later_era], wanted)

        assert index.coverage(wanted) == (2, 2)
        assert index.lookup("trip-a", "stop-2") is not None
        assert index.lookup("trip-c", "stop-8") is not None

    def test_one_bundle_behaves_exactly_like_for_trips(self, bundle: Path) -> None:
        wanted = ["trip-a", "trip-b"]

        merged = ScheduleIndex.across_bundles([bundle], wanted)
        single = ScheduleIndex.for_trips(bundle, wanted)

        assert merged.coverage(wanted) == single.coverage(wanted)
        assert len(merged) == len(single)

    def test_no_bundles_is_an_empty_index_not_an_error(self) -> None:
        """The caller decides whether a missing bundle is fatal; reconciliation
        treats it as a thinner table, not a failure."""
        index = ScheduleIndex.across_bundles([], ["trip-a"])

        assert index.coverage(["trip-a"]) == (0, 1)
        assert len(index) == 0

    def test_earlier_bundles_win_when_both_describe_a_trip(
        self, bundle: Path, tmp_path: Path
    ) -> None:
        """Only arises for a trip present in both eras, where the entries are
        equivalent anyway — pinned so the precedence is not accidental."""
        conflicting = tmp_path / "gtfs_conflict.zip"
        with zipfile.ZipFile(conflicting, "w") as archive:
            archive.writestr(
                "stop_times.txt",
                "trip_id,arrival_time,departure_time,stop_id,stop_sequence\n"
                "trip-a,23:00:00,23:00:30,stop-2,99\n",
            )

        index = ScheduleIndex.across_bundles([bundle, conflicting], ["trip-a"])

        scheduled = index.lookup("trip-a", "stop-2")
        assert scheduled is not None
        assert scheduled.stop_sequence == 2
