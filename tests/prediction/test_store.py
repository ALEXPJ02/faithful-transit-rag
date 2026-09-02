"""Tests for the two observation sinks."""

from __future__ import annotations

import csv
from pathlib import Path

from transit_rag.prediction.collection.store import CsvSnapshotStore, SqliteObservationStore
from transit_rag.realtime.parsing import StopDelayObservation


def _observation(
    *,
    stop_id: str = "stop-a",
    delay: int = 60,
    observed_at: str = "2026-09-02T09:00:00+00:00",
    service_date: str = "2026-09-02",
    stops_ahead: int = 0,
) -> StopDelayObservation:
    return StopDelayObservation(
        service_date=service_date,
        trip_id="trip-1",
        stop_id=stop_id,
        route_id="APS_1a",
        route_short_name="T1",
        stop_sequence=1,
        stops_ahead=stops_ahead,
        arrival_delay_s=delay,
        departure_delay_s=None,
        schedule_relationship="SCHEDULED",
        observed_at_utc=observed_at,
    )


class TestSqliteObservationStore:
    def test_writes_and_counts_observations(self, tmp_path: Path) -> None:
        with SqliteObservationStore(tmp_path / "obs.db") as store:
            store.record_observations([_observation(stop_id="a"), _observation(stop_id="b")])

            assert store.observation_count() == 2

    def test_creates_missing_parent_directories(self, tmp_path: Path) -> None:
        """The default path is data/, which is gitignored and so absent on a
        fresh clone — collection must not fail on first run because of it."""
        with SqliteObservationStore(tmp_path / "nested" / "deeper" / "obs.db") as store:
            assert store.db_path.exists()

    def test_a_later_poll_replaces_the_stored_prediction(self, tmp_path: Path) -> None:
        """One row per stop event, holding the latest known value — that is
        what makes the table the training shape rather than a poll log."""
        with SqliteObservationStore(tmp_path / "obs.db") as store:
            store.record_observations(
                [_observation(delay=60, observed_at="2026-09-02T09:00:00+00:00")]
            )
            store.record_observations(
                [_observation(delay=180, observed_at="2026-09-02T09:02:00+00:00")]
            )

            assert store.observation_count() == 1
            row = store._connection.execute(
                "SELECT arrival_delay_s, observation_count, first_seen_utc, last_seen_utc "
                "FROM stop_observations"
            ).fetchone()
            assert row[0] == 180
            assert row[1] == 2
            assert row[2] == "2026-09-02T09:00:00+00:00"
            assert row[3] == "2026-09-02T09:02:00+00:00"

    def test_an_out_of_order_poll_does_not_clobber_a_newer_value(self, tmp_path: Path) -> None:
        """After a retry or a clock adjustment an older prediction can arrive
        last. The last value before a stop leaves the feed is the outcome
        proxy, so 'latest' must mean latest by time, not by arrival."""
        with SqliteObservationStore(tmp_path / "obs.db") as store:
            store.record_observations(
                [_observation(delay=180, observed_at="2026-09-02T09:02:00+00:00")]
            )
            store.record_observations(
                [_observation(delay=60, observed_at="2026-09-02T09:00:00+00:00")]
            )

            delay = store._connection.execute(
                "SELECT arrival_delay_s FROM stop_observations"
            ).fetchone()[0]
            assert delay == 180

    def test_the_same_stop_on_two_service_dates_is_two_rows(self, tmp_path: Path) -> None:
        """Trip ids repeat daily. Without the service date in the key, every
        day would overwrite the last."""
        with SqliteObservationStore(tmp_path / "obs.db") as store:
            store.record_observations([_observation(service_date="2026-09-01")])
            store.record_observations([_observation(service_date="2026-09-02")])

            assert store.observation_count() == 2

    def test_reports_the_service_date_range(self, tmp_path: Path) -> None:
        with SqliteObservationStore(tmp_path / "obs.db") as store:
            store.record_observations(
                [_observation(service_date="2026-09-01"), _observation(service_date="2026-09-05")]
            )

            assert store.service_date_range() == ("2026-09-01", "2026-09-05")

    def test_service_date_range_is_none_when_empty(self, tmp_path: Path) -> None:
        with SqliteObservationStore(tmp_path / "obs.db") as store:
            assert store.service_date_range() is None

    def test_records_poll_outcomes_newest_first(self, tmp_path: Path) -> None:
        with SqliteObservationStore(tmp_path / "obs.db") as store:
            store.record_poll("2026-09-02T09:00:00+00:00", 500, 120, "ok")
            store.record_poll("2026-09-02T09:02:00+00:00", 0, 0, "error: boom")

            recent = store.recent_poll_status()

            assert [row[0] for row in recent] == [
                "2026-09-02T09:02:00+00:00",
                "2026-09-02T09:00:00+00:00",
            ]
            assert recent[0][3].startswith("error")

    def test_empty_write_is_a_no_op(self, tmp_path: Path) -> None:
        with SqliteObservationStore(tmp_path / "obs.db") as store:
            assert store.record_observations([]) == 0


class TestCsvSnapshotStore:
    def test_writes_one_file_per_poll_partitioned_by_date(self, tmp_path: Path) -> None:
        with CsvSnapshotStore(tmp_path) as store:
            store.record_observations([_observation(observed_at="2026-09-02T09:00:00+00:00")])
            store.record_observations([_observation(observed_at="2026-09-02T09:02:00+00:00")])
            store.record_observations([_observation(observed_at="2026-09-03T09:00:00+00:00")])

        assert sorted(p.name for p in tmp_path.iterdir()) == ["2026-09-02", "2026-09-03"]
        assert len(list((tmp_path / "2026-09-02").glob("2026*.csv"))) == 2

    def test_snapshot_files_are_never_rewritten(self, tmp_path: Path) -> None:
        """A scheduled runner commits its output. A file that is only ever
        added is stored once; one rewritten every few minutes stores a fresh
        copy of its whole contents in history each time."""
        with CsvSnapshotStore(tmp_path) as store:
            store.record_observations([_observation(observed_at="2026-09-02T09:00:00+00:00")])
            first = next((tmp_path / "2026-09-02").glob("2026*.csv"))
            before = first.read_text()

            store.record_observations(
                [_observation(delay=999, observed_at="2026-09-02T09:02:00+00:00")]
            )

            assert first.read_text() == before

    def test_snapshot_carries_every_column(self, tmp_path: Path) -> None:
        with CsvSnapshotStore(tmp_path) as store:
            store.record_observations([_observation(delay=75, stops_ahead=2)])

        path = next((tmp_path / "2026-09-02").glob("2026*.csv"))
        with path.open(newline="", encoding="utf-8") as handle:
            row = next(iter(csv.DictReader(handle)))

        assert row["arrival_delay_s"] == "75"
        assert row["stops_ahead"] == "2"
        assert row["service_date"] == "2026-09-02"
        assert row["route_short_name"] == "T1"

    def test_poll_log_appends_within_a_day(self, tmp_path: Path) -> None:
        with CsvSnapshotStore(tmp_path) as store:
            store.record_poll("2026-09-02T09:00:00+00:00", 500, 120, "ok")
            store.record_poll("2026-09-02T09:02:00+00:00", 0, 0, "error: 401")

            recent = store.recent_poll_status()

        assert [row[0] for row in recent] == [
            "2026-09-02T09:02:00+00:00",
            "2026-09-02T09:00:00+00:00",
        ]
        assert recent[0][3] == "error: 401"

    def test_counts_rows_across_snapshots(self, tmp_path: Path) -> None:
        with CsvSnapshotStore(tmp_path) as store:
            store.record_observations(
                [
                    _observation(stop_id="a", observed_at="2026-09-02T09:00:00+00:00"),
                    _observation(stop_id="b", observed_at="2026-09-02T09:00:00+00:00"),
                ]
            )
            store.record_observations([_observation(observed_at="2026-09-02T09:02:00+00:00")])
            store.record_poll("2026-09-02T09:02:00+00:00", 1, 1, "ok")

            assert store.observation_count() == 3

    def test_empty_write_produces_no_file(self, tmp_path: Path) -> None:
        with CsvSnapshotStore(tmp_path) as store:
            assert store.record_observations([]) == 0

        assert list(tmp_path.glob("*/*.csv")) == []
