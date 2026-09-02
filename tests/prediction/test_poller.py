"""Tests for the collection loop.

The failure path gets the most attention here on purpose. A collector that
crashes is obvious; a collector that keeps running while every poll fails
costs a week of irreplaceable training data before anyone notices.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from transit_rag.config import CollectionConfig, ConfigError
from transit_rag.prediction.collection.poller import (
    build_sink,
    poll_once,
    print_probe,
    print_status,
)
from transit_rag.prediction.collection.store import (
    CsvSnapshotStore,
    SchemaMismatchError,
    SqliteObservationStore,
)
from transit_rag.realtime.client import FeedFetchError

LOOKUP = {"APS_1a": "T1"}
TRACKED = ("T1", "T4")


class FakeClient:
    """Stands in for GtfsRealtimeClient: returns a feed, or raises."""

    def __init__(self, feed: Any = None, error: Exception | None = None) -> None:
        self._feed = feed
        self._error = error
        self.calls = 0

    def fetch_trip_updates(self) -> Any:
        self.calls += 1
        if self._error is not None:
            raise self._error
        return self._feed


def test_successful_poll_writes_observations_and_logs_ok(tmp_path: Path, make_feed: Any) -> None:
    feed = make_feed([("trip-1", "APS_1a", [("stop-a", 1, 60, None), ("stop-b", 2, 90, None)])])
    client = FakeClient(feed=feed)

    with SqliteObservationStore(tmp_path / "obs.db") as store:
        ok = poll_once(client, store, LOOKUP, TRACKED)  # type: ignore[arg-type]

        assert ok is True
        assert store.observation_count() == 2
        assert store.recent_poll_status()[0][3] == "ok"


def test_failed_poll_is_recorded_rather_than_raised(tmp_path: Path) -> None:
    """A transient 502 must not kill a collector that has been running for
    weeks — but it must leave a trace in the poll log."""
    client = FakeClient(error=FeedFetchError("GET ... failed: 502 Bad Gateway"))

    with SqliteObservationStore(tmp_path / "obs.db") as store:
        ok = poll_once(client, store, LOOKUP, TRACKED)  # type: ignore[arg-type]

        assert ok is False
        assert store.observation_count() == 0
        status = store.recent_poll_status()[0][3]
        assert status.startswith("error:")
        assert "502" in status


def test_poll_with_no_tracked_trips_still_logs_the_poll(tmp_path: Path, make_feed: Any) -> None:
    """Zero rows is a real outcome at 3am. It has to be distinguishable from
    a poll that never happened, or the volume checkpoint reads wrong.

    The route here resolves — it is just not a tracked line. That is what an
    off-peak poll looks like, and it is a different thing from a feed whose
    ids do not resolve at all (see TestLookupMismatch)."""
    feed = make_feed([("trip-t8", "APS_8a", [("stop-z", 1, 30, None)])])
    lookup = {**LOOKUP, "APS_8a": "T8"}
    client = FakeClient(feed=feed)

    with SqliteObservationStore(tmp_path / "obs.db") as store:
        poll_once(client, store, lookup, TRACKED)  # type: ignore[arg-type]

        entities_seen, rows_written, status = store.recent_poll_status()[0][1:]
        assert (entities_seen, rows_written, status) == (1, 0, "ok")


def test_poll_respects_the_upcoming_stop_limit(tmp_path: Path, make_feed: Any) -> None:
    stops = [(f"stop-{i}", i, i * 10, None) for i in range(12)]
    client = FakeClient(feed=make_feed([("trip-1", "APS_1a", stops)]))

    with SqliteObservationStore(tmp_path / "obs.db") as store:
        poll_once(client, store, LOOKUP, TRACKED, max_upcoming_stops=2)  # type: ignore[arg-type]

        assert store.observation_count() == 2


def test_poll_writes_a_snapshot_when_given_the_csv_sink(tmp_path: Path, make_feed: Any) -> None:
    feed = make_feed([("trip-1", "APS_1a", [("stop-a", 1, 60, None)])])
    client = FakeClient(feed=feed)

    with CsvSnapshotStore(tmp_path / "snapshots") as store:
        poll_once(client, store, LOOKUP, TRACKED)  # type: ignore[arg-type]

        assert store.observation_count() == 1
        assert len(list((tmp_path / "snapshots").glob("*/*.csv"))) == 2  # snapshot + _polls


def test_status_flags_a_run_where_every_poll_failed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    client = FakeClient(error=FeedFetchError("401 Unauthorized"))

    with SqliteObservationStore(tmp_path / "obs.db") as store:
        poll_once(client, store, LOOKUP, TRACKED)  # type: ignore[arg-type]
        print_status(store)

    output = capsys.readouterr().out
    assert "Every recent poll failed" in output


def test_status_on_an_empty_database_says_so(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with SqliteObservationStore(tmp_path / "obs.db") as store:
        print_status(store)

    assert "No polls recorded yet." in capsys.readouterr().out


def test_build_sink_honours_the_cli_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("COLLECTION_SINK", "sqlite")
    monkeypatch.setenv("COLLECTION_DB_PATH", str(tmp_path / "obs.db"))
    monkeypatch.setenv("COLLECTION_SNAPSHOT_DIR", str(tmp_path / "snapshots"))

    assert isinstance(build_sink(CollectionConfig()), SqliteObservationStore)
    assert isinstance(build_sink(CollectionConfig(), "csv"), CsvSnapshotStore)


def test_an_unknown_sink_is_rejected_at_config_time(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COLLECTION_SINK", "postgres")

    with pytest.raises(ConfigError, match="COLLECTION_SINK"):
        CollectionConfig()


class TestProbe:
    def test_reports_active_trips_on_the_tracked_lines(
        self, make_feed: Any, capsys: pytest.CaptureFixture[str]
    ) -> None:
        feed = make_feed(
            [
                ("a", "APS_1a", [("s", 1, 60, None)]),
                ("b", "APS_1a", [("s", 1, 60, None)]),
            ]
        )

        ok = print_probe(FakeClient(feed=feed), LOOKUP, TRACKED)  # type: ignore[arg-type]

        output = capsys.readouterr().out
        assert ok is True
        assert "T1: 2 active trips" in output
        assert "Feed and lookup agree" in output

    def test_names_the_version_mismatch_when_nothing_resolves(
        self, make_feed: Any, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A static bundle paired with the wrong realtime feed version: the
        fetch works, the parse works, and every route_id is a stranger."""
        feed = make_feed([("a", "V2_STYLE_ID", [("s", 1, 60, None)])])

        ok = print_probe(FakeClient(feed=feed), LOOKUP, TRACKED)  # type: ignore[arg-type]

        output = capsys.readouterr().out
        assert ok is False
        assert "different versions" in output

    def test_distinguishes_quiet_hours_from_a_broken_filter(
        self, make_feed: Any, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The lookup resolves fine, T1/T4 just are not running. Telling the
        user to re-probe beats sending them to re-download a correct bundle."""
        feed = make_feed([("a", "APS_8a", [("s", 1, 60, None)])])
        lookup = {**LOOKUP, "APS_8a": "T8"}  # the id resolves; it is just not tracked

        ok = print_probe(FakeClient(feed=feed), lookup, TRACKED)  # type: ignore[arg-type]

        output = capsys.readouterr().out
        assert ok is True
        assert "none of T1, T4 are running" in output

    def test_says_so_when_the_lookup_is_missing(
        self, make_feed: Any, capsys: pytest.CaptureFixture[str]
    ) -> None:
        feed = make_feed([("a", "APS_1a", [("s", 1, 60, None)])])

        print_probe(FakeClient(feed=feed), {}, TRACKED)  # type: ignore[arg-type]

        assert "No route lookup loaded" in capsys.readouterr().out

    def test_a_failed_fetch_reports_rather_than_raises(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        client = FakeClient(error=FeedFetchError("not a GTFS-Realtime protobuf"))

        ok = print_probe(client, LOOKUP, TRACKED)  # type: ignore[arg-type]

        assert ok is False
        assert "Could not read the feed" in capsys.readouterr().out


class TestPollResilience:
    """A collector that has run for weeks must not die on one bad poll."""

    def test_a_sink_failure_is_recorded_not_raised(self, make_feed: Any) -> None:
        class BrokenSink:
            def __init__(self) -> None:
                self.failures: list[str] = []

            def record_observations(self, observations: Any) -> int:
                raise OSError(28, "No space left on device")

            def record_poll(self, poll_time: str, entities: int, rows: int, status: str) -> None:
                self.failures.append(status)

        sink = BrokenSink()
        client = FakeClient(feed=make_feed([("t", "APS_1a", [("s", 1, 60, None)])]))

        ok = poll_once(client, sink, LOOKUP, TRACKED)  # type: ignore[arg-type]

        assert ok is False
        assert sink.failures and sink.failures[0].startswith("error: OSError")

    def test_a_sink_that_cannot_even_log_does_not_raise(self, make_feed: Any) -> None:
        """If recording the failure also fails, the loop still has to survive
        — the alternative is a dead process and a silent collection window."""

        class TotallyBrokenSink:
            def record_observations(self, observations: Any) -> int:
                raise RuntimeError("boom")

            def record_poll(self, *args: Any) -> None:
                raise sqlite3.OperationalError("database is locked")

        client = FakeClient(feed=make_feed([("t", "APS_1a", [("s", 1, 60, None)])]))

        assert poll_once(client, TotallyBrokenSink(), LOOKUP, TRACKED) is False  # type: ignore[arg-type]

    def test_a_malformed_feed_is_recorded_not_raised(self, tmp_path: Path) -> None:
        class Nonsense:
            @property
            def entity(self) -> Any:
                raise AttributeError("not a feed")

        client = FakeClient(feed=Nonsense())

        with SqliteObservationStore(tmp_path / "obs.db") as store:
            assert poll_once(client, store, LOOKUP, TRACKED) is False  # type: ignore[arg-type]
            assert store.recent_poll_status()[0][3].startswith("error: AttributeError")


class TestLookupMismatch:
    """A static bundle and a realtime feed from different versions.

    The fetch succeeds, the parse succeeds, the filter matches nothing, and
    the poll log says `ok`. Left undetected it collects zero rows for as long
    as nobody looks — and it can appear mid-collection, when TfNSW republishes
    the bundle and route_ids shift under a collector that has worked for weeks.
    """

    def test_a_feed_whose_ids_never_resolve_is_an_error(
        self, tmp_path: Path, make_feed: Any
    ) -> None:
        feed = make_feed([("a", "104C", [("s", 1, 60, None)]), ("b", "220N", [("s", 1, 60, None)])])

        with SqliteObservationStore(tmp_path / "obs.db") as store:
            ok = poll_once(FakeClient(feed=feed), store, LOOKUP, TRACKED)  # type: ignore[arg-type]

            assert ok is False
            status = store.recent_poll_status()[0][3]
            assert "different versions" in status

    def test_an_empty_feed_is_not_a_mismatch(self, tmp_path: Path, make_feed: Any) -> None:
        """Overnight there are no trips at all. That must stay `ok`, or the
        collector cries wolf every night and the signal stops meaning anything."""
        with SqliteObservationStore(tmp_path / "obs.db") as store:
            ok = poll_once(FakeClient(feed=make_feed([])), store, LOOKUP, TRACKED)  # type: ignore[arg-type]

            assert ok is True
            assert store.recent_poll_status()[0][3] == "ok"

    def test_a_partially_resolving_feed_is_not_a_mismatch(
        self, tmp_path: Path, make_feed: Any
    ) -> None:
        """Other operators share the feed. One resolvable id proves the bundle
        and the feed agree, whatever else is in there."""
        feed = make_feed(
            [("a", "APS_1a", [("s", 1, 60, None)]), ("b", "UNKNOWN", [("s", 1, 60, None)])]
        )

        with SqliteObservationStore(tmp_path / "obs.db") as store:
            assert poll_once(FakeClient(feed=feed), store, LOOKUP, TRACKED) is True  # type: ignore[arg-type]

    def test_no_lookup_at_all_is_not_reported_as_a_mismatch(
        self, tmp_path: Path, make_feed: Any
    ) -> None:
        """Unfiltered local collection is a supported mode; --require-routes
        is what refuses it, and this check must not duplicate that."""
        feed = make_feed([("a", "104C", [("s", 1, 60, None)])])

        with SqliteObservationStore(tmp_path / "obs.db") as store:
            assert poll_once(FakeClient(feed=feed), store, {}, TRACKED) is True  # type: ignore[arg-type]


class TestSchemaGuard:
    def test_a_database_on_the_old_primary_key_is_refused_at_open(self, tmp_path: Path) -> None:
        """CREATE TABLE IF NOT EXISTS keeps the old table silently, and then
        every write raises against an ON CONFLICT target that matches no
        constraint — a collector that records nothing but errors."""
        db_path = tmp_path / "old.db"
        legacy = sqlite3.connect(db_path)
        legacy.execute(
            "CREATE TABLE stop_observations ("
            "service_date TEXT NOT NULL, trip_id TEXT NOT NULL, stop_id TEXT NOT NULL, "
            "last_seen_utc TEXT NOT NULL, PRIMARY KEY (service_date, trip_id, stop_id))"
        )
        legacy.commit()
        legacy.close()

        with pytest.raises(SchemaMismatchError, match="older schema"):
            SqliteObservationStore(db_path)

    def test_a_fresh_database_opens_normally(self, tmp_path: Path) -> None:
        with SqliteObservationStore(tmp_path / "new.db") as store:
            assert store.observation_count() == 0
