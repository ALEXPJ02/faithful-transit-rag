"""Tests for the feed -> observation transform."""

from __future__ import annotations

from typing import Any

from transit_rag.realtime.parsing import extract_delay_observations

POLL_TIME = "2026-09-02T09:00:00+00:00"
LOOKUP = {"APS_1a": "T1", "APS_4a": "T4", "APS_8a": "T8"}
TRACKED = ("T1", "T4")


def test_extracts_one_observation_per_stop_with_a_delay(make_feed: Any) -> None:
    feed = make_feed([("trip-1", "APS_1a", [("stop-a", 1, 60, 65), ("stop-b", 2, 120, None)])])

    observations = extract_delay_observations(feed, LOOKUP, TRACKED, POLL_TIME)

    assert [o.stop_id for o in observations] == ["stop-a", "stop-b"]
    assert observations[0].arrival_delay_s == 60
    assert observations[0].departure_delay_s == 65
    assert observations[1].departure_delay_s is None
    assert all(o.route_short_name == "T1" for o in observations)
    assert all(o.poll_time_utc == POLL_TIME for o in observations)


def test_skips_untracked_lines(make_feed: Any) -> None:
    feed = make_feed(
        [
            ("trip-t1", "APS_1a", [("stop-a", 1, 60, None)]),
            ("trip-t8", "APS_8a", [("stop-z", 1, 999, None)]),
        ]
    )

    observations = extract_delay_observations(feed, LOOKUP, TRACKED, POLL_TIME)

    assert [o.trip_id for o in observations] == ["trip-t1"]


def test_skips_stops_with_no_delay_prediction(make_feed: Any) -> None:
    """A stop with neither arrival nor departure delay carries no signal."""
    feed = make_feed([("trip-1", "APS_1a", [("stop-a", 1, None, None), ("stop-b", 2, 30, None)])])

    observations = extract_delay_observations(feed, LOOKUP, TRACKED, POLL_TIME)

    assert [o.stop_id for o in observations] == ["stop-b"]


def test_empty_lookup_collects_everything_rather_than_nothing(make_feed: Any) -> None:
    """Filtering against an empty lookup would silently collect zero rows for
    days. Collecting unfiltered is the louder, recoverable failure."""
    feed = make_feed([("trip-t8", "APS_8a", [("stop-z", 1, 45, None)])])

    observations = extract_delay_observations(feed, {}, TRACKED, POLL_TIME)

    assert len(observations) == 1
    assert observations[0].route_short_name is None


def test_ignores_entities_without_a_trip_update(make_feed: Any) -> None:
    feed = make_feed([("trip-1", "APS_1a", [("stop-a", 1, 60, None)])])
    feed.entity.add().id = "alert-only"

    observations = extract_delay_observations(feed, LOOKUP, TRACKED, POLL_TIME)

    assert len(observations) == 1


def test_row_order_matches_the_table_schema(make_feed: Any) -> None:
    feed = make_feed([("trip-1", "APS_1a", [("stop-a", 3, 60, 65)])])

    row = extract_delay_observations(feed, LOOKUP, TRACKED, POLL_TIME)[0].as_row()

    assert row == (POLL_TIME, "trip-1", "APS_1a", "T1", "stop-a", 3, 60, 65, "SCHEDULED")
