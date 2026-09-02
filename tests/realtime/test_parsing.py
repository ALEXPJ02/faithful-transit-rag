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
    assert all(o.observed_at_utc == POLL_TIME for o in observations)


def test_records_how_far_ahead_each_stop_was(make_feed: Any) -> None:
    """stops_ahead is what lets a later analysis distinguish a prediction made
    one stop out from one made twenty stops out."""
    feed = make_feed(
        [("trip-1", "APS_1a", [("a", 1, 10, None), ("b", 2, 20, None), ("c", 3, 30, None)])]
    )

    observations = extract_delay_observations(feed, LOOKUP, TRACKED, POLL_TIME)

    assert [o.stops_ahead for o in observations] == [0, 1, 2]


def test_keeps_only_the_imminent_stops(make_feed: Any) -> None:
    """The volume control: a trip publishes ~20 upcoming stops, and all but
    the nearest few are distant guesses that will be revised many times."""
    stops = [(f"stop-{i}", i, i * 10, None) for i in range(20)]
    feed = make_feed([("trip-1", "APS_1a", stops)])

    observations = extract_delay_observations(
        feed, LOOKUP, TRACKED, POLL_TIME, max_upcoming_stops=3
    )

    assert [o.stop_id for o in observations] == ["stop-0", "stop-1", "stop-2"]


def test_zero_max_keeps_every_stop(make_feed: Any) -> None:
    stops = [(f"stop-{i}", i, i * 10, None) for i in range(8)]
    feed = make_feed([("trip-1", "APS_1a", stops)])

    observations = extract_delay_observations(
        feed, LOOKUP, TRACKED, POLL_TIME, max_upcoming_stops=0
    )

    assert len(observations) == 8


def test_a_stop_without_a_prediction_does_not_consume_a_slot(make_feed: Any) -> None:
    """Otherwise a trip whose next stop has no prediction silently loses the
    stops behind it — exactly the ones worth having."""
    feed = make_feed(
        [("trip-1", "APS_1a", [("a", 1, None, None), ("b", 2, 20, None), ("c", 3, 30, None)])]
    )

    observations = extract_delay_observations(
        feed, LOOKUP, TRACKED, POLL_TIME, max_upcoming_stops=2
    )

    assert [o.stop_id for o in observations] == ["b", "c"]


def test_skips_untracked_lines(make_feed: Any) -> None:
    feed = make_feed(
        [
            ("trip-t1", "APS_1a", [("stop-a", 1, 60, None)]),
            ("trip-t8", "APS_8a", [("stop-z", 1, 999, None)]),
        ]
    )

    observations = extract_delay_observations(feed, LOOKUP, TRACKED, POLL_TIME)

    assert [o.trip_id for o in observations] == ["trip-t1"]


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


def test_uses_the_trips_own_service_date_when_present(make_feed: Any) -> None:
    feed = make_feed([("trip-1", "APS_1a", [("stop-a", 1, 60, None)], "20260901")])

    observations = extract_delay_observations(feed, LOOKUP, TRACKED, POLL_TIME)

    assert observations[0].service_date == "2026-09-01"


def test_service_date_falls_back_to_the_sydney_local_date(make_feed: Any) -> None:
    """22:00 UTC is already the next calendar day in Sydney. Falling back to
    the UTC date would split every evening peak across two service dates."""
    feed = make_feed([("trip-1", "APS_1a", [("stop-a", 1, 60, None)])])

    observations = extract_delay_observations(feed, LOOKUP, TRACKED, "2026-09-02T22:00:00+00:00")

    assert observations[0].service_date == "2026-09-03"
