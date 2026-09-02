"""Tests for the feed -> observation transform."""

from __future__ import annotations

from typing import Any

from conftest import NO_SEQUENCE, TIME_ONLY
from transit_rag.realtime.parsing import (
    extract_delay_observations,
    service_day,
    summarise_routes,
)

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


class TestSummariseRoutes:
    """The probe exists to catch a static bundle and a realtime feed published
    in different versions — the failure that looks like success."""

    def test_counts_trips_per_route(self, make_feed: Any) -> None:
        feed = make_feed(
            [
                ("t1-a", "APS_1a", [("s", 1, 10, None)]),
                ("t1-b", "APS_1a", [("s", 1, 10, None)]),
                ("t4-a", "APS_4a", [("s", 1, 10, None)]),
            ]
        )

        summaries = summarise_routes(feed, LOOKUP)

        assert [(s.route_id, s.trip_count) for s in summaries] == [
            ("APS_1a", 2),
            ("APS_4a", 1),
        ]

    def test_resolves_line_names_and_flags_unknown_ids(self, make_feed: Any) -> None:
        feed = make_feed(
            [
                ("known", "APS_1a", [("s", 1, 10, None)]),
                ("unknown", "SOMETHING_ELSE", [("s", 1, 10, None)]),
            ]
        )

        summaries = summarise_routes(feed, LOOKUP)
        by_id = {s.route_id: s for s in summaries}

        assert by_id["APS_1a"].route_short_name == "T1"
        assert by_id["APS_1a"].matched is True
        assert by_id["SOMETHING_ELSE"].route_short_name is None
        assert by_id["SOMETHING_ELSE"].matched is False

    def test_matched_routes_sort_ahead_of_unmatched(self, make_feed: Any) -> None:
        """The head of the list is what a human reads, so what they can act on
        goes there — even when the unmatched ids carry far more traffic."""
        feed = make_feed(
            [(f"u{i}", "UNKNOWN_X", [("s", 1, 10, None)]) for i in range(9)]
            + [("k", "APS_1a", [("s", 1, 10, None)])]
        )

        summaries = summarise_routes(feed, LOOKUP)

        assert summaries[0].route_id == "APS_1a"

    def test_ignores_entities_without_a_trip_update(self, make_feed: Any) -> None:
        feed = make_feed([("t", "APS_1a", [("s", 1, 10, None)])])
        feed.entity.add().id = "alert-only"

        assert sum(s.trip_count for s in summarise_routes(feed, LOOKUP)) == 1

    def test_empty_feed_summarises_to_nothing(self, make_feed: Any) -> None:
        assert summarise_routes(make_feed([]), LOOKUP) == []


class TestDelayPresence:
    """GTFS-Realtime has two levels of presence and both matter: the
    StopTimeEvent can exist while carrying no delay at all."""

    def test_a_stop_with_a_time_but_no_delay_is_not_a_zero_delay(self, make_feed: Any) -> None:
        """Reading .delay off a time-only event returns the proto default of
        0, fabricating an on-time observation in the training target — and
        one indistinguishable, afterwards, from a genuinely punctual train."""
        feed = make_feed([("trip-1", "APS_1a", [("stop-a", 1, TIME_ONLY, TIME_ONLY)])])

        observations = extract_delay_observations(feed, LOOKUP, TRACKED, POLL_TIME)

        assert observations == []

    def test_a_time_only_arrival_alongside_a_real_departure_delay(self, make_feed: Any) -> None:
        feed = make_feed([("trip-1", "APS_1a", [("stop-a", 1, TIME_ONLY, 45)])])

        observation = extract_delay_observations(feed, LOOKUP, TRACKED, POLL_TIME)[0]

        assert observation.arrival_delay_s is None
        assert observation.departure_delay_s == 45

    def test_a_genuine_zero_delay_is_still_recorded(self, make_feed: Any) -> None:
        """The fix must not throw away real on-time observations — they are
        the majority class the model has to learn."""
        feed = make_feed([("trip-1", "APS_1a", [("stop-a", 1, 0, None)])])

        observation = extract_delay_observations(feed, LOOKUP, TRACKED, POLL_TIME)[0]

        assert observation.arrival_delay_s == 0


class TestFieldPresence:
    def test_an_unset_stop_sequence_is_none_not_zero(self, make_feed: Any) -> None:
        feed = make_feed([("trip-1", "APS_1a", [("stop-a", NO_SEQUENCE, 60, None)])])

        assert extract_delay_observations(feed, LOOKUP, TRACKED, POLL_TIME)[0].stop_sequence is None

    def test_a_trip_without_an_id_is_skipped(self, make_feed: Any) -> None:
        """Every such row would collapse onto one (service_date, '', stop_id)
        key and overwrite the last."""
        feed = make_feed([("", "APS_1a", [("stop-a", 1, 60, None)])])

        assert extract_delay_observations(feed, LOOKUP, TRACKED, POLL_TIME) == []


class TestServiceDay:
    def test_after_midnight_belongs_to_the_previous_service_date(self) -> None:
        """00:20 Sydney is still the previous service day, which is what GTFS
        start_date says too. Rolling over at local midnight would make the
        derived date disagree with the reported one for the same stop event."""
        assert service_day("2026-09-02T14:20:00+00:00") == "2026-09-02"

    def test_the_boundary_sits_in_the_service_gap(self) -> None:
        assert service_day("2026-09-02T16:59:00+00:00") == "2026-09-02"  # 02:59 Sydney
        assert service_day("2026-09-02T17:01:00+00:00") == "2026-09-03"  # 03:01 Sydney

    def test_the_derived_date_agrees_with_a_reported_start_date(self, make_feed: Any) -> None:
        after_midnight = "2026-09-02T14:20:00+00:00"
        with_start = make_feed([("a", "APS_1a", [("s", 1, 60, None)], "20260902")])
        without_start = make_feed([("b", "APS_1a", [("s", 1, 60, None)])])

        reported = extract_delay_observations(with_start, LOOKUP, TRACKED, after_midnight)[0]
        derived = extract_delay_observations(without_start, LOOKUP, TRACKED, after_midnight)[0]

        assert reported.service_date == derived.service_date
