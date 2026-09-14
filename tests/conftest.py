"""Shared fixtures.

The GTFS-Realtime protobuf classes are the only awkward dependency in the
test suite: building a real ``FeedMessage`` is the honest way to test the
parser, so tests that need one skip cleanly when the optional ``realtime``
extra is not installed.
"""

from __future__ import annotations

from typing import Any

import pytest

gtfs_realtime_pb2 = pytest.importorskip(
    "google.transit.gtfs_realtime_pb2",
    reason="install the 'realtime' extra: pip install -e '.[realtime]'",
)

# Stops are (stop_id, stop_sequence, arrival_delay, departure_delay).
#
# A delay of None means the StopTimeEvent is absent entirely. TIME_ONLY means
# it is present carrying a predicted time but no delay — the shape the real
# feed sends constantly, and the one a fixture that can only set .delay is
# structurally incapable of producing.
TIME_ONLY = object()
# stop_sequence sentinel: leave the field unset rather than assigning 0.
NO_SEQUENCE = object()


@pytest.fixture
def make_feed() -> Any:
    """Build a Trip Update FeedMessage from plain Python descriptions.

    Each trip is ``(trip_id, route_id, [stops])``, optionally with a fourth
    element giving the GTFS ``start_date`` (``YYYYMMDD``).

    A delay of ``None`` omits the StopTimeEvent entirely; ``TIME_ONLY``
    includes it with a predicted time and no delay. Pass ``NO_SEQUENCE`` as
    the sequence to leave that field unset.
    """

    def _make(trips: list[tuple[Any, ...]]) -> Any:
        feed = gtfs_realtime_pb2.FeedMessage()
        feed.header.gtfs_realtime_version = "2.0"
        for trip in trips:
            trip_id, route_id, stops = trip[0], trip[1], trip[2]
            start_date = trip[3] if len(trip) > 3 else None

            entity = feed.entity.add()
            entity.id = trip_id
            entity.trip_update.trip.trip_id = trip_id
            entity.trip_update.trip.route_id = route_id
            if start_date:
                entity.trip_update.trip.start_date = start_date

            for stop_id, sequence, arrival_delay, departure_delay in stops:
                stop_time_update = entity.trip_update.stop_time_update.add()
                stop_time_update.stop_id = stop_id
                if sequence is not NO_SEQUENCE:
                    stop_time_update.stop_sequence = sequence
                for field, delay in (("arrival", arrival_delay), ("departure", departure_delay)):
                    if delay is None:
                        continue
                    event = getattr(stop_time_update, field)
                    if delay is TIME_ONLY:
                        event.time = 1_788_000_000  # present, but no delay
                    else:
                        event.delay = delay
        return feed

    return _make


# active_period sentinels, mirroring TIME_ONLY/NO_SEQUENCE above: a fixture
# that can only set both ends cannot produce the half-open windows the real
# feed sends, nor the empty active_period the spec defines as "always active".
NO_PERIODS = object()
UNSET = object()


@pytest.fixture
def make_alert_feed() -> Any:
    """Build a Service Alerts FeedMessage from plain Python descriptions.

    Each alert is a dict:

    ``id``          entity id (the feed's stable UUID)
    ``selectors``   list of ``(route_id, direction_id, stop_id)``; pass
                    :data:`UNSET` for a direction or ``""`` for a stop to
                    leave the field unset
    ``periods``     list of ``(start, end)``; :data:`UNSET` for either end
                    leaves it unset, and :data:`NO_PERIODS` omits
                    ``active_period`` entirely
    ``cause`` / ``effect`` / ``severity``   enum ints, optional
    ``header`` / ``description`` / ``url``  either a string, or a list of
                    ``(language, text)`` pairs so a test can reproduce the
                    real feed's ``en`` + ``en/html`` pair
    """

    def _translate(message: Any, value: Any) -> None:
        pairs = value if isinstance(value, list) else [("en", value)]
        for language, text in pairs:
            translation = message.translation.add()
            translation.language = language
            translation.text = text

    def _make(alerts: list[dict[str, Any]]) -> Any:
        feed = gtfs_realtime_pb2.FeedMessage()
        feed.header.gtfs_realtime_version = "2.0"
        for spec in alerts:
            entity = feed.entity.add()
            entity.id = spec.get("id", "")
            alert = entity.alert

            for field in ("cause", "effect", "severity_level"):
                if field in spec:
                    setattr(alert, field, spec[field])

            for key, target in (
                ("header", "header_text"),
                ("description", "description_text"),
                ("url", "url"),
            ):
                if key in spec:
                    _translate(getattr(alert, target), spec[key])

            periods = spec.get("periods", NO_PERIODS)
            if periods is not NO_PERIODS:
                for start, end in periods:
                    time_range = alert.active_period.add()
                    if start is not UNSET:
                        time_range.start = start
                    if end is not UNSET:
                        time_range.end = end

            for route_id, direction_id, stop_id in spec.get("selectors", []):
                selector = alert.informed_entity.add()
                if route_id:
                    selector.route_id = route_id
                if stop_id:
                    selector.stop_id = stop_id
                if direction_id is not UNSET:
                    selector.direction_id = direction_id
        return feed

    return _make
