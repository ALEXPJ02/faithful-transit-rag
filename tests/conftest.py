"""Shared fixtures.

The GTFS-Realtime protobuf classes are the only awkward dependency in the
test suite: building a real ``FeedMessage`` is the honest way to test the
parser, so tests that need one skip cleanly when the optional ``realtime``
extra is not installed.
"""

from __future__ import annotations

from typing import Any, TypeAlias

import pytest

gtfs_realtime_pb2 = pytest.importorskip(
    "google.transit.gtfs_realtime_pb2",
    reason="install the 'realtime' extra: pip install -e '.[realtime]'",
)

# (stop_id, stop_sequence, arrival_delay, departure_delay)
#
# A delay of None means the StopTimeEvent is absent entirely. TIME_ONLY means
# it is present carrying a predicted time but no delay — the shape the real
# feed sends constantly, and the one a fixture that can only set .delay is
# structurally incapable of producing.
TIME_ONLY = object()
Delay: TypeAlias = "int | object | None"
Stop: TypeAlias = "tuple[str, int, Delay, Delay]"

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
