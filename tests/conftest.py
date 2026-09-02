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


@pytest.fixture
def make_feed() -> Any:
    """Build a Trip Update FeedMessage from plain Python descriptions.

    Each trip is ``(trip_id, route_id, [(stop_id, sequence, arrival_delay,
    departure_delay), ...])``. A delay of ``None`` leaves that field unset,
    which is how the real feed represents "no prediction".
    """

    def _make(trips: list[tuple[str, str, list[tuple[str, int, int | None, int | None]]]]) -> Any:
        feed = gtfs_realtime_pb2.FeedMessage()
        feed.header.gtfs_realtime_version = "2.0"
        for trip_id, route_id, stops in trips:
            entity = feed.entity.add()
            entity.id = trip_id
            entity.trip_update.trip.trip_id = trip_id
            entity.trip_update.trip.route_id = route_id
            for stop_id, sequence, arrival_delay, departure_delay in stops:
                stop_time_update = entity.trip_update.stop_time_update.add()
                stop_time_update.stop_id = stop_id
                stop_time_update.stop_sequence = sequence
                if arrival_delay is not None:
                    stop_time_update.arrival.delay = arrival_delay
                if departure_delay is not None:
                    stop_time_update.departure.delay = departure_delay
        return feed

    return _make
