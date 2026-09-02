"""Pure transforms over a parsed GTFS-Realtime feed.

Nothing here touches the network or the database, so every branch is unit
testable against a synthetic ``FeedMessage``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StopDelayObservation:
    """One predicted delay, for one stop of one trip, at one poll instant.

    This is an *observation of a prediction*, not an outcome. GTFS-Realtime
    only reports delay for stops a trip has not yet reached; once a vehicle
    passes a stop, that stop leaves the feed. The last observation naming a
    given ``(trip_id, stop_id)`` is therefore the closest available proxy
    for what actually happened — reconciling that is a separate offline step
    (see ``transit_rag.prediction.collection.reconcile``).
    """

    poll_time_utc: str
    trip_id: str
    route_id: str
    route_short_name: str | None
    stop_id: str
    stop_sequence: int | None
    arrival_delay_s: int | None
    departure_delay_s: int | None
    schedule_relationship: str

    def as_row(self) -> tuple[Any, ...]:
        """Column order matches the ``raw_polls`` table definition."""
        return (
            self.poll_time_utc,
            self.trip_id,
            self.route_id,
            self.route_short_name,
            self.stop_id,
            self.stop_sequence,
            self.arrival_delay_s,
            self.departure_delay_s,
            self.schedule_relationship,
        )


def _schedule_relationship_name(value: int) -> str:
    from google.transit import gtfs_realtime_pb2

    try:
        return str(gtfs_realtime_pb2.TripUpdate.StopTimeUpdate.ScheduleRelationship.Name(value))
    except ValueError:
        return f"UNKNOWN({value})"


def extract_delay_observations(
    feed: Any,
    route_lookup: Mapping[str, str],
    tracked_routes: Iterable[str],
    poll_time_utc: str,
) -> list[StopDelayObservation]:
    """Flatten a Trip Update feed into per-stop delay observations.

    When ``route_lookup`` is empty the feed is logged unfiltered: that is a
    louder failure mode than silently collecting nothing, which is what
    filtering against an empty lookup would do.
    """
    tracked = set(tracked_routes)
    observations: list[StopDelayObservation] = []

    for entity in feed.entity:
        if not entity.HasField("trip_update"):
            continue
        trip_update = entity.trip_update
        route_id = trip_update.trip.route_id
        route_short_name = route_lookup.get(route_id)

        if route_lookup and tracked and route_short_name not in tracked:
            continue

        for stop_time_update in trip_update.stop_time_update:
            arrival_delay = (
                stop_time_update.arrival.delay if stop_time_update.HasField("arrival") else None
            )
            departure_delay = (
                stop_time_update.departure.delay if stop_time_update.HasField("departure") else None
            )
            if arrival_delay is None and departure_delay is None:
                continue

            observations.append(
                StopDelayObservation(
                    poll_time_utc=poll_time_utc,
                    trip_id=trip_update.trip.trip_id,
                    route_id=route_id,
                    route_short_name=route_short_name,
                    stop_id=stop_time_update.stop_id,
                    stop_sequence=stop_time_update.stop_sequence,
                    arrival_delay_s=arrival_delay,
                    departure_delay_s=departure_delay,
                    schedule_relationship=_schedule_relationship_name(
                        stop_time_update.schedule_relationship
                    ),
                )
            )
    return observations
