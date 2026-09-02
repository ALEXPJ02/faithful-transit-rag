"""Pure transforms over a parsed GTFS-Realtime feed.

Nothing here touches the network or the database, so every branch is unit
testable against a synthetic ``FeedMessage``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

# Service dates are calendar days in the network's own timezone. Deriving one
# from a UTC instant without converting first would roll the date over at 10am
# Sydney time and split every morning peak across two service dates.
SYDNEY = ZoneInfo("Australia/Sydney")

# How many upcoming stops to keep per trip, per poll.
#
# The feed republishes a prediction for every stop a trip has not yet reached —
# twenty-odd rows per trip, nearly all of them a distant guess that will be
# revised many times before it matters. Only the imminent stops are close
# enough to the event to serve as an outcome proxy, and keeping three (rather
# than one) means a trip can pass two stops between polls without the middle
# one going unrecorded. This single constant is the difference between roughly
# 1M rows a day and roughly 25k.
DEFAULT_MAX_UPCOMING_STOPS = 3


@dataclass(frozen=True)
class StopDelayObservation:
    """The latest known predicted delay for one stop of one trip.

    This is an *observation of a prediction*, not an outcome. GTFS-Realtime
    only reports delay for stops a trip has not yet reached; once a vehicle
    passes a stop, that stop leaves the feed. The last observation naming a
    given ``(service_date, trip_id, stop_id)`` is therefore the closest
    available proxy for what actually happened — and ``stops_ahead`` records
    how close to the event that final prediction was made, so a later analysis
    can weight or filter on it rather than trusting all rows equally.
    """

    service_date: str
    trip_id: str
    stop_id: str
    route_id: str
    route_short_name: str | None
    stop_sequence: int | None
    stops_ahead: int
    arrival_delay_s: int | None
    departure_delay_s: int | None
    schedule_relationship: str
    observed_at_utc: str

    def as_dict(self) -> dict[str, Any]:
        """Field name -> value. Sinks map this to their own column order."""
        return asdict(self)


def _schedule_relationship_name(value: int) -> str:
    from google.transit import gtfs_realtime_pb2

    try:
        return str(gtfs_realtime_pb2.TripUpdate.StopTimeUpdate.ScheduleRelationship.Name(value))
    except ValueError:
        return f"UNKNOWN({value})"


def _service_date(trip: Any, fallback_utc: str) -> str:
    """The trip's GTFS start_date, or the Sydney-local date of the poll.

    ``start_date`` is optional in the spec and TfNSW does not always set it, so
    the fallback has to exist — but it must be the *local* date, not the UTC
    one, or every service date rolls over mid-morning.
    """
    raw = getattr(trip, "start_date", "") or ""
    if len(raw) == 8 and raw.isdigit():
        return f"{raw[0:4]}-{raw[4:6]}-{raw[6:8]}"
    return datetime.fromisoformat(fallback_utc).astimezone(SYDNEY).date().isoformat()


def extract_delay_observations(
    feed: Any,
    route_lookup: Mapping[str, str],
    tracked_routes: Iterable[str],
    poll_time_utc: str,
    max_upcoming_stops: int = DEFAULT_MAX_UPCOMING_STOPS,
) -> list[StopDelayObservation]:
    """Flatten a Trip Update feed into per-stop delay observations.

    Only the first ``max_upcoming_stops`` stops of each trip are kept; pass a
    value < 1 to keep all of them.

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

        service_date = _service_date(trip_update.trip, poll_time_utc)
        kept = 0

        for stops_ahead, stop_time_update in enumerate(trip_update.stop_time_update):
            if 0 < max_upcoming_stops <= kept:
                break

            arrival_delay = (
                stop_time_update.arrival.delay if stop_time_update.HasField("arrival") else None
            )
            departure_delay = (
                stop_time_update.departure.delay if stop_time_update.HasField("departure") else None
            )
            if arrival_delay is None and departure_delay is None:
                # Carries no signal, and must not consume one of the kept
                # slots — otherwise a trip whose next stop has no prediction
                # loses the stops behind it too.
                continue

            kept += 1
            observations.append(
                StopDelayObservation(
                    service_date=service_date,
                    trip_id=trip_update.trip.trip_id,
                    stop_id=stop_time_update.stop_id,
                    route_id=route_id,
                    route_short_name=route_short_name,
                    stop_sequence=stop_time_update.stop_sequence,
                    stops_ahead=stops_ahead,
                    arrival_delay_s=arrival_delay,
                    departure_delay_s=departure_delay,
                    schedule_relationship=_schedule_relationship_name(
                        stop_time_update.schedule_relationship
                    ),
                    observed_at_utc=poll_time_utc,
                )
            )
    return observations
