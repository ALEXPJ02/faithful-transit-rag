"""Pure transforms over a parsed GTFS-Realtime feed.

Nothing here touches the network or the database, so every branch is unit
testable against a synthetic ``FeedMessage``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
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
# ~1M rows a day and roughly 60k (see docs/01-architecture.md §5).
DEFAULT_MAX_UPCOMING_STOPS = 3

# Only SCHEDULED calls are observations of a delay.
#
# SKIPPED means the train did not stop there at all, so its "delay" is a
# prediction for an event that never happened — and it would land in the
# training target beside real ones, indistinguishable afterwards. NO_DATA
# explicitly means no realtime information is available; a producer emitting
# an empty StopTimeEvent under it reintroduces the fabricated zero one level
# up. Both are recoverable from the stored schedule_relationship if a later
# analysis wants them, but neither belongs in the default collection.
COLLECTED_SCHEDULE_RELATIONSHIPS = frozenset({"SCHEDULED"})

# Transit service days do not end at midnight — a trip departing 23:50 belongs
# to the day it started, and GTFS says so via the trip's own ``start_date``.
# When that field is absent we have to derive one, and rolling over at local
# midnight would disagree with GTFS for every after-midnight trip: the same
# stop event would land under two different service dates depending on whether
# start_date happened to be populated, splitting the very key the upsert
# depends on. Backing off three hours puts the boundary in the service gap.
SERVICE_DAY_START_HOUR = 3


def service_day(instant_utc: str) -> str:
    """The Sydney service date an instant belongs to (see above)."""
    local = datetime.fromisoformat(instant_utc).astimezone(SYDNEY)
    return (local - timedelta(hours=SERVICE_DAY_START_HOUR)).date().isoformat()


@dataclass(frozen=True)
class StopDelayObservation:
    """The latest known predicted delay for one stop of one trip.

    This is an *observation of a prediction*, not an outcome. GTFS-Realtime
    only reports delay for stops a trip has not yet reached; once a vehicle
    passes a stop, that stop leaves the feed. The last observation naming a
    given ``(service_date, trip_id, stop_id, stop_sequence)`` is therefore the closest
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


def _delay_seconds(stop_time_update: Any, field: str) -> int | None:
    """Read ``arrival.delay`` / ``departure.delay``, or None if absent.

    Two levels of presence, and both matter. ``HasField(field)`` only says the
    StopTimeEvent sub-message exists — it is routinely present carrying just a
    predicted ``time`` and no ``delay``. Reading ``.delay`` off that returns
    the proto default of 0, which would record a fabricated on-time
    observation in the column the model is trained to predict. Indistinguish-
    able, afterwards, from a train that was genuinely on time.
    """
    if not stop_time_update.HasField(field):
        return None
    event = getattr(stop_time_update, field)
    if not event.HasField("delay"):
        return None
    return int(event.delay)


def _optional_int(message: Any, field: str) -> int | None:
    if not message.HasField(field):
        return None
    return int(getattr(message, field))


def _schedule_relationship_name(value: int) -> str:
    from google.transit import gtfs_realtime_pb2

    try:
        return str(gtfs_realtime_pb2.TripUpdate.StopTimeUpdate.ScheduleRelationship.Name(value))
    except ValueError:
        return f"UNKNOWN({value})"


def _service_date(trip: Any, fallback_utc: str) -> str:
    """The trip's GTFS start_date, or the Sydney-local date of the poll.

    ``start_date`` is optional in the spec and TfNSW does not always set it, so
    a fallback has to exist. See :func:`service_day` for why that fallback uses
    a 3am boundary rather than midnight.
    """
    raw = getattr(trip, "start_date", "") or ""
    if len(raw) == 8 and raw.isdigit():
        return f"{raw[0:4]}-{raw[4:6]}-{raw[6:8]}"
    return service_day(fallback_utc)


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

        if not trip_update.trip.trip_id:
            # Without one there is no key: every such row would collapse onto
            # a single (service_date, "", stop_id) and overwrite the last.
            continue

        service_date = _service_date(trip_update.trip, poll_time_utc)
        kept = 0

        for stops_ahead, stop_time_update in enumerate(trip_update.stop_time_update):
            if 0 < max_upcoming_stops <= kept:
                break

            relationship = _schedule_relationship_name(stop_time_update.schedule_relationship)
            if relationship not in COLLECTED_SCHEDULE_RELATIONSHIPS:
                continue

            arrival_delay = _delay_seconds(stop_time_update, "arrival")
            departure_delay = _delay_seconds(stop_time_update, "departure")
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
                    stop_sequence=_optional_int(stop_time_update, "stop_sequence"),
                    stops_ahead=stops_ahead,
                    arrival_delay_s=arrival_delay,
                    departure_delay_s=departure_delay,
                    schedule_relationship=relationship,
                    observed_at_utc=poll_time_utc,
                )
            )
    return observations


@dataclass(frozen=True)
class RouteSummary:
    """How many trips a feed carried for one route_id, and whether the static
    bundle knows that id."""

    route_id: str
    route_short_name: str | None
    trip_count: int

    @property
    def matched(self) -> bool:
        return self.route_short_name is not None


def summarise_routes(feed: Any, route_lookup: Mapping[str, str]) -> list[RouteSummary]:
    """Count trips per route_id in a feed, resolved against the lookup.

    Exists to diagnose the quietest failure in the whole pipeline: the static
    bundle and the realtime feed are published in versioned pairs, and mixing
    versions gives a feed full of route_ids that the lookup has never heard
    of. Everything still "works" — the fetch succeeds, the parse succeeds —
    and zero rows are collected, for days. This turns that into a sentence.
    """
    counts: dict[str, int] = {}
    for entity in feed.entity:
        if not entity.HasField("trip_update"):
            continue
        route_id = entity.trip_update.trip.route_id
        counts[route_id] = counts.get(route_id, 0) + 1

    summaries = [
        RouteSummary(route_id, route_lookup.get(route_id), count)
        for route_id, count in counts.items()
    ]
    # Matched routes first, then by how much traffic each carries — the head of
    # the list is what you actually need to read.
    summaries.sort(key=lambda s: (not s.matched, -s.trip_count, s.route_id))
    return summaries
