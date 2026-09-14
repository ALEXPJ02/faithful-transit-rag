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


# An alert whose ``active_period`` is empty is active whenever it is published
# (GTFS-Realtime spec). Storing that as "no rows" would make such an alert
# vanish entirely, so it is stored as a single unbounded window instead.
UNBOUNDED = 0

# ``direction_id`` is optional on an EntitySelector. SQLite treats NULLs in a
# composite primary key as distinct from one another, so a nullable key column
# silently stops deduplicating -- the same trap ``stop_sequence`` already
# carries a sentinel for. -1 is not a valid GTFS direction.
NO_DIRECTION = -1


@dataclass(frozen=True)
class ServiceAlert:
    """One published alert, independent of what it applies to.

    Split from :class:`AlertScope` because the two have very different
    cardinalities: a single alert routinely names dozens of routes and stops,
    and flattening them together would store a ~400 character
    ``description_text`` once per selector per poll.
    """

    alert_id: str
    cause: str
    effect: str
    severity: str
    header_text: str
    description_text: str
    url: str
    observed_at_utc: str

    def as_dict(self) -> dict[str, Any]:
        """Field name -> value. Sinks map this to their own column order."""
        return asdict(self)


@dataclass(frozen=True)
class AlertScope:
    """What one alert applies to, for one of its active windows.

    The cross product of ``informed_entity`` and ``active_period``. Times are
    POSIX seconds rather than ISO strings — unlike the rest of this module —
    because the feed gives them that way, ``0`` is an unambiguous "unbounded"
    sentinel (no real alert starts at the Unix epoch), and the reconcile join
    that will eventually consume these is a numeric ``BETWEEN`` against an
    index.
    """

    alert_id: str
    route_id: str
    route_short_name: str | None
    direction_id: int
    stop_id: str
    active_period_start: int
    active_period_end: int
    observed_at_utc: str

    def as_dict(self) -> dict[str, Any]:
        """Field name -> value. Sinks map this to their own column order."""
        return asdict(self)


def _alert_enum_name(enum_name: str, value: int) -> str:
    """Resolve one of ``Alert``'s enum ints to its name, or ``UNKNOWN(n)``.

    Same shape and same deferred import as :func:`_schedule_relationship_name`.
    A value the bindings do not know is recorded rather than dropped, because a
    new TfNSW cause code should surface in the data as an unknown rather than
    silently becoming the proto default.
    """
    from google.transit import gtfs_realtime_pb2

    try:
        return str(getattr(gtfs_realtime_pb2.Alert, enum_name).Name(value))
    except ValueError:
        return f"UNKNOWN({value})"


def _translated_text(message: Any, language: str = "en") -> str:
    """The plain-text translation of a ``TranslatedString``.

    TfNSW sends ``description_text`` **twice**: once as prose tagged ``en``
    and once as a ~1.3 KB HTML fragment tagged ``en/html``. Taking
    ``translation[0]`` happens to pick the prose today, and nothing in the
    feed guarantees that ordering — the day it flips, every stored row gains a
    kilobyte of markup in the column the RAG layer reads as text. So: exact
    language match first, then any translation whose language carries no
    content-type suffix, and only then fall back to whatever came first.
    """
    translations = list(getattr(message, "translation", []))
    if not translations:
        return ""
    for translation in translations:
        if translation.language == language:
            return str(translation.text)
    for translation in translations:
        if "/" not in translation.language:
            return str(translation.text)
    return str(translations[0].text)


def _active_periods(alert: Any) -> list[tuple[int, int]]:
    """``(start, end)`` pairs in POSIX seconds, ``0`` meaning unbounded.

    An empty ``active_period`` yields one unbounded window rather than none —
    see :data:`UNBOUNDED`.
    """
    periods = [
        (
            int(period.start) if period.HasField("start") else UNBOUNDED,
            int(period.end) if period.HasField("end") else UNBOUNDED,
        )
        for period in alert.active_period
    ]
    return periods or [(UNBOUNDED, UNBOUNDED)]


def extract_alerts(
    feed: Any,
    route_lookup: Mapping[str, str],
    poll_time_utc: str,
) -> tuple[list[ServiceAlert], list[AlertScope]]:
    """Flatten a Service Alerts feed into alerts and the scopes they apply to.

    Note the absence of a ``tracked_routes`` parameter, unlike
    :func:`extract_delay_observations`. That is deliberate and not an
    oversight: measured against the live feed, 32% of the route ids named by
    alerts are absent from the static bundle's lookup entirely (intercity and
    regional services the For Realtime bundle never describes), and filtering
    to T1/T4 at collection time would discard 28 of 41 alerts permanently.
    Collection is irreversible; reconciliation is not. ``route_lookup`` is
    therefore used only to *annotate* — a miss records ``None`` and keeps the
    row — and the filtering happens downstream where it can be re-run.
    """
    alerts: list[ServiceAlert] = []
    scopes: list[AlertScope] = []

    for entity in feed.entity:
        if not entity.HasField("alert"):
            continue
        if not entity.id:
            # Without an id there is no key, and every such alert would
            # collapse onto a single row and overwrite the last.
            continue

        alert = entity.alert
        alerts.append(
            ServiceAlert(
                alert_id=entity.id,
                cause=_alert_enum_name("Cause", alert.cause),
                effect=_alert_enum_name("Effect", alert.effect),
                severity=_alert_enum_name("SeverityLevel", alert.severity_level),
                header_text=_translated_text(alert.header_text),
                description_text=_translated_text(alert.description_text),
                url=_translated_text(alert.url),
                observed_at_utc=poll_time_utc,
            )
        )

        periods = _active_periods(alert)
        # The live feed repeats the same (route, direction, stop) triple within
        # a single alert, and handing duplicate keys to one executemany batch
        # makes the upsert's row count meaningless. Dedup before emitting.
        seen: set[tuple[str, int, str]] = set()
        for informed in alert.informed_entity:
            direction = (
                int(informed.direction_id) if informed.HasField("direction_id") else NO_DIRECTION
            )
            selector = (informed.route_id, direction, informed.stop_id)
            if selector in seen:
                continue
            seen.add(selector)

            for start, end in periods:
                scopes.append(
                    AlertScope(
                        alert_id=entity.id,
                        route_id=informed.route_id,
                        route_short_name=route_lookup.get(informed.route_id),
                        direction_id=direction,
                        stop_id=informed.stop_id,
                        active_period_start=start,
                        active_period_end=end,
                        observed_at_utc=poll_time_utc,
                    )
                )

    return alerts, scopes
