"""The static timetable, indexed for a specific set of trips.

The realtime feed does not populate ``stop_sequence`` — every observation
carries the ``-1`` sentinel — so the order of stops within a trip has to come
from the static bundle's ``stop_times.txt``. That file is also the only source
of scheduled arrival times.

**The join is lossy, by design of TfNSW's ids.** A realtime ``trip_id`` looks
like ``162F.1396.159.32.A.8.90986110``, where ``1396.159.32`` encodes the
timetable and version the trip was planned under. Trips already running when a
new timetable is published keep the old version and will not be in the current
bundle.

**One bundle cannot cover a window that spans a republication.** Measured over
3-9 September 2026, TfNSW rolled the timetable over between Sunday the 6th and
Monday the 7th. The bundle fetched on the 3rd matched 99-100% of trips before
that boundary and 0-1% after; a bundle fetched on the 10th did the exact
reverse. Each alone matched roughly half the window (52% and 47%); together they
matched **100%** of it.

So refreshing the bundle does not simply "raise the rate" — past a rollover it
trades one era's coverage for the other's. Keep the superseded bundle and pass
both to :meth:`ScheduleIndex.across_bundles`. The rate is still measured on
every run rather than assumed (:meth:`ScheduleIndex.coverage`); a drop now means
an era is missing a bundle, not merely that the current one has aged.
"""

from __future__ import annotations

import csv
import io
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


def parse_gtfs_time(value: str) -> int | None:
    """``HH:MM:SS`` to seconds after the start of the service day.

    Hours past 23 are legal and meaningful in GTFS: ``25:10:00`` is 1:10am on
    the service day that began the previous morning. Parsing this as a clock
    time would silently reject every after-midnight service.
    """
    parts = value.strip().split(":")
    if len(parts) != 3:
        return None
    try:
        hours, minutes, seconds = (int(part) for part in parts)
    except ValueError:
        return None
    return hours * 3600 + minutes * 60 + seconds


@dataclass(frozen=True)
class ScheduledStop:
    """One planned call, as the timetable describes it."""

    stop_sequence: int
    scheduled_arrival_s: int | None
    scheduled_departure_s: int | None


class ScheduleIndex:
    """Stop order and scheduled times, for a bounded set of trips.

    Loaded per-trip rather than wholesale: ``stop_times.txt`` for the Sydney
    Trains bundle covers ~83,000 trips and is far larger than memory should be
    spent on, when a reconciliation run only ever asks about the few thousand
    trips it actually observed.
    """

    def __init__(self, stops: dict[tuple[str, str], ScheduledStop], known_trips: set[str]) -> None:
        self._stops = stops
        self._known_trips = known_trips

    @classmethod
    def for_trips(cls, bundle: Path, trip_ids: Iterable[str]) -> ScheduleIndex:
        """Index only the given trips, streaming the bundle once."""
        wanted = set(trip_ids)
        stops: dict[tuple[str, str], ScheduledStop] = {}
        known: set[str] = set()

        with zipfile.ZipFile(bundle) as archive:
            if "stop_times.txt" not in archive.namelist():
                raise FileNotFoundError(f"No stop_times.txt in {bundle}")
            with archive.open("stop_times.txt") as handle:
                # Streamed, not read().decode(): the file is hundreds of MB
                # uncompressed and only a fraction of it is ever wanted.
                text = io.TextIOWrapper(handle, encoding="utf-8-sig", newline="")
                for row in csv.DictReader(text):
                    trip_id = row["trip_id"]
                    if trip_id not in wanted:
                        continue
                    known.add(trip_id)
                    try:
                        sequence = int(row["stop_sequence"])
                    except (KeyError, ValueError):
                        continue
                    stops[(trip_id, row["stop_id"])] = ScheduledStop(
                        stop_sequence=sequence,
                        scheduled_arrival_s=parse_gtfs_time(row.get("arrival_time", "")),
                        scheduled_departure_s=parse_gtfs_time(row.get("departure_time", "")),
                    )
        return cls(stops, known)

    @classmethod
    def across_bundles(cls, bundles: Iterable[Path], trip_ids: Iterable[str]) -> ScheduleIndex:
        """Index the given trips across several bundles, merging the results.

        A collection window that spans a timetable republication is only ever
        *partly* covered by any single bundle, because a realtime ``trip_id``
        embeds the version it was planned under. Measured over 3-9 September
        2026, TfNSW rolled the timetable over between Sunday the 6th and Monday
        the 7th: the bundle fetched on the 3rd matched 99-100% of trips before
        the boundary and 0-1% after it, and a bundle fetched on the 10th did the
        reverse. Each alone matched about half the window; merged, they matched
        **all** of it.

        Passing one bundle is the same as :meth:`for_trips`. Earlier bundles win
        on conflict, which only arises for a trip both describe — the same trip,
        so the entries are equivalent.
        """
        wanted = set(trip_ids)
        stops: dict[tuple[str, str], ScheduledStop] = {}
        known: set[str] = set()
        for bundle in bundles:
            era = cls.for_trips(bundle, wanted)
            for key, scheduled in era._stops.items():
                stops.setdefault(key, scheduled)
            known |= era._known_trips
        return cls(stops, known)

    def lookup(self, trip_id: str, stop_id: str) -> ScheduledStop | None:
        return self._stops.get((trip_id, stop_id))

    def knows_trip(self, trip_id: str) -> bool:
        return trip_id in self._known_trips

    def coverage(self, trip_ids: Iterable[str]) -> tuple[int, int]:
        """``(matched, total)`` over the given trips.

        Reported on every run. A falling rate has two causes and they need
        opposite responses: the bundle has aged behind the timetable (re-fetch
        it), or the window now spans a republication and an era has no bundle
        at all (keep the superseded one and pass both -- re-fetching alone just
        swaps which half matches).
        """
        requested = set(trip_ids)
        return len(requested & self._known_trips), len(requested)

    def __len__(self) -> int:
        return len(self._stops)
