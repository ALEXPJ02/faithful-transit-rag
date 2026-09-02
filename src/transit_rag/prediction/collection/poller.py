"""Collect GTFS-Realtime delay observations for the tracked Sydney Trains lines.

Two modes, because collection has to survive a laptop lid closing:

    transit-poller                    # loop forever (systemd / an always-on box)
    transit-poller --once             # a single poll (scheduled runs — see docs/04)
    transit-poller --once --sink csv  # stateless: write one snapshot file
    transit-poller --status           # how much data so far, and are polls succeeding
    transit-poller --probe            # what the feed contains, and does the filter match
    transit-poller --once --require-routes   # refuse to run unfiltered (scheduled runs)

Every day of missed collection is a day of training data that cannot be
recovered later, so ``--status`` exists to make silent failure loud: an auth
or endpoint problem shows up as consecutive ``error:`` rows in the poll log
while the process keeps happily running.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
from datetime import UTC, datetime
from types import FrameType
from typing import Any

from transit_rag.config import CollectionConfig, ConfigError, TfnswConfig
from transit_rag.prediction.collection.routes import load_routes_lookup
from transit_rag.prediction.collection.store import (
    CsvSnapshotStore,
    ObservationSink,
    SqliteObservationStore,
)
from transit_rag.realtime.client import FeedFetchError, GtfsRealtimeClient
from transit_rag.realtime.parsing import extract_delay_observations, summarise_routes

log = logging.getLogger("transit_rag.poller")

# An Event rather than a bool: `wait(timeout)` doubles as the inter-poll
# sleep, so SIGTERM is answered immediately instead of up to an interval
# later. That matters when the collector is a systemd unit or a container
# being cycled — a poll interrupted between polls loses nothing, but a
# shutdown that takes two minutes gets SIGKILLed.
_shutdown = threading.Event()


def _request_shutdown(signum: int, frame: FrameType | None) -> None:
    _shutdown.set()
    log.info("Signal %s received — finishing the current poll then stopping.", signum)


def build_sink(config: CollectionConfig, sink_name: str | None = None) -> ObservationSink:
    """Construct the configured sink. ``sink_name`` overrides the environment."""
    chosen = (sink_name or config.sink).lower()
    if chosen == "csv":
        return CsvSnapshotStore(config.snapshot_dir)
    if chosen == "sqlite":
        return SqliteObservationStore(config.db_path)
    raise ConfigError(f"Unknown sink {chosen!r} — expected 'sqlite' or 'csv'.")


def poll_once(
    client: GtfsRealtimeClient,
    sink: ObservationSink,
    route_lookup: dict[str, str],
    tracked_routes: tuple[str, ...],
    max_upcoming_stops: int = 3,
) -> bool:
    """Run one poll. Returns True if the feed was fetched and stored.

    Nothing raised in here escapes. A collector that has been running for
    weeks must not die on a malformed feed, a full disk, or a locked
    database — the cost of one lost poll is a two-minute gap; the cost of an
    unnoticed dead process is the rest of the collection window.
    """
    poll_time = datetime.now(UTC).isoformat()

    try:
        feed = client.fetch_trip_updates()
        observations = extract_delay_observations(
            feed, route_lookup, tracked_routes, poll_time, max_upcoming_stops
        )
        written = sink.record_observations(observations)

        mismatch = _lookup_mismatch(feed, route_lookup)
        if mismatch is not None:
            # Not an empty night: the feed is carrying trips and not one of
            # their route_ids is known to the lookup. That is a static bundle
            # and a realtime feed from different versions, and left alone it
            # collects zero rows behind an "ok" poll log for as long as nobody
            # looks. It can also appear mid-collection, when TfNSW republishes
            # the bundle and route_ids shift under a collector that has been
            # working for weeks.
            log.error("%s", mismatch)
            sink.record_poll(poll_time, len(feed.entity), written, f"error: {mismatch}")
            return False

        sink.record_poll(poll_time, len(feed.entity), written, "ok")
    except FeedFetchError as exc:
        # Routine and expected — a 502 or a timeout. No traceback needed.
        log.error("Poll failed: %s", exc)
        _try_record_failure(sink, poll_time, f"error: {exc}")
        return False
    except Exception as exc:
        log.exception("Poll failed unexpectedly")
        _try_record_failure(sink, poll_time, f"error: {type(exc).__name__}: {exc}")
        return False

    log.info("poll ok — %d entities seen, %d observations written", len(feed.entity), written)
    return True


def _lookup_mismatch(feed: Any, route_lookup: dict[str, str]) -> str | None:
    """Describe a feed whose route_ids are all unknown to the lookup, else None.

    Deliberately narrow. Zero *tracked* trips is normal overnight and must not
    raise anything; zero *resolvable* trips, while trips are being published,
    can only mean the two data sources disagree.
    """
    if not route_lookup:
        return None
    summaries = summarise_routes(feed, route_lookup)
    if not summaries or any(summary.matched for summary in summaries):
        return None
    return (
        f"none of the {len(summaries)} route_ids in this feed resolve against the route "
        f"lookup — the static bundle and the realtime feed are from different versions. "
        f"Re-download the bundle that pairs with this feed and rebuild the lookup; "
        f"run `transit-poller --probe` to confirm."
    )


def _try_record_failure(sink: ObservationSink, poll_time: str, status: str) -> None:
    """Record a failed poll, tolerating a sink that is itself the problem."""
    try:
        sink.record_poll(poll_time, 0, 0, status)
    except Exception:
        log.exception("Could not record the failed poll either")


def run_forever(
    client: GtfsRealtimeClient,
    sink: ObservationSink,
    route_lookup: dict[str, str],
    config: CollectionConfig,
) -> None:
    log.info(
        "Collecting every %ss into %s, tracking %s, keeping %d upcoming stops per trip",
        config.interval_seconds,
        sink.describe(),
        ", ".join(config.tracked_routes) or "all lines",
        config.max_upcoming_stops,
    )
    while not _shutdown.is_set():
        poll_once(client, sink, route_lookup, config.tracked_routes, config.max_upcoming_stops)
        if _shutdown.wait(timeout=config.interval_seconds):
            break
    log.info("Stopped. %d observations collected in total.", sink.observation_count())


def print_probe(
    client: GtfsRealtimeClient, route_lookup: dict[str, str], tracked_routes: tuple[str, ...]
) -> bool:
    """Fetch once and report what the feed actually contains.

    Answers the two questions a failed first run leaves open — is the endpoint
    right, and does the static bundle match the feed — without writing
    anything.
    """
    try:
        feed = client.fetch_trip_updates()
    except FeedFetchError as exc:
        print(f"Could not read the feed:\n  {exc}")
        return False

    summaries = summarise_routes(feed, route_lookup)
    trip_updates = sum(s.trip_count for s in summaries)
    print(f"Entities: {len(feed.entity)}  (trip updates: {trip_updates})")
    print(f"Distinct route_ids: {len(summaries)}")

    if not route_lookup:
        print("\nNo route lookup loaded — build it before collecting (docs/05 §5).")
        return trip_updates > 0

    print(f"\n{'route_id':<24} {'line':<8} trips")
    for summary in summaries[:20]:
        line = summary.route_short_name or "—"
        print(f"  {summary.route_id:<22} {line:<8} {summary.trip_count}")
    if len(summaries) > 20:
        print(f"  … and {len(summaries) - 20} more")

    unmatched = [s for s in summaries if not s.matched]
    tracked_present = {
        s.route_short_name: s.trip_count for s in summaries if s.route_short_name in tracked_routes
    }

    print()
    if unmatched and not tracked_present:
        # The version-mismatch signature: a feed full of ids the bundle has
        # never seen. Worth naming explicitly, because everything else looks
        # healthy.
        print(
            f"None of the {len(summaries)} route_ids in this feed are in the route lookup.\n"
            "That is what a static bundle and a realtime feed from different versions\n"
            "looks like — re-download the static bundle that pairs with this feed."
        )
        return False
    if not tracked_present:
        print(
            f"The lookup matched, but none of {', '.join(tracked_routes)} are running right now.\n"
            "Normal overnight; re-probe during service hours before concluding anything."
        )
        return True

    for line, count in sorted(tracked_present.items()):
        print(f"{line}: {count} active trip{'' if count == 1 else 's'}")
    if unmatched:
        print(f"({len(unmatched)} route_ids not in the lookup — fine, they are other lines.)")
    print("\nFeed and lookup agree. Safe to start collecting.")
    return True


def print_status(sink: ObservationSink) -> None:
    print(f"Store: {sink.describe()}")
    print(f"Observations collected: {sink.observation_count():,}")

    date_range = getattr(sink, "service_date_range", None)
    if callable(date_range):
        span = date_range()
        if span:
            print(f"Service dates covered: {span[0]} to {span[1]}")

    recent = sink.recent_poll_status(limit=10)
    if not recent:
        print("No polls recorded yet.")
        return

    failures = sum(1 for row in recent if not row[3].startswith("ok"))
    print(f"\nLast {len(recent)} polls ({failures} failed):")
    for poll_time, entities, rows, status in recent:
        print(f"  {poll_time}  entities={entities:<6} rows={rows:<6} {status[:80]}")
    if failures == len(recent):
        print("\nEvery recent poll failed — check the API key and the endpoint URL.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--once", action="store_true", help="Run a single poll and exit")
    parser.add_argument("--status", action="store_true", help="Report collection progress and exit")
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Fetch once and report what the feed contains, without storing anything",
    )
    parser.add_argument(
        "--sink",
        choices=("sqlite", "csv"),
        default=None,
        help="Where to write observations (default: COLLECTION_SINK, else sqlite)",
    )
    parser.add_argument(
        "--require-routes",
        action="store_true",
        help="Refuse to collect if the route lookup is missing (use for scheduled runs)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    try:
        config = CollectionConfig()
    except ConfigError as exc:
        log.error("%s", exc)
        sys.exit(1)

    if args.status:
        # Built here rather than up front so --probe stays side-effect free:
        # constructing the SQLite sink creates the database file, which is a
        # surprising thing for a command documented as storing nothing to do.
        try:
            sink = build_sink(config, args.sink)
        except ConfigError as exc:
            log.error("%s", exc)
            sys.exit(1)
        print_status(sink)
        sink.close()
        return

    try:
        tfnsw = TfnswConfig.from_env()
    except ConfigError as exc:
        log.error("%s", exc)
        sys.exit(1)

    client = GtfsRealtimeClient(tfnsw, timeout_seconds=config.request_timeout_seconds)
    route_lookup = load_routes_lookup(config.routes_lookup_path)

    if args.probe:
        sys.exit(0 if print_probe(client, route_lookup, config.tracked_routes) else 1)

    if not route_lookup and (args.require_routes or config.require_routes_lookup):
        log.error(
            "No route lookup at %s, and this run requires one. Without it every line is "
            "collected unfiltered, which in a scheduled job is weeks of unattributed rows "
            "behind a green tick. Build it with "
            "`python -m transit_rag.prediction.collection.routes <gtfs.zip>` and commit it.",
            config.routes_lookup_path,
        )
        sys.exit(1)

    signal.signal(signal.SIGINT, _request_shutdown)
    signal.signal(signal.SIGTERM, _request_shutdown)

    try:
        sink = build_sink(config, args.sink)
    except ConfigError as exc:
        log.error("%s", exc)
        sys.exit(1)

    try:
        if args.once:
            ok = poll_once(
                client, sink, route_lookup, config.tracked_routes, config.max_upcoming_stops
            )
            print_status(sink)
            sys.exit(0 if ok else 1)
        run_forever(client, sink, route_lookup, config)
    finally:
        sink.close()


if __name__ == "__main__":
    main()
