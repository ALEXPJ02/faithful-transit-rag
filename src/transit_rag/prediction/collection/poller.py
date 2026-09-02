"""Collect GTFS-Realtime delay observations for the tracked Sydney Trains lines.

Two modes, because collection has to survive a laptop lid closing:

    transit-poller                    # loop forever (systemd / an always-on box)
    transit-poller --once             # a single poll (scheduled runs — see docs/04)
    transit-poller --once --sink csv  # stateless: write one snapshot file
    transit-poller --status           # how much data so far, and are polls succeeding

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

from transit_rag.config import CollectionConfig, ConfigError, TfnswConfig
from transit_rag.prediction.collection.routes import load_routes_lookup
from transit_rag.prediction.collection.store import (
    CsvSnapshotStore,
    ObservationSink,
    SqliteObservationStore,
)
from transit_rag.realtime.client import FeedFetchError, GtfsRealtimeClient
from transit_rag.realtime.parsing import extract_delay_observations

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
    """Run one poll. Returns True if the feed was fetched and stored."""
    poll_time = datetime.now(UTC).isoformat()

    try:
        feed = client.fetch_trip_updates()
    except FeedFetchError as exc:
        log.error("Poll failed: %s", exc)
        sink.record_poll(poll_time, 0, 0, f"error: {exc}")
        return False

    observations = extract_delay_observations(
        feed, route_lookup, tracked_routes, poll_time, max_upcoming_stops
    )
    written = sink.record_observations(observations)
    sink.record_poll(poll_time, len(feed.entity), written, "ok")
    log.info("poll ok — %d entities seen, %d observations written", len(feed.entity), written)
    return True


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
        "--sink",
        choices=("sqlite", "csv"),
        default=None,
        help="Where to write observations (default: COLLECTION_SINK, else sqlite)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    try:
        config = CollectionConfig()
        sink = build_sink(config, args.sink)
    except ConfigError as exc:
        log.error("%s", exc)
        sys.exit(1)

    if args.status:
        print_status(sink)
        sink.close()
        return

    try:
        tfnsw = TfnswConfig.from_env()
    except ConfigError as exc:
        log.error("%s", exc)
        sys.exit(1)

    signal.signal(signal.SIGINT, _request_shutdown)
    signal.signal(signal.SIGTERM, _request_shutdown)

    client = GtfsRealtimeClient(tfnsw, timeout_seconds=config.request_timeout_seconds)
    route_lookup = load_routes_lookup(config.routes_lookup_path)

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
