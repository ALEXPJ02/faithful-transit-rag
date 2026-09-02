"""Build and load the ``route_id -> route_short_name`` lookup.

The realtime feed identifies trips by ``route_id`` only (``ESI_1c``,
``NTH_1a``, ``RTTA_REV``); the human line name (``T1``, ``T4``) lives in the
static GTFS bundle's ``routes.txt``. Without this lookup the collector cannot
scope itself to the tracked lines.

**Which bundle.** It has to be the "For Realtime" one, served from
``/v1/gtfs/schedule/<operator>``. TfNSW's "Timetables Complete GTFS" states
outright that its *"identifiers do not match the GTFS-realtime APIs and
data"* — building the lookup from it resolves nothing, and the collector then
records zero rows behind a healthy-looking poll log. ``--fetch`` pulls the
right one with the API key already in ``.env``.
"""

from __future__ import annotations

import csv
import logging
import zipfile
from pathlib import Path

log = logging.getLogger(__name__)


def read_routes_txt(source: Path) -> list[dict[str, str]]:
    """Read ``routes.txt`` from a static GTFS zip or an unzipped directory."""
    if source.is_dir():
        routes_path = source / "routes.txt"
        if not routes_path.exists():
            raise FileNotFoundError(f"No routes.txt in {source}")
        with routes_path.open(newline="", encoding="utf-8-sig") as handle:
            return list(csv.DictReader(handle))

    with zipfile.ZipFile(source) as archive:
        if "routes.txt" not in archive.namelist():
            raise FileNotFoundError(f"No routes.txt inside {source}")
        with archive.open("routes.txt") as handle:
            text = handle.read().decode("utf-8-sig")
    return list(csv.DictReader(text.splitlines()))


def write_routes_lookup(source: Path, destination: Path) -> int:
    """Write the two-column lookup CSV. Returns the number of routes written."""
    routes = read_routes_txt(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["route_id", "route_short_name"])
        for route in routes:
            writer.writerow([route["route_id"], route.get("route_short_name", "")])
    return len(routes)


def load_routes_lookup(path: Path) -> dict[str, str]:
    """Load the lookup, returning an empty mapping (with a warning) if absent."""
    if not path.exists():
        log.warning(
            "Route lookup not found at %s — the feed will be logged unfiltered by line. "
            "Build it with `python -m transit_rag.prediction.collection.routes <gtfs.zip>`.",
            path,
        )
        return {}

    lookup: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            lookup[row["route_id"]] = row["route_short_name"]
    log.info("Loaded %d routes from %s", len(lookup), path)
    return lookup


def fetch_schedule_bundle(destination: Path) -> Path:
    """Download the "For Realtime" static bundle using the configured API key.

    Streams to disk: the bundle is tens of megabytes and there is no reason to
    hold it in memory when only ``routes.txt`` is ever read out of it.
    """
    import requests

    from transit_rag.config import TfnswConfig

    config = TfnswConfig.from_env()
    destination.parent.mkdir(parents=True, exist_ok=True)

    log.info("Downloading %s", config.schedule_url)
    response = requests.get(
        config.schedule_url,
        headers={"Authorization": f"apikey {config.api_key}"},
        timeout=300,
        stream=True,
    )
    if response.status_code in (401, 403):
        raise RuntimeError(
            f"{response.status_code} from {config.schedule_url}. The key works for the "
            "realtime feed but this is a separate API product — add "
            "'Public Transport - Timetables - For Realtime' to your application on the "
            "Open Data Hub, then retry."
        )
    response.raise_for_status()

    with destination.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1 << 20):
            handle.write(chunk)

    size_mb = destination.stat().st_size / (1 << 20)
    log.info("Saved %.1f MB to %s", size_mb, destination)
    return destination


def main() -> None:
    """CLI: build the lookup, fetching the bundle first if asked."""
    import argparse

    from transit_rag.config import PROJECT_ROOT, CollectionConfig

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source",
        type=Path,
        nargs="?",
        help="Path to a static GTFS .zip or unzipped folder (omit with --fetch)",
    )
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="Download the For Realtime bundle first, using TFNSW_API_KEY",
    )
    parser.add_argument("--out", type=Path, default=None, help="Output CSV path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not args.fetch and args.source is None:
        parser.error("give a path to a bundle, or --fetch to download one")

    source = args.source
    if args.fetch:
        source = fetch_schedule_bundle(PROJECT_ROOT / "data" / "gtfs_schedule.zip")

    assert source is not None
    destination = args.out or CollectionConfig().routes_lookup_path
    count = write_routes_lookup(source, destination)

    print(f"\nWrote {count} routes to {destination}")
    tracked = CollectionConfig().tracked_routes
    lookup = load_routes_lookup(destination)
    for line in tracked:
        ids = sorted(rid for rid, short in lookup.items() if short == line)
        print(
            f"  {line}: {len(ids)} route_ids" + (f" — {', '.join(ids[:6])}" if ids else " — NONE")
        )
    if not any(short in tracked for short in lookup.values()):
        print("\nNone of the tracked lines appear in this bundle — it is the wrong one.")
    print("\nNow run `transit-poller --probe` to confirm it pairs with the live feed.")


if __name__ == "__main__":
    main()
