"""Build and load the ``route_id -> route_short_name`` lookup.

The realtime feed identifies trips by ``route_id`` only; the human line name
(``T1``, ``T4``) lives in the static GTFS bundle's ``routes.txt``. Without
this lookup the collector cannot scope itself to the tracked lines.
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


def main() -> None:
    """CLI: build the lookup from a downloaded static GTFS bundle."""
    import argparse

    from transit_rag.config import CollectionConfig

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Path to the static GTFS .zip or unzipped folder")
    parser.add_argument("--out", type=Path, default=None, help="Output CSV path")
    args = parser.parse_args()

    destination = args.out or CollectionConfig().routes_lookup_path
    count = write_routes_lookup(args.source, destination)
    print(f"Wrote {count} routes to {destination}")
    print("Confirm T1 and T4 appear with the route_ids you expect before a long collection run.")


if __name__ == "__main__":
    main()
