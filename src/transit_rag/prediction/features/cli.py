"""``transit-reconcile`` — build the training table from collected observations.

    transit-reconcile                      # both sources, default paths
    transit-reconcile --report-only        # summarise without writing anything
    transit-reconcile --split              # also write train/validation/test

Reads from whichever sources exist: the always-on collector's SQLite database
and the scheduled collector's CSV snapshots. They overlap — both watch the same
feed — and the collapse step deduplicates across them, so having both is
redundancy rather than double counting.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from transit_rag.config import PROJECT_ROOT, CollectionConfig
from transit_rag.prediction.features.quality import Split, report, time_based_split
from transit_rag.prediction.features.reconcile import (
    build_training_table,
    load_from_snapshots,
    load_from_sqlite,
)
from transit_rag.prediction.features.schedule import ScheduleIndex

log = logging.getLogger("transit_rag.reconcile")


def _write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        frame.to_parquet(path, index=False)
    else:
        frame.to_csv(path, index=False)
    log.info("wrote %s rows to %s", f"{len(frame):,}", path)


def _write_split(split: Split, destination: Path) -> None:
    for name, frame in (
        ("train", split.train),
        ("validation", split.validation),
        ("test", split.test),
    ):
        _write(frame, destination.with_name(f"{destination.stem}_{name}{destination.suffix}"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    config = CollectionConfig()
    parser.add_argument("--db", type=Path, default=config.db_path, help="SQLite observations")
    parser.add_argument(
        "--snapshots", type=Path, default=config.snapshot_dir, help="CSV snapshot directory"
    )
    # Several bundles, not one. A collection window spanning a timetable
    # republication needs a bundle per era or half the trips lose their
    # schedule — see ScheduleIndex.across_bundles. The default glob picks up
    # archived era bundles (gtfs_schedule_YYYYMMDD.zip) alongside the current
    # one, so the correct behaviour does not depend on remembering a flag.
    parser.add_argument(
        "--bundle",
        type=Path,
        nargs="+",
        default=sorted((PROJECT_ROOT / "data").glob("gtfs_schedule*.zip")),
        help="Static GTFS bundle(s), for stop order and scheduled times. Pass one per "
        "timetable era the collection window spans",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "data" / "training_table.csv",
        help="Output path; .parquet writes parquet",
    )
    parser.add_argument("--split", action="store_true", help="Also write a time-based split")
    parser.add_argument("--report-only", action="store_true", help="Summarise, write nothing")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s"
    )

    sources = []
    from_db = load_from_sqlite(args.db)
    if not from_db.empty:
        log.info("%s observations from %s", f"{len(from_db):,}", args.db)
        sources.append(from_db)
    from_snapshots = load_from_snapshots(args.snapshots)
    if not from_snapshots.empty:
        log.info("%s observations from %s", f"{len(from_snapshots):,}", args.snapshots)
        sources.append(from_snapshots)

    if not sources:
        log.error(
            "No observations found. Looked in %s and %s. Pull the database from the "
            "collector first — see docs/06.",
            args.db,
            args.snapshots,
        )
        sys.exit(1)

    observations = pd.concat(sources, ignore_index=True)

    index = ScheduleIndex({}, set())
    bundles = [path for path in args.bundle if path.exists()]
    if bundles:
        trip_ids = observations["trip_id"].astype(str).unique()
        for bundle in bundles:
            log.info("indexing %s", bundle)
        index = ScheduleIndex.across_bundles(bundles, trip_ids)
        matched, total = index.coverage(trip_ids)
        log.info("schedule matched %d of %d trips (%.0f%%)", matched, total, 100 * matched / total)
        if matched < total:
            # Past a timetable rollover this is a missing era, not an aged
            # bundle, and re-fetching would swap which half matches rather than
            # fixing it. Name both possibilities so the reader checks the right
            # one.
            log.warning(
                "%d trips have no schedule. If the collection window spans a timetable "
                "republication, the bundle for one era is missing and re-fetching will "
                "not recover it — keep the superseded bundle and pass both to --bundle.",
                total - matched,
            )
    else:
        # Not fatal: delays are still real observations without it. But the
        # ordering falls back to observation time and scheduled arrival is lost,
        # so say so rather than silently producing a thinner table.
        log.warning(
            "No static bundle found at %s — stop order falls back to observation time and "
            "scheduled arrival will be empty. Fetch it with "
            "`python -m transit_rag.prediction.collection.routes --fetch`.",
            ", ".join(str(path) for path in args.bundle) or "the default location",
        )

    table = build_training_table(observations, index)
    print()
    print(report(table))

    if args.report_only:
        return

    _write(table, args.out)
    if args.split:
        split = time_based_split(table)
        print()
        print(split.describe())
        _write_split(split, args.out)


if __name__ == "__main__":
    main()
