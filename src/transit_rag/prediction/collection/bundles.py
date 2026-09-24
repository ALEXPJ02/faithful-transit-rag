"""``python -m transit_rag.prediction.collection.bundles`` — keep every timetable era.

    python -m transit_rag.prediction.collection.bundles --archive   # fetch, keep if new
    python -m transit_rag.prediction.collection.bundles --list      # what has been kept

**Why this exists.** TfNSW's static API serves exactly one bundle: the era that
is current right now. When an era is superseded it is gone, and a realtime
trip_id planned under it can never be joined to a timetable again.

That is not hypothetical. Between 2026-09-11 and 2026-09-15 an era was
published and superseded before anything fetched it, and the result is 77,000
collected stop events -- five whole service dates -- with no
``scheduled_arrival_s`` and no ``stop_sequence``, permanently. Measured cost on
the delay model: MASE 0.934 on those dates against 0.823 on a schedule-matched
one, because ``stop_sequence`` is the second most important feature it has.

So this runs daily and keeps a copy of every distinct era it sees. Bundles are
about 10 MB, and an era that is never superseded is one that cost 10 MB to
insure. **Archived bundles are never deleted**: a superseded era cannot be
re-fetched, which makes them exactly as irreplaceable as the observations
themselves.

Deliberately a separate process from the poller, not a step inside it. The
collector holds data that cannot be re-collected, and a daily HTTP download of
a hundred-megabyte archive has no business sharing a process with it.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from transit_rag.config import PROJECT_ROOT

log = logging.getLogger("transit_rag.bundles")

DEFAULT_ARCHIVE_DIR = PROJECT_ROOT / "data" / "bundles"
#: Enough of the digest to name a file unambiguously without being unreadable.
FINGERPRINT_PREFIX = 12


@dataclass(frozen=True)
class ArchivedBundle:
    path: Path
    fingerprint: str

    @property
    def size_mb(self) -> float:
        return self.path.stat().st_size / 1_000_000


def fingerprint(path: Path) -> str:
    """sha256 of the bundle.

    The whole zip, not its members: TfNSW serves a byte-identical archive for
    repeated downloads of the same era (verified), so the cheap hash is also
    the correct one. If that ever stops being true the symptom is an archive
    that grows daily, which ``--list`` makes obvious.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def archived(archive_dir: Path) -> list[ArchivedBundle]:
    """Every bundle already kept, newest name last."""
    if not archive_dir.exists():
        return []
    return [
        ArchivedBundle(path=path, fingerprint=fingerprint(path))
        for path in sorted(archive_dir.glob("gtfs_schedule_*.zip"))
    ]


def discover(
    data_dir: Path = PROJECT_ROOT / "data",
    archive_dir: Path = DEFAULT_ARCHIVE_DIR,
) -> list[Path]:
    """Every distinct timetable era on disk, from both places a bundle lands.

    :func:`archive_current` writes to ``archive_dir``; a hand-fetched bundle
    lands in ``data_dir`` beside the database. Reconciliation globbed only the
    second for the first three weeks of collection, so nine archived eras sat
    unused and the trip match rate fell to 49% -- the eras existed and nothing
    looked at them. Missing an era is unrecoverable, so the default has to find
    them all rather than rely on remembering a flag.

    **Deduplicated by content, not by name.** Filename alone is not enough:
    ``data/gtfs_schedule_20260917.zip`` and
    ``data/bundles/gtfs_schedule_20260916_17f09b0539f6.zip`` are byte-identical
    and named differently, so a name-keyed dict indexes the same 11 MB bundle
    twice. Sizes are compared first and :func:`fingerprint` runs only within a
    size group, so the common case costs a ``stat`` per file and no reads.

    Ordered by name for determinism, not for chronology -- ``.`` sorts before
    ``_``, so an undated ``gtfs_schedule.zip`` leads regardless of when it was
    fetched. :meth:`ScheduleIndex.across_bundles` resolves a conflict in favour
    of the earlier entry, but a conflict only arises for a trip two bundles both
    describe, and then they describe it identically -- so the order decides
    nothing beyond reproducibility.
    """
    candidates: dict[str, Path] = {}
    for directory in (archive_dir, data_dir):
        if not directory.exists():
            continue
        for path in sorted(directory.glob("gtfs_schedule*.zip")):
            candidates.setdefault(path.name, path)

    by_size: dict[int, list[Path]] = {}
    for name in sorted(candidates):
        path = candidates[name]
        by_size.setdefault(path.stat().st_size, []).append(path)

    kept: list[Path] = []
    for group in by_size.values():
        if len(group) == 1:
            kept.append(group[0])
            continue
        seen: set[str] = set()
        for path in group:
            digest = fingerprint(path)
            if digest not in seen:
                seen.add(digest)
                kept.append(path)
    return sorted(kept, key=lambda path: path.name)


def archive_current(archive_dir: Path = DEFAULT_ARCHIVE_DIR) -> tuple[Path | None, str]:
    """Fetch the current bundle; keep it only if it is an era we do not have.

    Returns ``(path, fingerprint)``, with ``path`` None when the era was
    already archived.
    """
    from transit_rag.prediction.collection.routes import fetch_schedule_bundle

    archive_dir.mkdir(parents=True, exist_ok=True)
    known = {bundle.fingerprint: bundle.path for bundle in archived(archive_dir)}

    # Staged inside the archive directory under a name the glob does not match,
    # so a partial or failed download can never be mistaken for a kept era --
    # and so the final step is an atomic rename within one filesystem rather
    # than a copy across /tmp.
    staging = archive_dir / ".candidate.zip.part"
    try:
        staged = fetch_schedule_bundle(staging)
        digest = fingerprint(staged)

        if digest in known:
            log.info(
                "unchanged era %s — already archived as %s",
                digest[:FINGERPRINT_PREFIX],
                known[digest].name,
            )
            return None, digest

        stamp = datetime.now(UTC).strftime("%Y%m%d")
        destination = archive_dir / f"gtfs_schedule_{stamp}_{digest[:FINGERPRINT_PREFIX]}.zip"
        staged.rename(destination)
    finally:
        staging.unlink(missing_ok=True)

    log.info(
        "NEW timetable era archived: %s (%.1f MB). This era could not have been "
        "recovered once superseded.",
        destination.name,
        destination.stat().st_size / 1_000_000,
    )
    return destination, digest


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--archive", action="store_true", help="Fetch, and keep if it is a new era")
    parser.add_argument("--list", action="store_true", help="Show what has been archived")
    parser.add_argument("--archive-dir", type=Path, default=DEFAULT_ARCHIVE_DIR)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    if args.list:
        bundles = archived(args.archive_dir)
        if not bundles:
            print(f"No bundles archived in {args.archive_dir}.")
            return
        total = sum(bundle.size_mb for bundle in bundles)
        print(f"{len(bundles)} era(s) archived in {args.archive_dir}, {total:.0f} MB total:")
        for bundle in bundles:
            print(f"  {bundle.path.name}  {bundle.size_mb:.1f} MB")
        return

    if not args.archive:
        parser.print_help()
        return

    try:
        path, _ = archive_current(args.archive_dir)
    except Exception as exc:
        # A failed fetch is not fatal: the timer tries again tomorrow, and the
        # non-zero exit is what `systemctl status` reports.
        log.error("Could not archive the bundle: %s", exc)
        sys.exit(1)

    if path is None:
        print("No new era — the current bundle is already archived.")
    else:
        print(f"Archived a new timetable era: {path}")


if __name__ == "__main__":
    main()
