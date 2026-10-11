"""``transit-freeze`` -- the evaluation dataset: one snapshot, nothing after the cut-off.

    transit-freeze --db data/delay_observations_20261019.db --cutoff 2026-10-18 \\
                   --out data/evaluation_20261018.db

``docs/08`` §3.1 fixes the evaluation data at the end of service date 2026-10-18, and
the split is drawn over it once. No scoring command takes an end date. Each splits
whatever dates its snapshot holds, so a snapshot pulled after the cut-off would carry
later dates into the test split. This writes a pruned copy for every scored run to read.

**What "after the cut-off" means, table by table.** A service date runs until 03:00 the
next morning (``SERVICE_DAY_START_HOUR``), so the cut-off *instant* is 03:00 Sydney
time on the day after the cut-off date.

- ``stop_observations`` and ``trip_statuses`` keep service dates up to the cut-off,
  late-night trains included.
- ``poll_log`` and ``alert_poll_log`` keep polls before the instant.
- ``service_alerts`` and ``alert_scopes`` lose anything first seen at or after the
  instant. An alert still in the feed then has its ``last_seen_utc`` clipped to the last
  alert poll before the instant. The 24-hour standing-notice rule must see only what
  the feed showed by the cut-off. The alert's text stays its final text, the residual
  ``docs/10`` §5 already states.

**The source is never written.** It is read through a read-only connection, its sha256
is checked before and after, and an existing output is refused. **A snapshot whose last
successful poll is before the instant is refused**, because a frozen dataset missing the
cut-off date's last hours could not be told apart from a complete one later. A failed
poll read nothing, so it counts neither for completeness nor as an alert's last sighting.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sqlite3
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path

from transit_rag.realtime.parsing import SERVICE_DAY_START_HOUR, SYDNEY

#: Tables pruned by service date, and those pruned by when they were seen.
BY_SERVICE_DATE = ("stop_observations", "trip_statuses")
POLL_TABLES = ("poll_log", "alert_poll_log")
ALERT_TABLES = ("service_alerts", "alert_scopes")


def cutoff_instant(cutoff: str) -> datetime:
    """When the cut-off date's service day ends: 03:00 Sydney time the next morning, in UTC."""
    morning = date.fromisoformat(cutoff) + timedelta(days=1)
    return datetime.combine(morning, time(SERVICE_DAY_START_HOUR), tzinfo=SYDNEY).astimezone(UTC)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class Frozen:
    """What a freeze kept and removed, for the record the report quotes."""

    out: Path
    instant: datetime
    removed: dict[str, int]
    clipped_alerts: int
    last_service_date: str
    sha256: str


def _rows_at_or_after(
    connection: sqlite3.Connection, table: str, column: str, instant: datetime
) -> list[int]:
    """Rowids whose ``column`` is at or after ``instant``, compared as instants, not strings."""
    return [
        rowid
        for rowid, value in connection.execute(f"SELECT rowid, {column} FROM {table}")
        if datetime.fromisoformat(value) >= instant
    ]


def _delete(connection: sqlite3.Connection, table: str, rowids: list[int]) -> int:
    connection.executemany(f"DELETE FROM {table} WHERE rowid = ?", [(r,) for r in rowids])
    return len(rowids)


def _last_ok_poll(connection: sqlite3.Connection, table: str) -> str | None:
    """The last poll that read the feed. A failed poll saw nothing, so it covers nothing."""
    (last,) = connection.execute(
        f"SELECT MAX(poll_time_utc) FROM {table} WHERE status = 'ok'"
    ).fetchone()
    return None if last is None else str(last)


def _write_pruned(
    db: Path, target_path: Path, cutoff: str, instant: datetime
) -> tuple[dict[str, int], int, str]:
    """Copy ``db`` into ``target_path`` and prune it. Returns removals, clips, last date."""
    source = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        target = sqlite3.connect(target_path)
        try:
            source.backup(target)
        finally:
            target.close()
    finally:
        source.close()

    target = sqlite3.connect(target_path)
    try:
        removed: dict[str, int] = {}
        for table in BY_SERVICE_DATE:
            cursor = target.execute(f"DELETE FROM {table} WHERE service_date > ?", (cutoff,))
            removed[table] = cursor.rowcount
        for table in POLL_TABLES:
            removed[table] = _delete(
                target, table, _rows_at_or_after(target, table, "poll_time_utc", instant)
            )
        for table in ALERT_TABLES:
            removed[table] = _delete(
                target, table, _rows_at_or_after(target, table, "first_seen_utc", instant)
            )
        # What the feed showed by the cut-off: the last alert poll before it that read it.
        last_alert_poll = _last_ok_poll(target, "alert_poll_log")
        clipped = 0
        for table in ALERT_TABLES:
            late = _rows_at_or_after(target, table, "last_seen_utc", instant)
            if late and last_alert_poll is None:
                raise ValueError(f"{db} has no successful alert poll before the cut-off")
            target.executemany(
                f"UPDATE {table} SET last_seen_utc = ? WHERE rowid = ?",
                [(last_alert_poll, rowid) for rowid in late],
            )
            if table == "service_alerts":
                clipped = len(late)
        target.commit()
        # One self-contained file: no write-ahead log beside it, and no freed pages.
        target.execute("PRAGMA journal_mode=DELETE")
        target.execute("VACUUM")
        (integrity,) = target.execute("PRAGMA integrity_check").fetchone()
        if integrity != "ok":
            raise ValueError(f"the frozen copy failed its integrity check: {integrity}")
        (last_date,) = target.execute("SELECT MAX(service_date) FROM stop_observations").fetchone()
    finally:
        target.close()
    return removed, clipped, str(last_date)


def freeze(db: Path, cutoff: str, out: Path) -> Frozen:
    """Write ``out``: ``db`` with nothing after ``cutoff``. ``db`` is never written."""
    if not db.exists():
        raise FileNotFoundError(f"{db} does not exist")
    if out.exists():  # the source included: it exists
        raise ValueError(f"{out} already exists; a frozen dataset is never overwritten")
    instant = cutoff_instant(cutoff)
    before = sha256(db)

    source = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        last_poll = _last_ok_poll(source, "poll_log")
    finally:
        source.close()
    if last_poll is None or datetime.fromisoformat(last_poll) < instant:
        raise ValueError(
            f"{db}'s last successful poll is {last_poll}, before the cut-off instant "
            f"{instant.isoformat()}; pull a snapshot after it"
        )

    # Built under a temporary name and renamed once complete, so a run that fails part
    # way leaves nothing to block a retry (found by Cursor Bugbot on #31).
    out.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=f".{out.name}.", suffix=".partial", dir=out.parent)
    os.close(handle)
    partial = Path(name)
    try:
        removed, clipped, last_date = _write_pruned(db, partial, cutoff, instant)
        partial.replace(out)
    finally:
        for suffix in ("", "-wal", "-shm", "-journal"):
            Path(f"{partial}{suffix}").unlink(missing_ok=True)

    if sha256(db) != before:  # pragma: no cover - a read-only connection cannot write
        raise RuntimeError(f"{db} changed while it was frozen; restore it from the VM")
    return Frozen(out, instant, removed, clipped, last_date, sha256(out))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="transit-freeze",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--db", type=Path, required=True, help="a collection snapshot (read-only)")
    parser.add_argument("--cutoff", required=True, help="the last service date kept, YYYY-MM-DD")
    parser.add_argument("--out", type=Path, required=True, help="the frozen copy to write")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        frozen = freeze(args.db, args.cutoff, args.out)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Froze {args.db} at service date {args.cutoff} into {frozen.out}")
    print(f"  cut-off instant: {frozen.instant.isoformat()} (03:00 Sydney the next morning)")
    for table, count in frozen.removed.items():
        print(f"  removed {count:>8,} row(s) from {table}")
    print(f"  clipped {frozen.clipped_alerts:>8,} alert(s) still in the feed at the cut-off")
    print(f"  last service date kept: {frozen.last_service_date}")
    print(f"  sha256: {frozen.sha256}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
