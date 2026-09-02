"""Where collected delay observations go.

Two sinks, because collection runs in two very different places:

* :class:`SqliteObservationStore` — a long-lived process on a machine that
  keeps its own state. Upserts, so the table is always "latest known value
  per stop event" and is already the shape the training table needs.
* :class:`CsvSnapshotStore` — a stateless scheduled run that has no database
  to read back, only a repository to append to. Writes one immutable file per
  poll; the upsert happens later, during reconciliation, by taking the last
  file that mentions each stop event.

Both satisfy :class:`ObservationSink`, so the poller does not care which it
was handed.
"""

from __future__ import annotations

import csv
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from types import TracebackType
from typing import Protocol

from transit_rag.realtime.parsing import StopDelayObservation, service_day

COLUMNS = (
    "service_date",
    "trip_id",
    "stop_id",
    "route_id",
    "route_short_name",
    "stop_sequence",
    "stops_ahead",
    "arrival_delay_s",
    "departure_delay_s",
    "schedule_relationship",
    "observed_at_utc",
)


class ObservationSink(Protocol):
    """What the poller needs from a place to put observations."""

    def record_observations(self, observations: Sequence[StopDelayObservation]) -> int: ...

    def record_poll(
        self, poll_time_utc: str, entities_seen: int, rows_written: int, status: str
    ) -> None: ...

    def observation_count(self) -> int: ...

    def recent_poll_status(self, limit: int = 10) -> list[tuple[str, int, int, str]]: ...

    def describe(self) -> str: ...

    def close(self) -> None: ...


_SCHEMA = """
CREATE TABLE IF NOT EXISTS stop_observations (
    service_date          TEXT    NOT NULL,
    trip_id               TEXT    NOT NULL,
    stop_id               TEXT    NOT NULL,
    -- Part of the key, and NOT NULL with a -1 sentinel: a T1 service running
    -- via the City Circle calls at the same stop_id twice in one trip, and
    -- those are two distinct stop events. SQLite treats NULLs in a composite
    -- primary key as distinct from each other, so a nullable column here
    -- would silently stop deduplicating instead.
    stop_sequence         INTEGER NOT NULL DEFAULT -1,
    route_id              TEXT,
    route_short_name      TEXT,
    stops_ahead           INTEGER,
    arrival_delay_s       INTEGER,
    departure_delay_s     INTEGER,
    schedule_relationship TEXT,
    first_seen_utc        TEXT    NOT NULL,
    last_seen_utc         TEXT    NOT NULL,
    observation_count     INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (service_date, trip_id, stop_id, stop_sequence)
);

CREATE INDEX IF NOT EXISTS idx_obs_route_date
    ON stop_observations (route_short_name, service_date);
CREATE INDEX IF NOT EXISTS idx_obs_trip
    ON stop_observations (service_date, trip_id, stop_sequence);

CREATE TABLE IF NOT EXISTS poll_log (
    poll_time_utc TEXT PRIMARY KEY,
    entities_seen INTEGER,
    rows_written  INTEGER,
    status        TEXT
);
"""

# The guard matters: polls can arrive out of order after a retry or a clock
# adjustment, and an older prediction must never overwrite a newer one — the
# whole point of the table is that the *last* value before a stop leaves the
# feed is the outcome proxy.
_UPSERT = """
INSERT INTO stop_observations (
    service_date, trip_id, stop_id, stop_sequence, route_id, route_short_name,
    stops_ahead, arrival_delay_s, departure_delay_s,
    schedule_relationship, first_seen_utc, last_seen_utc, observation_count
) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,1)
ON CONFLICT (service_date, trip_id, stop_id, stop_sequence) DO UPDATE SET
    stops_ahead           = excluded.stops_ahead,
    arrival_delay_s       = excluded.arrival_delay_s,
    departure_delay_s     = excluded.departure_delay_s,
    schedule_relationship = excluded.schedule_relationship,
    route_short_name      = excluded.route_short_name,
    last_seen_utc         = excluded.last_seen_utc,
    observation_count     = stop_observations.observation_count + 1
WHERE excluded.last_seen_utc > stop_observations.last_seen_utc
"""


class SqliteObservationStore:
    """Upserting store for a long-running collector."""

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._connection = sqlite3.connect(db_path)
        # WAL so the weekly `--status` check can read while collection writes.
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.executescript(_SCHEMA)
        self._connection.commit()

    def record_observations(self, observations: Sequence[StopDelayObservation]) -> int:
        """Upsert observations. Returns rows actually **applied**.

        Not rows submitted: the guard on ``last_seen_utc`` rejects anything
        older than what is already stored, so submitted and applied differ
        whenever a poll is retried or arrives late. Reporting submitted counts
        would inflate ``poll_log.rows_written``, which is the number the
        collection-volume checkpoint is read against.
        """
        if not observations:
            return 0
        rows = [
            (
                o.service_date,
                o.trip_id,
                o.stop_id,
                # -1 stands in for an absent sequence; see the schema comment.
                o.stop_sequence if o.stop_sequence is not None else -1,
                o.route_id,
                o.route_short_name,
                o.stops_ahead,
                o.arrival_delay_s,
                o.departure_delay_s,
                o.schedule_relationship,
                o.observed_at_utc,
                o.observed_at_utc,
            )
            for o in observations
        ]
        before = self._connection.total_changes
        self._connection.executemany(_UPSERT, rows)
        self._connection.commit()
        return self._connection.total_changes - before

    def record_poll(
        self, poll_time_utc: str, entities_seen: int, rows_written: int, status: str
    ) -> None:
        self._connection.execute(
            "INSERT OR REPLACE INTO poll_log "
            "(poll_time_utc, entities_seen, rows_written, status) VALUES (?,?,?,?)",
            (poll_time_utc, entities_seen, rows_written, status),
        )
        self._connection.commit()

    def observation_count(self) -> int:
        cursor = self._connection.execute("SELECT COUNT(*) FROM stop_observations")
        return int(cursor.fetchone()[0])

    def service_date_range(self) -> tuple[str, str] | None:
        cursor = self._connection.execute(
            "SELECT MIN(service_date), MAX(service_date) FROM stop_observations"
        )
        first, last = cursor.fetchone()
        return (str(first), str(last)) if first else None

    def recent_poll_status(self, limit: int = 10) -> list[tuple[str, int, int, str]]:
        cursor = self._connection.execute(
            "SELECT poll_time_utc, entities_seen, rows_written, status "
            "FROM poll_log ORDER BY poll_time_utc DESC LIMIT ?",
            (limit,),
        )
        return [(str(r[0]), int(r[1]), int(r[2]), str(r[3])) for r in cursor.fetchall()]

    def describe(self) -> str:
        return f"SQLite {self.db_path}"

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> SqliteObservationStore:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


class CsvSnapshotStore:
    """One immutable CSV per poll, partitioned by date.

    Immutable files rather than one growing file: a scheduled runner commits
    its output to git, and a file that is only ever added is stored once,
    whereas a file rewritten every few minutes accumulates a new copy of its
    entire contents in history each time.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        root.mkdir(parents=True, exist_ok=True)

    def _partition(self, poll_time_utc: str) -> Path:
        partition = self.root / service_day(poll_time_utc)
        partition.mkdir(parents=True, exist_ok=True)
        return partition

    def record_observations(self, observations: Sequence[StopDelayObservation]) -> int:
        if not observations:
            return 0
        poll_time = observations[0].observed_at_utc
        stamp = poll_time.replace(":", "").replace("-", "").replace("+0000", "Z")
        path = self._partition(poll_time) / f"{stamp}.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(COLUMNS))
            writer.writeheader()
            for observation in observations:
                writer.writerow(observation.as_dict())
        return len(observations)

    def record_poll(
        self, poll_time_utc: str, entities_seen: int, rows_written: int, status: str
    ) -> None:
        path = self._partition(poll_time_utc) / "_polls.csv"
        is_new = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            if is_new:
                writer.writerow(["poll_time_utc", "entities_seen", "rows_written", "status"])
            writer.writerow([poll_time_utc, entities_seen, rows_written, status])

    def observation_count(self) -> int:
        """Rows across every snapshot. Not deduplicated — reconciliation does
        that. Useful as a volume check, not as a count of stop events."""
        total = 0
        for path in sorted(self.root.glob("*/*.csv")):
            if path.name == "_polls.csv":
                continue
            with path.open(encoding="utf-8") as handle:
                total += max(sum(1 for _ in handle) - 1, 0)
        return total

    def recent_poll_status(self, limit: int = 10) -> list[tuple[str, int, int, str]]:
        rows: list[tuple[str, int, int, str]] = []
        for path in sorted(self.root.glob("*/_polls.csv"), reverse=True):
            with path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    rows.append(
                        (
                            row["poll_time_utc"],
                            int(row["entities_seen"]),
                            int(row["rows_written"]),
                            row["status"],
                        )
                    )
            if len(rows) >= limit:
                break
        rows.sort(key=lambda r: r[0], reverse=True)
        return rows[:limit]

    def describe(self) -> str:
        return f"CSV snapshots under {self.root}"

    def close(self) -> None:
        return None

    def __enter__(self) -> CsvSnapshotStore:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None
