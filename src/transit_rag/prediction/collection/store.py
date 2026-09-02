"""SQLite storage for raw delay observations.

Append-only and deliberately dumb. ``raw_polls`` is a log of what the feed
said at each poll, not a training table; deriving features from it is an
offline step run once the collection window closes.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from pathlib import Path
from types import TracebackType

from transit_rag.realtime.parsing import StopDelayObservation

_SCHEMA = """
CREATE TABLE IF NOT EXISTS raw_polls (
    poll_time_utc         TEXT    NOT NULL,
    trip_id               TEXT    NOT NULL,
    route_id              TEXT,
    route_short_name      TEXT,
    stop_id               TEXT    NOT NULL,
    stop_sequence         INTEGER,
    arrival_delay_s       INTEGER,
    departure_delay_s     INTEGER,
    schedule_relationship TEXT,
    PRIMARY KEY (poll_time_utc, trip_id, stop_id)
);

CREATE INDEX IF NOT EXISTS idx_raw_polls_trip_stop ON raw_polls (trip_id, stop_id);
CREATE INDEX IF NOT EXISTS idx_raw_polls_route ON raw_polls (route_short_name, poll_time_utc);

CREATE TABLE IF NOT EXISTS poll_log (
    poll_time_utc TEXT PRIMARY KEY,
    entities_seen INTEGER,
    rows_written  INTEGER,
    status        TEXT
);
"""


class ObservationStore:
    """Owns the SQLite connection for a collection run."""

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._connection = sqlite3.connect(db_path)
        # WAL keeps a long-running writer from blocking ad-hoc reads, so the
        # progress checks in docs/05 can run while collection continues.
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.executescript(_SCHEMA)
        self._connection.commit()

    def record_observations(self, observations: Sequence[StopDelayObservation]) -> int:
        if not observations:
            return 0
        self._connection.executemany(
            "INSERT OR REPLACE INTO raw_polls VALUES (?,?,?,?,?,?,?,?,?)",
            [observation.as_row() for observation in observations],
        )
        self._connection.commit()
        return len(observations)

    def record_poll(
        self, poll_time_utc: str, entities_seen: int, rows_written: int, status: str
    ) -> None:
        self._connection.execute(
            "INSERT OR REPLACE INTO poll_log VALUES (?,?,?,?)",
            (poll_time_utc, entities_seen, rows_written, status),
        )
        self._connection.commit()

    def observation_count(self) -> int:
        cursor = self._connection.execute("SELECT COUNT(*) FROM raw_polls")
        return int(cursor.fetchone()[0])

    def recent_poll_status(self, limit: int = 10) -> list[tuple[str, int, int, str]]:
        cursor = self._connection.execute(
            "SELECT poll_time_utc, entities_seen, rows_written, status "
            "FROM poll_log ORDER BY poll_time_utc DESC LIMIT ?",
            (limit,),
        )
        return [(str(r[0]), int(r[1]), int(r[2]), str(r[3])) for r in cursor.fetchall()]

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> ObservationStore:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
