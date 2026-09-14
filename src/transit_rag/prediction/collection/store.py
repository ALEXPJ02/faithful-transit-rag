"""Where collected delay observations go.

Two sinks, because collection runs in two very different places:

* :class:`SqliteObservationStore` — a long-lived process on a machine that
  keeps its own state. Upserts, so the table holds the latest known value per
  stop event and is close to the shape the training table needs. Close, not
  identical: the key includes ``stop_sequence``, so if the feed revises a
  stop's sequence mid-trip that one event lands as two rows. Reconciliation
  has to collapse on ``(service_date, trip_id, stop_id)`` taking the latest
  ``last_seen_utc`` — which it must do for the CSV sink regardless, since
  that one does not deduplicate at all.
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

from transit_rag.realtime.parsing import (
    AlertScope,
    ServiceAlert,
    StopDelayObservation,
    service_day,
)

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


def key_stop_sequence(observation: StopDelayObservation) -> int:
    """The sequence as both sinks record it.

    ``-1`` stands in for absent so the two sinks agree: SQLite needs a
    non-NULL key column, and a CSV that wrote an empty field instead would
    make reconciliation guess which convention it was reading.
    """
    return observation.stop_sequence if observation.stop_sequence is not None else -1


class AlertSink(Protocol):
    """What the poller needs from a place to put service alerts.

    Deliberately separate from :class:`ObservationSink` rather than added to
    it. ``poll_once`` takes an ``ObservationSink``, so a ``record_alerts`` on
    that Protocol would be reachable from inside the trip-update try/except --
    and the first tidy-up that moved the call there would make an alerts
    outage look like a delay-collection outage. Keeping the Protocols apart
    makes that isolation structural instead of a convention.

    ``CsvSnapshotStore`` implements only ``ObservationSink``: it backs the
    GitHub Actions burst path, which commits its snapshots to a branch, and
    ~1,000 scope rows per burst is real repo bloat for a feed the always-on
    collector already covers completely.
    """

    def record_alerts(
        self, alerts: Sequence[ServiceAlert], scopes: Sequence[AlertScope]
    ) -> int: ...

    def record_alert_poll(
        self, poll_time_utc: str, alerts_seen: int, rows_written: int, status: str
    ) -> None: ...

    def alert_count(self) -> int: ...


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

-- Alerts are split across two tables because the cardinalities differ by an
-- order of magnitude: one live poll carries 41 alerts but 1,051 scopes, and
-- flattening them together would store a ~400 character description once per
-- selector per poll.
CREATE TABLE IF NOT EXISTS service_alerts (
    alert_id          TEXT PRIMARY KEY,
    cause             TEXT,
    effect            TEXT,
    -- Stored, but do not build a feature on it: TfNSW populates
    -- severity_level as UNKNOWN_SEVERITY on every alert it publishes.
    severity          TEXT,
    header_text       TEXT,
    description_text  TEXT,
    url               TEXT,
    first_seen_utc    TEXT    NOT NULL,
    last_seen_utc     TEXT    NOT NULL,
    observation_count INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS alert_scopes (
    alert_id            TEXT    NOT NULL,
    -- Every key column is NOT NULL with a sentinel, for the same reason
    -- stop_sequence is: SQLite treats NULLs in a composite primary key as
    -- distinct from each other, so a nullable key column here would silently
    -- stop deduplicating and the table would grow by ~1,000 rows a poll.
    route_id            TEXT    NOT NULL DEFAULT '',
    direction_id        INTEGER NOT NULL DEFAULT -1,
    stop_id             TEXT    NOT NULL DEFAULT '',
    -- POSIX seconds; 0 means unbounded. In the key because an alert with two
    -- nightly trackwork windows is genuinely two scopes.
    active_period_start INTEGER NOT NULL DEFAULT 0,
    -- Deliberately NOT in the key: a publisher extending a window should
    -- update the row rather than fork it.
    active_period_end   INTEGER NOT NULL DEFAULT 0,
    route_short_name    TEXT,
    first_seen_utc      TEXT    NOT NULL,
    last_seen_utc       TEXT    NOT NULL,
    PRIMARY KEY (alert_id, route_id, direction_id, stop_id, active_period_start)
);

CREATE INDEX IF NOT EXISTS idx_alert_scope_route
    ON alert_scopes (route_short_name, active_period_start);
CREATE INDEX IF NOT EXISTS idx_alert_scope_alert
    ON alert_scopes (alert_id);

-- A separate log, not a `feed` column on poll_log. poll_log.poll_time_utc is a
-- PRIMARY KEY written with INSERT OR REPLACE, so an alert poll sharing a
-- timestamp would overwrite the trip-update row -- destroying the only signal
-- --status and the volume checkpoint read. Widening poll_log instead would be
-- worse: CREATE TABLE IF NOT EXISTS would leave the live collector's existing
-- table in place and every trip-update write would then fail.
CREATE TABLE IF NOT EXISTS alert_poll_log (
    poll_time_utc TEXT PRIMARY KEY,
    alerts_seen   INTEGER,
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


# Both alert upserts carry the same monotonicity guard as _UPSERT: a retried
# or late poll must not roll last_seen_utc backwards, because the observed
# [first_seen, last_seen] window is the fallback for when a publisher's claimed
# active_period is absent or wrong.
_UPSERT_ALERT = """
INSERT INTO service_alerts (
    alert_id, cause, effect, severity, header_text, description_text, url,
    first_seen_utc, last_seen_utc, observation_count
) VALUES (?,?,?,?,?,?,?,?,?,1)
ON CONFLICT (alert_id) DO UPDATE SET
    cause             = excluded.cause,
    effect            = excluded.effect,
    severity          = excluded.severity,
    header_text       = excluded.header_text,
    description_text  = excluded.description_text,
    url               = excluded.url,
    last_seen_utc     = excluded.last_seen_utc,
    observation_count = service_alerts.observation_count + 1
WHERE excluded.last_seen_utc > service_alerts.last_seen_utc
"""

_UPSERT_SCOPE = """
INSERT INTO alert_scopes (
    alert_id, route_id, direction_id, stop_id, active_period_start,
    active_period_end, route_short_name, first_seen_utc, last_seen_utc
) VALUES (?,?,?,?,?,?,?,?,?)
ON CONFLICT (alert_id, route_id, direction_id, stop_id, active_period_start) DO UPDATE SET
    active_period_end = excluded.active_period_end,
    route_short_name  = excluded.route_short_name,
    last_seen_utc     = excluded.last_seen_utc
WHERE excluded.last_seen_utc > alert_scopes.last_seen_utc
"""


EXPECTED_PRIMARY_KEY = ("service_date", "trip_id", "stop_id", "stop_sequence")
ALERT_EXPECTED_PRIMARY_KEY = ("alert_id",)
SCOPE_EXPECTED_PRIMARY_KEY = (
    "alert_id",
    "route_id",
    "direction_id",
    "stop_id",
    "active_period_start",
)


class SchemaMismatchError(RuntimeError):
    """An existing database was built by an incompatible version of this code."""


class SqliteObservationStore:
    """Upserting store for a long-running collector."""

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._connection = sqlite3.connect(db_path)
        # WAL so the weekly `--status` check can read while collection writes.
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._assert_compatible_schema()
        self._connection.executescript(_SCHEMA)
        self._connection.commit()

    def _assert_compatible_schema(self) -> None:
        """Refuse a database whose primary key predates the current schema.

        ``CREATE TABLE IF NOT EXISTS`` keeps an existing table silently, and
        the upsert's ON CONFLICT target then matches no constraint — so every
        single write raises, and a local collector records nothing but errors
        until somebody reads the poll log. Fail at open, where the message can
        say what to do about it.
        """
        self._assert_primary_key("stop_observations", EXPECTED_PRIMARY_KEY)
        self._assert_primary_key("service_alerts", ALERT_EXPECTED_PRIMARY_KEY)
        self._assert_primary_key("alert_scopes", SCOPE_EXPECTED_PRIMARY_KEY)

    def _assert_primary_key(self, table: str, expected: tuple[str, ...]) -> None:
        """Check one table's primary key, or return if it does not exist yet."""
        columns = self._connection.execute(f"PRAGMA table_info({table})").fetchall()
        if not columns:
            return  # New table; the schema below will create it.

        key = tuple(
            column[1] for column in sorted((c for c in columns if c[5]), key=lambda c: c[5])
        )
        if key != expected:
            raise SchemaMismatchError(
                f"{self.db_path} has {table} primary key {key}, but this version expects "
                f"{expected}. It was written by an older schema and every "
                f"write against it would fail. Move it aside and start a fresh "
                f"collection, or migrate it before continuing."
            )

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
                key_stop_sequence(o),
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

    def record_alerts(self, alerts: Sequence[ServiceAlert], scopes: Sequence[AlertScope]) -> int:
        """Upsert alerts and their scopes. Returns rows actually **applied**.

        Both tables are written on one connection and committed once, so a
        crash cannot leave scopes referring to an alert that was never stored.
        """
        if not alerts and not scopes:
            return 0
        alert_rows = [
            (
                a.alert_id,
                a.cause,
                a.effect,
                a.severity,
                a.header_text,
                a.description_text,
                a.url,
                a.observed_at_utc,
                a.observed_at_utc,
            )
            for a in alerts
        ]
        scope_rows = [
            (
                s.alert_id,
                s.route_id,
                s.direction_id,
                s.stop_id,
                s.active_period_start,
                s.active_period_end,
                s.route_short_name,
                s.observed_at_utc,
                s.observed_at_utc,
            )
            for s in scopes
        ]
        before = self._connection.total_changes
        self._connection.executemany(_UPSERT_ALERT, alert_rows)
        self._connection.executemany(_UPSERT_SCOPE, scope_rows)
        self._connection.commit()
        return self._connection.total_changes - before

    def record_alert_poll(
        self, poll_time_utc: str, alerts_seen: int, rows_written: int, status: str
    ) -> None:
        self._connection.execute(
            "INSERT OR REPLACE INTO alert_poll_log "
            "(poll_time_utc, alerts_seen, rows_written, status) VALUES (?,?,?,?)",
            (poll_time_utc, alerts_seen, rows_written, status),
        )
        self._connection.commit()

    def alert_count(self) -> int:
        cursor = self._connection.execute("SELECT COUNT(*) FROM service_alerts")
        return int(cursor.fetchone()[0])

    def alert_scope_count(self) -> int:
        cursor = self._connection.execute("SELECT COUNT(*) FROM alert_scopes")
        return int(cursor.fetchone()[0])

    def recent_alert_poll_status(self, limit: int = 10) -> list[tuple[str, int, int, str]]:
        cursor = self._connection.execute(
            "SELECT poll_time_utc, alerts_seen, rows_written, status "
            "FROM alert_poll_log ORDER BY poll_time_utc DESC LIMIT ?",
            (limit,),
        )
        return [(str(r[0]), int(r[1]), int(r[2]), str(r[3])) for r in cursor.fetchall()]

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
                row = observation.as_dict()
                row["stop_sequence"] = key_stop_sequence(observation)
                writer.writerow(row)
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
