"""Tests for the situation the reasons stage is given (``agent/situation.py``).

The situation is the only thing standing between the reasons model and the
answer it is scored on, so these pin what it contains: the feed before the
moment, bounded as the label bounds it, and nothing after.
"""

from __future__ import annotations

import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from transit_rag.agent.situation import Situation, situation_at, station_names

# 16:00 Sydney time on a Monday, before daylight saving.
AT = datetime(2026, 9, 21, 6, 0, tzinfo=UTC)

STOPS = (
    "stop_id,stop_name,parent_station,location_type\n"
    "200060,Central Station,,1\n"
    "2000331,Central Station Platform 1,200060,0\n"
    "206710,Chatswood Station,,1\n"
    "2067141,Chatswood Station Platform 1,206710,0\n"
    "999,Olympic Park Station,,0\n"
)


def _bundle(tmp_path: Path) -> Path:
    path = tmp_path / "bundle.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("stops.txt", "﻿" + STOPS)  # TfNSW ships a byte-order mark
    return path


def _events(*rows: tuple[str, str, str, float, float]) -> pd.DataFrame:
    """``(line, trip_id, stop_id, delay_s, minutes before AT)``."""
    return pd.DataFrame(
        {
            "line": [row[0] for row in rows],
            "service_date": "2026-09-21",
            "trip_id": [row[1] for row in rows],
            "stop_id": [row[2] for row in rows],
            "stops_ahead": 0,
            "delay_s": pd.array([row[3] for row in rows], dtype="Float64"),
            "observed_at": pd.to_datetime(
                [AT - timedelta(minutes=row[4]) for row in rows], utc=True
            ),
        }
    )


def test_platforms_take_their_station_name(tmp_path: Path) -> None:
    names = station_names(_bundle(tmp_path))
    assert names["2000331"] == "Central"
    assert names["2067141"] == "Chatswood"
    assert names["999"] == "Olympic Park"


class TestSituationAt:
    def test_it_counts_late_services_and_where_they_were(self, tmp_path: Path) -> None:
        names = station_names(_bundle(tmp_path))
        events = _events(
            ("T1", "a", "2000331", 600, 20),
            ("T1", "b", "2000331", 420, 10),
            ("T1", "b", "2067141", 480, 5),
            ("T1", "c", "2067141", 30, 5),
        )
        situation = situation_at(events, ["T1"], AT, names)
        assert (situation.services, situation.late) == (3, 2)
        assert situation.max_delay_s == 600
        assert situation.delayed_stations == (("Central", 2), ("Chatswood", 1))

    def test_nothing_at_or_after_the_moment_is_seen(self, tmp_path: Path) -> None:
        """The incident's own aftermath is the future; the boundary is strict."""
        events = _events(("T1", "a", "2000331", 900, 0), ("T1", "b", "2000331", 900, -5))
        situation = situation_at(events, ["T1"], AT, station_names(_bundle(tmp_path)))
        assert situation.services == 0

    def test_only_the_lookback_and_the_named_lines_count(self, tmp_path: Path) -> None:
        events = _events(
            ("T1", "old", "2000331", 900, 45),
            ("T4", "other-line", "2000331", 900, 5),
            ("T1", "far-guess", "2000331", 900, 5),
        )
        events.loc[events["trip_id"] == "far-guess", "stops_ahead"] = 6
        situation = situation_at(events, ["T1"], AT, station_names(_bundle(tmp_path)))
        assert situation.services == 0

    def test_a_service_is_judged_by_its_latest_observation(self, tmp_path: Path) -> None:
        events = _events(("T1", "a", "2000331", 600, 20), ("T1", "a", "2000331", 60, 5))
        situation = situation_at(events, ["T1"], AT, station_names(_bundle(tmp_path)))
        assert (situation.services, situation.late) == (1, 0)
        # Where it *was* late still counts as a place delays were seen.
        assert situation.delayed_stations == (("Central", 1),)


class TestDescribe:
    def test_it_says_when_where_and_how_late_in_sydney_time(self) -> None:
        text = Situation(
            lines=("T1",),
            at=AT,
            lookback=timedelta(minutes=30),
            services=40,
            late=6,
            max_delay_s=720,
            delayed_stations=(("Chatswood", 4), ("Artarmon", 2)),
        ).describe()
        assert text.startswith("T1, 16:00 on Monday 21 September 2026 (Sydney time).")
        assert "40 services were observed and 6 were more than five minutes late" in text
        assert "12 minutes behind timetable" in text
        assert "Chatswood (4), Artarmon (2)" in text

    def test_no_service_is_said_plainly(self) -> None:
        text = Situation(("T4",), AT, timedelta(minutes=30), 0, 0, None, ()).describe()
        assert "No T4 service was observed in the 30 minutes before." in text

    def test_a_naive_time_is_refused(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            Situation(("T1",), datetime(2026, 9, 21, 16, 0), timedelta(minutes=30), 0, 0, None, ())
