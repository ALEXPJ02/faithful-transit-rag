"""Tests for building the route_id -> line-name lookup."""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any

import pytest

from transit_rag.prediction.collection.routes import (
    fetch_schedule_bundle,
    load_routes_lookup,
    read_routes_txt,
    write_routes_lookup,
)

ROUTES_CSV = (
    "route_id,agency_id,route_short_name,route_long_name,route_type\n"
    "APS_1a,SydneyTrains,T1,North Shore & Western Line,2\n"
    "APS_4a,SydneyTrains,T4,Eastern Suburbs & Illawarra Line,2\n"
)


def test_reads_routes_from_a_zip_bundle(tmp_path: Path) -> None:
    bundle = tmp_path / "gtfs.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("routes.txt", ROUTES_CSV)

    routes = read_routes_txt(bundle)

    assert [r["route_short_name"] for r in routes] == ["T1", "T4"]


def test_reads_routes_from_an_unzipped_directory(tmp_path: Path) -> None:
    (tmp_path / "routes.txt").write_text(ROUTES_CSV, encoding="utf-8")

    routes = read_routes_txt(tmp_path)

    assert len(routes) == 2


def test_strips_the_utf8_bom_tfnsw_ships(tmp_path: Path) -> None:
    """TfNSW's routes.txt is BOM-prefixed; a naive read turns the first
    column name into '\\ufeffroute_id' and every lookup silently misses."""
    (tmp_path / "routes.txt").write_text("﻿" + ROUTES_CSV, encoding="utf-8")

    routes = read_routes_txt(tmp_path)

    assert routes[0]["route_id"] == "APS_1a"


def test_round_trips_through_the_lookup_csv(tmp_path: Path) -> None:
    (tmp_path / "routes.txt").write_text(ROUTES_CSV, encoding="utf-8")
    destination = tmp_path / "out" / "routes_lookup.csv"

    count = write_routes_lookup(tmp_path, destination)
    lookup = load_routes_lookup(destination)

    assert count == 2
    assert lookup == {"APS_1a": "T1", "APS_4a": "T4"}


def test_missing_lookup_returns_empty_mapping(tmp_path: Path) -> None:
    assert load_routes_lookup(tmp_path / "absent.csv") == {}


def test_missing_routes_txt_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_routes_txt(tmp_path)


class TestFetchScheduleBundle:
    """The bundle is a separate API product from the realtime feed, so a key
    that polls fine can still be refused here — and the message has to say so
    rather than looking like a broken URL."""

    def test_a_403_names_the_missing_api_product(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TFNSW_API_KEY", "key")

        class Refused:
            status_code = 403

            def raise_for_status(self) -> None:
                raise AssertionError("should have been handled before this")

        monkeypatch.setattr("requests.get", lambda *a, **k: Refused())

        with pytest.raises(RuntimeError, match="Timetables - For Realtime"):
            fetch_schedule_bundle(tmp_path / "gtfs.zip")

    def test_a_successful_download_streams_to_disk(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TFNSW_API_KEY", "key")

        class Ok:
            status_code = 200

            def raise_for_status(self) -> None:
                return None

            def iter_content(self, chunk_size: int) -> Any:
                yield b"PK\x03\x04"
                yield b"payload"

        monkeypatch.setattr("requests.get", lambda *a, **k: Ok())
        destination = tmp_path / "nested" / "gtfs.zip"

        result = fetch_schedule_bundle(destination)

        assert result == destination
        assert destination.read_bytes() == b"PK\x03\x04payload"

    def test_it_sends_the_apikey_header(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TFNSW_API_KEY", "secret-key")
        captured: dict[str, Any] = {}

        class Ok:
            status_code = 200

            def raise_for_status(self) -> None:
                return None

            def iter_content(self, chunk_size: int) -> Any:
                yield b""

        def fake_get(url: str, **kwargs: Any) -> Ok:
            captured.update(url=url, headers=kwargs["headers"])
            return Ok()

        monkeypatch.setattr("requests.get", fake_get)
        fetch_schedule_bundle(tmp_path / "gtfs.zip")

        assert captured["headers"]["Authorization"] == "apikey secret-key"
        assert captured["url"].endswith("/v1/gtfs/schedule/sydneytrains")
