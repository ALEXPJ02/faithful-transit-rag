"""Tests for building the route_id -> line-name lookup."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from transit_rag.prediction.collection.routes import (
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
