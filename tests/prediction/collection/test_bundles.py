"""Archiving timetable eras before they are superseded.

No test fetches anything: the point of the module is what it does with bytes
once it has them, and CI has no business depending on TfNSW being up.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from transit_rag.prediction.collection.bundles import (
    archive_current,
    archived,
    fingerprint,
)

ERA_ONE = b"PK\x03\x04 pretend this is the September 11 era"
ERA_TWO = b"PK\x03\x04 pretend this is the September 16 era"


def _serving(payload: bytes) -> object:
    """A stand-in for fetch_schedule_bundle that writes ``payload``."""

    def _fetch(destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        return destination

    return _fetch


def _serve(monkeypatch: pytest.MonkeyPatch, payload: bytes) -> None:
    monkeypatch.setattr(
        "transit_rag.prediction.collection.routes.fetch_schedule_bundle",
        _serving(payload),
    )


class TestFingerprint:
    def test_is_content_based_and_stable(self, tmp_path: Path) -> None:
        a, b = tmp_path / "a.zip", tmp_path / "b.zip"
        a.write_bytes(ERA_ONE)
        b.write_bytes(ERA_ONE)
        assert fingerprint(a) == fingerprint(b)

    def test_differs_between_eras(self, tmp_path: Path) -> None:
        a, b = tmp_path / "a.zip", tmp_path / "b.zip"
        a.write_bytes(ERA_ONE)
        b.write_bytes(ERA_TWO)
        assert fingerprint(a) != fingerprint(b)


class TestArchiveCurrent:
    def test_a_new_era_is_kept(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _serve(monkeypatch, ERA_ONE)
        path, _ = archive_current(tmp_path)
        assert path is not None
        assert path.read_bytes() == ERA_ONE
        assert len(archived(tmp_path)) == 1

    def test_the_same_era_is_not_archived_twice(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Run daily, this is the case that happens almost every time.

        Without it the archive grows by 10 MB a day and the signal of "a new
        era appeared" is lost in the noise.
        """
        _serve(monkeypatch, ERA_ONE)
        archive_current(tmp_path)
        path, _ = archive_current(tmp_path)
        assert path is None
        assert len(archived(tmp_path)) == 1

    def test_a_genuinely_new_era_is_kept_alongside_the_old_one(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The superseded era must survive: it is the one that cannot be refetched."""
        _serve(monkeypatch, ERA_ONE)
        first, _ = archive_current(tmp_path)
        _serve(monkeypatch, ERA_TWO)
        second, _ = archive_current(tmp_path)

        assert first is not None and second is not None
        assert first != second
        assert first.exists(), "the superseded era was deleted; it cannot be re-fetched"
        assert {b.path.read_bytes() for b in archived(tmp_path)} == {ERA_ONE, ERA_TWO}

    def test_a_failed_fetch_leaves_the_archive_untouched(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A partial download must never look like a kept era."""
        _serve(monkeypatch, ERA_ONE)
        archive_current(tmp_path)

        def _explode(destination: Path) -> Path:
            raise RuntimeError("502 from the static API")

        monkeypatch.setattr(
            "transit_rag.prediction.collection.routes.fetch_schedule_bundle", _explode
        )
        with pytest.raises(RuntimeError):
            archive_current(tmp_path)
        assert len(archived(tmp_path)) == 1

    def test_archived_names_carry_the_fingerprint(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _serve(monkeypatch, ERA_ONE)
        path, digest = archive_current(tmp_path)
        assert path is not None
        assert digest[:12] in path.name
        assert path.name.startswith("gtfs_schedule_")


def test_archived_is_empty_for_a_missing_directory(tmp_path: Path) -> None:
    assert archived(tmp_path / "nothing-here") == []
