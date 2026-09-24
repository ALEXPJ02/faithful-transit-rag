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
    discover,
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


class TestDiscover:
    """Finding every era on disk, from both places a bundle lands.

    Regression cover for the gap that cost five service dates: the archiver
    wrote to ``data/bundles/`` while reconciliation globbed only ``data/``, so
    nine archived eras were never read and the trip match rate sat at 49%.
    """

    def test_finds_bundles_in_both_directories(self, tmp_path: Path) -> None:
        data = tmp_path / "data"
        archive = data / "bundles"
        archive.mkdir(parents=True)
        (data / "gtfs_schedule.zip").write_bytes(ERA_ONE)
        (archive / "gtfs_schedule_20260916_abcdef123456.zip").write_bytes(ERA_TWO)

        found = discover(data_dir=data, archive_dir=archive)

        assert [path.name for path in found] == [
            "gtfs_schedule.zip",
            "gtfs_schedule_20260916_abcdef123456.zip",
        ]

    def test_deduplicates_the_same_bundle_in_both_places(self, tmp_path: Path) -> None:
        # The archive is pulled down beside the database often enough that the
        # same file sits in both. Indexing it twice is wasted work, and the
        # duplicate would shadow the across_bundles conflict tie-break.
        data = tmp_path / "data"
        archive = data / "bundles"
        archive.mkdir(parents=True)
        name = "gtfs_schedule_20260917_bf36c40dee72.zip"
        (data / name).write_bytes(ERA_ONE)
        (archive / name).write_bytes(ERA_ONE)

        found = discover(data_dir=data, archive_dir=archive)

        assert len(found) == 1
        # The archive directory is the canonical home, so it wins.
        assert found[0].parent == archive

    def test_sorted_by_name_not_by_directory(self, tmp_path: Path) -> None:
        # Names are chronological, and across_bundles resolves a conflict in
        # favour of the earlier bundle -- so the order must not depend on
        # which directory a file happens to sit in.
        data = tmp_path / "data"
        archive = data / "bundles"
        archive.mkdir(parents=True)
        (archive / "gtfs_schedule_20260916_aaaaaaaaaaaa.zip").write_bytes(ERA_ONE)
        (data / "gtfs_schedule_20260903.zip").write_bytes(ERA_TWO)

        found = discover(data_dir=data, archive_dir=archive)

        assert [path.name for path in found] == [
            "gtfs_schedule_20260903.zip",
            "gtfs_schedule_20260916_aaaaaaaaaaaa.zip",
        ]

    def test_missing_directories_are_not_an_error(self, tmp_path: Path) -> None:
        # A fresh clone has neither directory; reconcile should report no
        # bundles rather than crash before it can say so.
        assert discover(data_dir=tmp_path / "nope", archive_dir=tmp_path / "gone") == []
