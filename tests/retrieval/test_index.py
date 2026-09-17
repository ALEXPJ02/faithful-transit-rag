"""The fingerprint, and what it is for.

These tests run against a real Chroma collection in ``tmp_path`` -- embedded
Chroma needs no server and no network, so there is nothing to gain from a mock
and a great deal to lose: the properties worth checking here are that citation
metadata survives a round trip through the store, and that a rebuild really
does remove what it replaces. Neither is a property of this code alone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from retrieval_fakes import HashingEmbedder
from transit_rag.ingestion.chunks import Chunk
from transit_rag.ingestion.corpus import CorpusDocument
from transit_rag.retrieval.index import (
    DEFAULT_COLLECTION,
    IndexFingerprint,
    build_index,
    corpus_fingerprint,
    describe,
    open_collection,
    stale_reasons,
)

pytest.importorskip("chromadb", reason="install the 'rag' extra: pip install -e '.[rag]'")


def document(key: str, sha: str = "a" * 64) -> CorpusDocument:
    return CorpusDocument(
        key=key,
        title=key.replace("-", " ").title(),
        version="June 2026",
        url="https://example.invalid/x.pdf",
        sha256=sha,
        pages=4,
    )


DOCS = (document("opal-terms-of-use", "1" * 64), document("opal-fares-business-rules", "2" * 64))


def make_chunks(count: int = 3, key: str = "opal-terms-of-use") -> list[Chunk]:
    return [
        Chunk(
            document_key=key,
            document_title="Opal Terms of Use",
            page=n + 1,
            chunk_index=0,
            text=f"passage number {n} about daily caps and travel credit",
        )
        for n in range(count)
    ]


def fingerprint_for(chunks: list[Chunk], **overrides: object) -> IndexFingerprint:
    defaults: dict[str, object] = {
        "embedding_model": "fake-embed-1",
        "embedding_dimension": 64,
        "target_chars": 1000,
        "overlap_chars": 150,
        "chunk_count": len(chunks),
        "documents": DOCS,
    }
    defaults.update(overrides)
    return IndexFingerprint.create(**defaults)  # type: ignore[arg-type]


class TestCorpusFingerprint:
    def test_it_does_not_change_when_documents_are_merely_reordered(self) -> None:
        assert corpus_fingerprint(DOCS) == corpus_fingerprint(tuple(reversed(DOCS)))

    def test_it_changes_when_a_document_is_revised(self) -> None:
        revised = (document("opal-terms-of-use", "9" * 64), DOCS[1])
        assert corpus_fingerprint(DOCS) != corpus_fingerprint(revised)

    def test_it_changes_when_a_document_is_added(self) -> None:
        assert corpus_fingerprint(DOCS) != corpus_fingerprint((*DOCS, document("brochure")))


class TestFingerprintRoundTrip:
    def test_it_survives_chromas_flat_scalar_metadata(self) -> None:
        original = fingerprint_for(make_chunks())
        assert IndexFingerprint.from_metadata(original.as_metadata()) == original

    def test_the_distance_space_travels_with_it(self) -> None:
        assert fingerprint_for(make_chunks()).as_metadata()["hnsw:space"] == "cosine"

    @pytest.mark.parametrize("metadata", [None, {}, {"hnsw:space": "cosine"}])
    def test_an_index_without_one_reads_as_absent(self, metadata: dict | None) -> None:
        assert IndexFingerprint.from_metadata(metadata) is None

    def test_a_partial_fingerprint_reads_as_absent_rather_than_half_true(self) -> None:
        partial = fingerprint_for(make_chunks()).as_metadata()
        del partial["fingerprint.corpus_hash"]
        assert IndexFingerprint.from_metadata(partial) is None


class TestStaleness:
    """Each mismatch has a different remedy, so each is reported separately."""

    def base(self) -> IndexFingerprint:
        return fingerprint_for(make_chunks())

    def test_a_matching_index_is_not_stale(self) -> None:
        assert (
            stale_reasons(
                self.base(),
                embedding_model="fake-embed-1",
                target_chars=1000,
                overlap_chars=150,
                documents=DOCS,
            )
            == []
        )

    def test_an_unfingerprinted_index_cannot_be_trusted(self) -> None:
        reasons = stale_reasons(
            None, embedding_model="fake-embed-1", target_chars=1000, overlap_chars=150
        )
        assert reasons and "cannot be identified" in reasons[0]

    def test_a_changed_embedding_model_is_reported(self) -> None:
        """The silent one: query vectors would land in a different space."""
        reasons = stale_reasons(
            self.base(),
            embedding_model="voyage-4-lite",
            target_chars=1000,
            overlap_chars=150,
            documents=DOCS,
        )
        assert any("fake-embed-1" in reason and "voyage-4-lite" in reason for reason in reasons)

    def test_a_changed_chunk_size_is_reported(self) -> None:
        reasons = stale_reasons(
            self.base(),
            embedding_model="fake-embed-1",
            target_chars=600,
            overlap_chars=150,
            documents=DOCS,
        )
        assert any("chunked at 1000" in reason for reason in reasons)

    def test_a_changed_overlap_is_reported(self) -> None:
        reasons = stale_reasons(
            self.base(),
            embedding_model="fake-embed-1",
            target_chars=1000,
            overlap_chars=0,
            documents=DOCS,
        )
        assert any("overlap 150" in reason for reason in reasons)

    def test_a_revised_corpus_is_reported(self) -> None:
        """The reason the corpus is hash-pinned in the first place."""
        revised = (document("opal-terms-of-use", "9" * 64), DOCS[1])
        reasons = stale_reasons(
            self.base(),
            embedding_model="fake-embed-1",
            target_chars=1000,
            overlap_chars=150,
            documents=revised,
        )
        assert any("different corpus revision" in reason for reason in reasons)

    def test_several_mismatches_are_all_reported(self) -> None:
        reasons = stale_reasons(
            self.base(),
            embedding_model="other",
            target_chars=600,
            overlap_chars=0,
            documents=DOCS,
        )
        assert len(reasons) == 3


class TestBuilding:
    def build(self, tmp_path: Path, chunks: list[Chunk] | None = None, **overrides: object) -> Any:
        chunks = chunks if chunks is not None else make_chunks()
        vectors = HashingEmbedder().embed_documents([chunk.text for chunk in chunks])
        return build_index(
            chunks,
            vectors,
            persist_dir=tmp_path / ".chroma",
            fingerprint=fingerprint_for(chunks, **overrides),
        )

    def test_every_chunk_lands_in_the_collection(self, tmp_path: Path) -> None:
        assert self.build(tmp_path, make_chunks(5)).count() == 5

    def test_citation_metadata_survives_the_round_trip(self, tmp_path: Path) -> None:
        """The contract ingestion enforces has to hold on the way out too."""
        collection = self.build(tmp_path)
        stored = collection.get(ids=["opal-terms-of-use:p1:0"], include=["metadatas"])
        metadata = stored["metadatas"][0]
        assert metadata["citation"] == "Opal Terms of Use, p. 1"
        assert metadata["document_title"] == "Opal Terms of Use"
        assert metadata["page"] == 1

    def test_the_fingerprint_can_be_read_back_off_disk(self, tmp_path: Path) -> None:
        self.build(tmp_path)
        reopened = open_collection(tmp_path / ".chroma", DEFAULT_COLLECTION)
        stored = describe(reopened)
        assert stored is not None
        assert stored.embedding_model == "fake-embed-1"
        assert stored.chunk_count == 3

    def test_a_rebuild_removes_what_it_replaces(self, tmp_path: Path) -> None:
        """A smaller chunk size must not leave the old passages retrievable."""
        self.build(tmp_path, make_chunks(5))
        rebuilt = self.build(tmp_path, make_chunks(2), target_chars=600)
        assert rebuilt.count() == 2
        stored = describe(open_collection(tmp_path / ".chroma", DEFAULT_COLLECTION))
        assert stored is not None
        assert stored.target_chars == 600

    def test_building_from_nothing_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="zero chunks"):
            build_index(
                [],
                [],
                persist_dir=tmp_path / ".chroma",
                fingerprint=fingerprint_for(make_chunks()),
            )

    def test_mismatched_parallel_lists_are_refused(self, tmp_path: Path) -> None:
        chunks = make_chunks(3)
        with pytest.raises(ValueError, match="parallel lists"):
            build_index(
                chunks,
                [[0.1] * 64, [0.2] * 64],
                persist_dir=tmp_path / ".chroma",
                fingerprint=fingerprint_for(chunks),
            )

    def test_duplicate_chunk_ids_are_refused(self, tmp_path: Path) -> None:
        """Chroma would keep one of each, breaking gold-evidence matching."""
        chunks = [make_chunks(1)[0], make_chunks(1)[0]]
        with pytest.raises(ValueError, match="duplicate chunk ids"):
            build_index(
                chunks,
                [[0.1] * 64, [0.2] * 64],
                persist_dir=tmp_path / ".chroma",
                fingerprint=fingerprint_for(chunks),
            )

    def test_a_write_larger_than_one_batch_still_lands_whole(self, tmp_path: Path) -> None:
        """WRITE_BATCH_SIZE is 256; 300 chunks exercises the second write."""
        assert self.build(tmp_path, make_chunks(300)).count() == 300


class TestOpening:
    def test_opening_a_collection_that_was_never_built_says_how_to_build_it(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(FileNotFoundError, match="transit-index build"):
            open_collection(tmp_path / ".chroma", DEFAULT_COLLECTION)
