"""Chunking, and the citation contract it exists to enforce.

Nothing here reads a real PDF: the corpus is gitignored, so CI never has it,
and a test that quietly skipped when the files were absent would be worse than
no test at all.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from transit_rag.ingestion.chunks import (
    Chunk,
    UnattributableChunkError,
    chunk_document,
    normalise,
    split_page,
)
from transit_rag.ingestion.corpus import CorpusDocument

DOCUMENT = CorpusDocument(
    key="opal-terms-of-use",
    title="Opal Terms of Use",
    version="June 2026",
    url="https://example.invalid/x.pdf",
    sha256="0" * 64,
    pages=2,
)


class TestTheCitationContract:
    """A chunk that cannot be attributed must not exist.

    Not a warning, because the faithfulness judge scores claims against the
    passage they came from: an unattributable passage is unusable evidence,
    and storing one puts it into the retrieval results the evaluation rests on.
    """

    def test_a_chunk_without_a_title_is_rejected(self) -> None:
        with pytest.raises(UnattributableChunkError, match="no document title"):
            Chunk(document_key="k", document_title="  ", page=1, chunk_index=0, text="x")

    @pytest.mark.parametrize("page", [0, -1])
    def test_a_chunk_without_a_real_page_is_rejected(self, page: int) -> None:
        """Pages are 1-indexed; a 0 means nobody set it."""
        with pytest.raises(UnattributableChunkError, match="1-indexed"):
            Chunk(document_key="k", document_title="T", page=page, chunk_index=0, text="x")

    def test_an_empty_chunk_is_rejected(self) -> None:
        with pytest.raises(UnattributableChunkError, match="is empty"):
            Chunk(document_key="k", document_title="T", page=1, chunk_index=0, text="   \n ")

    def test_citation_reads_as_a_human_would_write_it(self) -> None:
        chunk = Chunk(
            document_key="opal-terms-of-use",
            document_title="Opal Terms of Use",
            page=17,
            chunk_index=2,
            text="An Opal Card will expire.",
        )
        assert chunk.citation == "Opal Terms of Use, p. 17"
        assert chunk.chunk_id == "opal-terms-of-use:p17:2"
        assert chunk.metadata()["citation"] == chunk.citation


class TestSplitPage:
    def test_short_text_stays_one_passage(self) -> None:
        assert split_page("One short paragraph.") == ["One short paragraph."]

    def test_empty_text_yields_nothing(self) -> None:
        assert split_page("   \n\n  ") == []

    def test_passages_respect_the_target_plus_overlap(self) -> None:
        """Overlap is prepended, so target + overlap is the real bound."""
        text = "\n\n".join(f"Paragraph {i} " + "word " * 40 for i in range(12))
        passages = split_page(text, target_chars=500, overlap_chars=100)
        assert len(passages) > 1
        assert all(len(p) <= 500 + 100 for p in passages)

    def test_overlap_repeats_the_previous_tail(self) -> None:
        """A claim straddling a boundary has to survive whole somewhere."""
        text = "\n\n".join(f"Paragraph number {i}. " + "filler " * 30 for i in range(6))
        passages = split_page(text, target_chars=400, overlap_chars=80)
        assert any(passages[0][-40:] in p for p in passages[1:])

    def test_no_overlap_when_asked_for_none(self) -> None:
        text = "\n\n".join("x" * 200 for _ in range(4))
        passages = split_page(text, target_chars=300, overlap_chars=0)
        assert all(len(p) <= 300 for p in passages)

    def test_a_paragraph_longer_than_the_target_is_split(self) -> None:
        one_long_paragraph = " ".join(f"Sentence number {i} here." for i in range(60))
        passages = split_page(one_long_paragraph, target_chars=300, overlap_chars=0)
        assert len(passages) > 1
        assert all(len(p) <= 300 for p in passages)

    def test_a_single_sentence_longer_than_the_target_is_hard_cut(self) -> None:
        """Ugly, but it keeps the size bound rather than emitting a huge chunk."""
        passages = split_page("word" * 500, target_chars=200, overlap_chars=0)
        assert all(len(p) <= 200 for p in passages)


class TestChunkDocument:
    def test_chunks_never_cross_a_page_boundary(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A chunk spanning two pages has no honest citation."""
        monkeypatch.setattr(
            "transit_rag.ingestion.chunks.extract_pages",
            lambda path: [(1, "Page one content. " * 40), (2, "Page two content. " * 40)],
        )
        monkeypatch.setattr(Path, "exists", lambda self: True)
        chunks = chunk_document(DOCUMENT, Path("/nonexistent"), target_chars=300)

        for chunk in chunks:
            assert not ("Page one" in chunk.text and "Page two" in chunk.text)
        assert {c.page for c in chunks} == {1, 2}

    def test_near_empty_pages_are_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Dividers and artwork covers can be retrieved but never answer anything."""
        monkeypatch.setattr(
            "transit_rag.ingestion.chunks.extract_pages",
            lambda path: [(1, "Fares"), (2, "Real content. " * 30)],
        )
        monkeypatch.setattr(Path, "exists", lambda self: True)
        chunks = chunk_document(DOCUMENT, Path("/nonexistent"))
        assert {c.page for c in chunks} == {2}

    def test_every_chunk_carries_the_document_title(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "transit_rag.ingestion.chunks.extract_pages",
            lambda path: [(1, "Content. " * 60)],
        )
        monkeypatch.setattr(Path, "exists", lambda self: True)
        chunks = chunk_document(DOCUMENT, Path("/nonexistent"), target_chars=200)
        assert chunks
        assert all(c.document_title == "Opal Terms of Use" for c in chunks)
        assert len({c.chunk_id for c in chunks}) == len(chunks)

    def test_a_missing_pdf_says_how_to_get_it(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="corpus --fetch"):
            chunk_document(DOCUMENT, tmp_path)


def test_normalise_collapses_extractor_spacing_but_keeps_paragraphs() -> None:
    assert normalise("a   b\t\tc") == "a b c"
    assert normalise("one\n\n\n\ntwo") == "one\n\ntwo"
