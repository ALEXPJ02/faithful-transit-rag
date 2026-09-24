"""The corpus pins, and the ways a fetch can quietly go wrong.

No test here touches the network: CI has no business depending on
transportnsw.info being up, and a test that silently passes because it
downloaded the real document proves nothing about the pinning logic.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from transit_rag.ingestion.corpus import (
    DOCUMENTS,
    CorpusDocument,
    CorpusError,
    fetch,
    sha256_of,
    verify,
)

PDF = b"%PDF-1.7 pretend this is the Opal Terms of Use"


class TestThePins:
    def test_every_document_is_pinned_to_a_content_hash(self) -> None:
        for document in DOCUMENTS:
            assert len(document.sha256) == 64, document.key
            assert document.url.startswith("https://"), document.key
            assert document.pages > 0, document.key

    def test_keys_and_filenames_are_unique(self) -> None:
        """A collision would have one document silently overwrite another."""
        assert len({d.key for d in DOCUMENTS}) == len(DOCUMENTS)
        assert len({d.filename for d in DOCUMENTS}) == len(DOCUMENTS)

    def test_titles_read_as_real_document_names(self) -> None:
        """Titles end up in citations, so they must be attributable.

        A chunk whose provenance reads "doc3.pdf" is unusable evidence for the
        faithfulness judge (docs/08 §3.4).
        """
        for document in DOCUMENTS:
            assert len(document.title) > 10
            assert ".pdf" not in document.title.lower()


class TestFetch:
    def test_writes_the_documents_and_a_manifest(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pinned = {d.key: PDF + d.key.encode() for d in DOCUMENTS}
        monkeypatch.setattr(
            "transit_rag.ingestion.corpus.download", lambda d, timeout=60: pinned[d.key]
        )
        monkeypatch.setattr(
            "transit_rag.ingestion.corpus.DOCUMENTS",
            tuple(
                CorpusDocument(
                    key=d.key,
                    title=d.title,
                    version=d.version,
                    url=d.url,
                    sha256=sha256_of(pinned[d.key]),
                    pages=d.pages,
                )
                for d in DOCUMENTS
            ),
        )
        manifest = tmp_path / "manifest.json"
        entries = fetch(tmp_path / "corpus", manifest)

        assert len(entries) == len(DOCUMENTS)
        assert manifest.exists()
        for document in DOCUMENTS:
            assert (tmp_path / "corpus" / document.filename).exists()

    def test_a_changed_document_fails_rather_than_being_stored(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The pin is the whole point.

        TfNSW revises these without notice -- the April 2026 Terms of Use
        already 404s while June serves. A silently updated corpus would change
        every retrieval result without changing a line of code.
        """
        monkeypatch.setattr(
            "transit_rag.ingestion.corpus.download", lambda d, timeout=60: b"%PDF- revised"
        )
        with pytest.raises(CorpusError, match="has changed"):
            fetch(tmp_path / "corpus", tmp_path / "manifest.json")

    def test_a_new_revision_can_be_accepted_deliberately(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "transit_rag.ingestion.corpus.download", lambda d, timeout=60: b"%PDF- revised"
        )
        entries = fetch(tmp_path / "corpus", tmp_path / "manifest.json", accept_new_versions=True)
        assert len(entries) == len(DOCUMENTS)


class TestVerify:
    def test_reports_missing_and_changed_documents(self, tmp_path: Path) -> None:
        (tmp_path / DOCUMENTS[0].filename).write_bytes(b"%PDF- not the pinned one")
        problems = verify(tmp_path)
        # Match on the message prefix, not a substring: pytest names tmp_path
        # after the test function, so every path here contains "missing".
        assert sum(p.startswith("changed:") for p in problems) == 1
        assert sum(p.startswith("missing:") for p in problems) == len(DOCUMENTS) - 1

    def test_is_silent_when_everything_matches(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "transit_rag.ingestion.corpus.DOCUMENTS",
            (
                CorpusDocument(
                    key="k",
                    title="A Real Document Title",
                    version="v1",
                    url="https://example.invalid/x.pdf",
                    sha256=sha256_of(PDF),
                    pages=1,
                ),
            ),
        )
        (tmp_path / "k.pdf").write_bytes(PDF)
        assert verify(tmp_path) == []


def test_an_html_error_page_served_with_200_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Portals answer a moved document with a 200 and an HTML page.

    Without this check the "PDF" lands on disk and only fails much later,
    inside the parser, with a far less obvious message.
    """
    import transit_rag.ingestion.corpus as corpus

    class FakeResponse:
        content = b"<!DOCTYPE html><title>Not found</title>"

        def raise_for_status(self) -> None: ...

    import requests

    monkeypatch.setattr(requests, "get", lambda *a, **k: FakeResponse())
    with pytest.raises(CorpusError, match="did not return a PDF"):
        corpus.download(DOCUMENTS[0])
