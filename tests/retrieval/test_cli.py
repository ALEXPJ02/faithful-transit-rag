"""``transit-index`` argument handling, and the two paths that must not need a key.

``build --dry-run`` and ``status`` both exist to be runnable on a machine with
no Voyage credentials -- one to sweep chunk size for free, the other to report
what is already on disk. Both would be easy to break by constructing the
embedder a line too early, and neither breakage shows up anywhere else.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from retrieval_fakes import HashingEmbedder
from transit_rag.ingestion.chunks import Chunk
from transit_rag.retrieval import cli
from transit_rag.retrieval.index import DEFAULT_COLLECTION, IndexFingerprint, build_index

pytest.importorskip("chromadb", reason="install the 'rag' extra: pip install -e '.[rag]'")

CHUNKS = [
    Chunk(
        document_key="opal-terms-of-use",
        document_title="Opal Terms of Use",
        page=n + 1,
        chunk_index=0,
        text=f"passage {n} about daily caps",
    )
    for n in range(3)
]


def seed_index(persist_dir: Path, **overrides: object) -> None:
    embedder = HashingEmbedder()
    vectors = embedder.embed_documents([chunk.text for chunk in CHUNKS])
    defaults: dict[str, object] = {
        "embedding_model": embedder.model,
        "embedding_dimension": embedder.dimension,
        "target_chars": 1000,
        "overlap_chars": 150,
        "chunk_count": len(CHUNKS),
    }
    defaults.update(overrides)
    build_index(
        CHUNKS,
        vectors,
        persist_dir=persist_dir,
        fingerprint=IndexFingerprint.create(**defaults),  # type: ignore[arg-type]
        collection_name=DEFAULT_COLLECTION,
    )


class TestParsing:
    def test_a_subcommand_is_required(self) -> None:
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args([])

    def test_chunking_defaults_match_ingestion(self) -> None:
        """A build and a status check must compare like with like."""
        from transit_rag.ingestion.chunks import DEFAULT_OVERLAP_CHARS, DEFAULT_TARGET_CHARS

        args = cli.build_parser().parse_args(["status"])
        assert args.target_chars == DEFAULT_TARGET_CHARS
        assert args.overlap_chars == DEFAULT_OVERLAP_CHARS

    def test_the_sweep_parameters_are_settable_on_build(self) -> None:
        args = cli.build_parser().parse_args(
            ["build", "--target-chars", "600", "--overlap-chars", "0", "--dry-run"]
        )
        assert (args.target_chars, args.overlap_chars, args.dry_run) == (600, 0, True)

    def test_query_takes_k_and_a_document_filter(self) -> None:
        args = cli.build_parser().parse_args(
            ["query", "how do caps work", "--k", "10", "--document", "opal-terms-of-use"]
        )
        assert args.question == "how do caps work"
        assert args.k == 10
        assert args.document == "opal-terms-of-use"


class TestStatus:
    def test_it_reports_a_fresh_index_as_up_to_date(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # No key on the box: status must still work.
        monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("VOYAGE_EMBEDDING_MODEL", "fake-embed-1")
        seed_index(tmp_path / ".chroma")

        exit_code = cli.main(["status", "--persist-dir", str(tmp_path / ".chroma")])

        output = capsys.readouterr().out
        assert exit_code == 0
        assert "up to date" in output
        assert "passages   : 3" in output

    def test_it_reports_a_stale_index_and_fails(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A non-zero exit is what makes this usable as a pre-eval check."""
        monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
        monkeypatch.setenv("VOYAGE_EMBEDDING_MODEL", "fake-embed-1")
        seed_index(tmp_path / ".chroma", target_chars=600)

        exit_code = cli.main(
            ["status", "--persist-dir", str(tmp_path / ".chroma"), "--target-chars", "1000"]
        )

        output = capsys.readouterr().out
        assert exit_code == 1
        assert "STALE" in output
        assert "chunked at 600" in output

    def test_a_missing_index_is_an_error_naming_the_build_command(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        exit_code = cli.main(["status", "--persist-dir", str(tmp_path / "nothing-here")])
        assert exit_code == 1
        assert "transit-index build" in capsys.readouterr().err


class TestDryRunBuild:
    def test_it_reports_the_corpus_without_embedding_or_writing(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
        corpus_dir = tmp_path / "corpus"
        persist_dir = tmp_path / ".chroma"

        def fake_chunk_corpus(**kwargs: object) -> list[Chunk]:
            return CHUNKS

        monkeypatch.setattr(cli, "chunk_corpus", fake_chunk_corpus)

        exit_code = cli.main(
            [
                "build",
                "--dry-run",
                "--corpus-dir",
                str(corpus_dir),
                "--persist-dir",
                str(persist_dir),
            ]
        )

        output = capsys.readouterr().out
        assert exit_code == 0
        assert "3 chunks" in output
        assert "nothing embedded" in output
        assert not persist_dir.exists()

    def test_the_chunk_parameters_reach_the_chunker(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The sweep is worthless if --target-chars is not actually applied."""
        seen: dict[str, object] = {}

        def fake_chunk_corpus(**kwargs: object) -> list[Chunk]:
            seen.update(kwargs)
            return CHUNKS

        monkeypatch.setattr(cli, "chunk_corpus", fake_chunk_corpus)
        cli.main(["build", "--dry-run", "--target-chars", "600", "--overlap-chars", "25"])

        assert seen["target_chars"] == 600
        assert seen["overlap_chars"] == 25


class TestSummary:
    def test_it_names_each_document_and_its_share(self) -> None:
        summary = cli._summarise(CHUNKS)
        assert "3 chunks" in summary
        assert "opal-terms-of-use 3" in summary
        assert "tokens to embed" in summary
