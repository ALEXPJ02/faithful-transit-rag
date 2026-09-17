"""Querying, ranking, and the sign error this module exists to prevent.

The ordering tests lean on :class:`~retrieval_fakes.HashingEmbedder`,
whose similarity tracks word overlap. That matters: a distance-vs-similarity
mix-up still returns k passages with plausible citations and sensible-looking
scores -- it just returns the *worst* ones first. Only an assertion about which
passage ranks top catches it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from retrieval_fakes import HashingEmbedder
from transit_rag.ingestion.chunks import Chunk, UnattributableChunkError
from transit_rag.retrieval.index import DEFAULT_COLLECTION, IndexFingerprint, build_index
from transit_rag.retrieval.search import (
    RetrievedPassage,
    Retriever,
    format_passages,
)

pytest.importorskip("chromadb", reason="install the 'rag' extra: pip install -e '.[rag]'")

#: Deliberately distinguishable by vocabulary, so ranking is checkable.
PASSAGES: list[tuple[str, str, int, str]] = [
    (
        "opal-terms-of-use",
        "Opal Terms of Use",
        3,
        "A daily cap applies once total fares reach the daily cap amount for that day.",
    ),
    (
        "opal-fares-business-rules",
        "Opal Fares Business Rules",
        7,
        "Concession cards must be presented to an authorised officer on request.",
    ),
    (
        "fares-and-ticketing-brochure",
        "Fares and Ticketing Brochure",
        2,
        "Ferry services depart Circular Quay wharf for Manly throughout the day.",
    ),
]


@pytest.fixture
def retriever(tmp_path: Path) -> Retriever:
    chunks = [
        Chunk(
            document_key=key,
            document_title=title,
            page=page,
            chunk_index=0,
            text=text,
        )
        for key, title, page, text in PASSAGES
    ]
    embedder = HashingEmbedder()
    vectors = embedder.embed_documents([chunk.text for chunk in chunks])
    collection = build_index(
        chunks,
        vectors,
        persist_dir=tmp_path / ".chroma",
        fingerprint=IndexFingerprint.create(
            embedding_model=embedder.model,
            embedding_dimension=embedder.dimension,
            target_chars=1000,
            overlap_chars=150,
            chunk_count=len(chunks),
        ),
        collection_name=DEFAULT_COLLECTION,
    )
    return Retriever(collection, embedder)


class TestRanking:
    def test_the_relevant_passage_comes_back_first(self, retriever: Retriever) -> None:
        """Fails loudly if distance is ever handed back as a score."""
        results = retriever.search("what is the daily cap", k=3)
        assert results[0].document_key == "opal-terms-of-use"

    def test_scores_decrease_down_the_ranking(self, retriever: Retriever) -> None:
        scores = [passage.score for passage in retriever.search("daily cap", k=3)]
        assert scores == sorted(scores, reverse=True)

    def test_ranks_are_one_based_and_contiguous(self, retriever: Retriever) -> None:
        """Hit Rate@k and MRR read the rank directly."""
        assert [p.rank for p in retriever.search("ferry wharf", k=3)] == [1, 2, 3]

    def test_a_similarity_is_reported_not_a_distance(self, retriever: Retriever) -> None:
        top = retriever.search("daily cap amount for that day", k=1)[0]
        # Near-identical text: cosine similarity near 1, so distance is near 0.
        assert top.score > 0.5

    def test_an_unrelated_query_scores_lower_than_a_matching_one(
        self, retriever: Retriever
    ) -> None:
        matching = retriever.search("daily cap", k=1)[0].score
        unrelated = retriever.search("zzzz qqqq", k=1)[0].score
        assert matching > unrelated


class TestResultShape:
    def test_every_passage_carries_what_an_answer_would_cite(self, retriever: Retriever) -> None:
        for passage in retriever.search("cap", k=3):
            assert passage.citation
            assert passage.document_title
            assert passage.page >= 1
            assert passage.text

    def test_the_citation_matches_the_document_and_page(self, retriever: Retriever) -> None:
        top = retriever.search("what is the daily cap", k=1)[0]
        assert top.citation == "Opal Terms of Use, p. 3"

    def test_a_passage_missing_its_citation_is_refused(self) -> None:
        """Between ingestion and here sits an untyped, separately writable store."""
        with pytest.raises(UnattributableChunkError, match="without a usable citation"):
            RetrievedPassage(
                chunk_id="x:p1:0",
                text="t",
                document_key="x",
                document_title="",
                page=1,
                citation="",
                score=0.5,
                rank=1,
            )

    def test_a_passage_with_an_unset_page_is_refused(self) -> None:
        with pytest.raises(UnattributableChunkError, match="without a usable citation"):
            RetrievedPassage(
                chunk_id="x:p0:0",
                text="t",
                document_key="x",
                document_title="Title",
                page=0,
                citation="Title, p. 0",
                score=0.5,
                rank=1,
            )


class TestQueryArguments:
    def test_k_larger_than_the_collection_returns_what_exists(self, retriever: Retriever) -> None:
        """The sweep runs k ∈ {3, 5, 10} over a subset smaller than 10."""
        assert len(retriever.search("cap", k=50)) == 3

    def test_k_below_one_is_refused(self, retriever: Retriever) -> None:
        with pytest.raises(ValueError, match="k must be at least 1"):
            retriever.search("cap", k=0)

    def test_an_empty_question_is_refused(self, retriever: Retriever) -> None:
        with pytest.raises(ValueError, match="empty question"):
            retriever.search("   ")

    def test_results_can_be_restricted_to_one_document(self, retriever: Retriever) -> None:
        results = retriever.search("the", k=3, document_key="fares-and-ticketing-brochure")
        assert results
        assert {p.document_key for p in results} == {"fares-and-ticketing-brochure"}

    def test_the_question_is_embedded_as_a_query(self, retriever: Retriever) -> None:
        retriever.search("daily cap", k=1)
        assert retriever._embedder.query_calls == ["daily cap"]  # type: ignore[attr-defined]

    def test_count_reports_what_was_indexed(self, retriever: Retriever) -> None:
        assert retriever.count() == 3


class TestEmptyCollection:
    def test_searching_an_empty_collection_says_to_build_it(self, tmp_path: Path) -> None:
        import chromadb

        client = chromadb.PersistentClient(path=str(tmp_path / ".chroma"))
        collection = client.create_collection(name="empty_collection")
        retriever = Retriever(collection, HashingEmbedder())
        with pytest.raises(ValueError, match="transit-index build"):
            retriever.search("anything")


class TestFormatting:
    def test_nothing_found_says_so(self) -> None:
        assert format_passages([]) == "no passages returned"

    def test_each_line_leads_with_rank_score_and_citation(self, retriever: Retriever) -> None:
        rendered = format_passages(retriever.search("daily cap", k=2))
        assert "1. [" in rendered
        assert "Opal Terms of Use, p. 3" in rendered

    def test_a_long_passage_is_truncated(self, retriever: Retriever) -> None:
        passage = RetrievedPassage(
            chunk_id="x:p1:0",
            text="word " * 200,
            document_key="x",
            document_title="T",
            page=1,
            citation="T, p. 1",
            score=0.9,
            rank=1,
        )
        assert "…" in format_passages([passage], width=40)
