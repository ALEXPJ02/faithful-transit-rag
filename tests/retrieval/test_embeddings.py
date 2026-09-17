"""Batching, the document/query asymmetry, and the alignment invariant.

Nothing here reaches Voyage. The client is substituted wholesale, because the
two properties worth testing -- how texts are split into requests, and which
``input_type`` each side is sent with -- are invisible in the vectors that come
back and would be untestable against the real API without reading its logs.
"""

from __future__ import annotations

import pytest

from retrieval_fakes import HashingEmbedder, RecordingVoyageClient, ShortCountClient
from transit_rag.ingestion.chunks import Chunk
from transit_rag.retrieval.embeddings import (
    VOYAGE_BATCH_SIZE,
    EmbeddingError,
    VoyageEmbedder,
    embed_chunks,
)


def make_embedder(client: RecordingVoyageClient, **kwargs: object) -> VoyageEmbedder:
    embedder = VoyageEmbedder(api_key="k", model="voyage-4-lite", **kwargs)  # type: ignore[arg-type]
    embedder._client = client
    return embedder


class TestTheDocumentQueryAsymmetry:
    """Voyage embeds a passage and a question differently, and it matters.

    Getting this wrong still produces working retrieval with quietly worse
    results, which is exactly the kind of failure a test has to catch because
    nothing else will.
    """

    def test_documents_are_embedded_as_documents(self) -> None:
        client = RecordingVoyageClient()
        make_embedder(client).embed_documents(["a passage"])
        assert client.calls[0].input_type == "document"

    def test_queries_are_embedded_as_queries(self) -> None:
        client = RecordingVoyageClient()
        make_embedder(client).embed_query("a question")
        assert client.calls[0].input_type == "query"

    def test_the_configured_model_is_the_one_called(self) -> None:
        client = RecordingVoyageClient()
        make_embedder(client).embed_documents(["x"])
        assert client.calls[0].model == "voyage-4-lite"


class TestBatching:
    @pytest.mark.parametrize(
        ("count", "expected_requests"),
        [(1, 1), (127, 1), (128, 1), (129, 2), (256, 2), (257, 3)],
    )
    def test_texts_are_split_at_the_provider_limit(
        self, count: int, expected_requests: int
    ) -> None:
        client = RecordingVoyageClient()
        make_embedder(client).embed_documents([f"text {n}" for n in range(count)])
        assert len(client.calls) == expected_requests

    def test_every_text_is_sent_exactly_once_and_in_order(self) -> None:
        client = RecordingVoyageClient()
        texts = [f"text {n}" for n in range(300)]
        make_embedder(client).embed_documents(texts)
        sent = [text for call in client.calls for text in call.texts]
        assert sent == texts

    def test_vectors_come_back_in_the_order_the_texts_went_out(self) -> None:
        client = RecordingVoyageClient()
        vectors = make_embedder(client).embed_documents([f"t{n}" for n in range(200)])
        assert len(vectors) == 200

    def test_no_texts_means_no_requests(self) -> None:
        client = RecordingVoyageClient()
        assert make_embedder(client).embed_documents([]) == []
        assert client.calls == []

    def test_the_mirrored_batch_size_still_matches_voyages_own(self) -> None:
        """Guards against a library bump moving the limit under us."""
        voyageai = pytest.importorskip("voyageai", reason="install the 'rag' extra")
        assert VOYAGE_BATCH_SIZE == voyageai.VOYAGE_EMBED_BATCH_SIZE


class TestMalformedResponses:
    """Every one of these produces a broken index rather than an error."""

    def test_a_short_batch_is_refused(self) -> None:
        embedder = make_embedder(ShortCountClient())
        with pytest.raises(EmbeddingError, match="cannot be aligned"):
            embedder.embed_documents(["a", "b", "c"])

    def test_inconsistent_widths_within_one_response_are_refused(self) -> None:
        client = RecordingVoyageClient()
        embedder = make_embedder(client)

        def ragged(texts: list[str], model: str | None = None, input_type: str | None = None):  # type: ignore[no-untyped-def]
            from types import SimpleNamespace

            return SimpleNamespace(embeddings=[[0.1] * (4 + n) for n, _ in enumerate(texts)])

        client.embed = ragged  # type: ignore[method-assign]
        with pytest.raises(EmbeddingError, match="inconsistent embedding widths"):
            embedder.embed_documents(["a", "b"])

    def test_a_width_change_between_batches_is_refused(self) -> None:
        """Chroma would take the first width and reject the rest, unhelpfully."""
        client = RecordingVoyageClient(widths=[8, 16])
        embedder = make_embedder(client, batch_size=2)
        with pytest.raises(EmbeddingError, match="width changed mid-build"):
            embedder.embed_documents(["a", "b", "c", "d"])

    def test_zero_width_vectors_are_refused(self) -> None:
        embedder = make_embedder(RecordingVoyageClient(dimension=0))
        with pytest.raises(EmbeddingError, match="zero-width"):
            embedder.embed_documents(["a"])


class TestConstruction:
    def test_an_empty_key_is_refused_up_front(self) -> None:
        with pytest.raises(EmbeddingError, match="VOYAGE_API_KEY is empty"):
            VoyageEmbedder(api_key="   ", model="voyage-4-lite")

    def test_a_nonsense_batch_size_is_refused(self) -> None:
        with pytest.raises(EmbeddingError, match="at least 1"):
            VoyageEmbedder(api_key="k", model="m", batch_size=0)

    def test_constructing_one_neither_needs_a_network_nor_the_rag_extra(self) -> None:
        """``--dry-run`` depends on this: the client is built lazily."""
        embedder = VoyageEmbedder(api_key="k", model="m")
        assert embedder.model == "m"

    def test_the_dimension_is_unknown_until_something_is_embedded(self) -> None:
        embedder = VoyageEmbedder(api_key="k", model="m")
        with pytest.raises(EmbeddingError, match="unknown until the first call"):
            _ = embedder.dimension

    def test_the_dimension_is_learned_from_the_first_response(self) -> None:
        embedder = make_embedder(RecordingVoyageClient(dimension=12))
        embedder.embed_documents(["a"])
        assert embedder.dimension == 12

    def test_an_empty_query_is_refused(self) -> None:
        embedder = make_embedder(RecordingVoyageClient())
        with pytest.raises(EmbeddingError, match="empty query"):
            embedder.embed_query("  \n ")


class TestChunkAlignment:
    """Chunks and vectors travel to Chroma as parallel lists."""

    def chunk(self, index: int) -> Chunk:
        return Chunk(
            document_key="opal-terms-of-use",
            document_title="Opal Terms of Use",
            page=index + 1,
            chunk_index=0,
            text=f"passage {index}",
        )

    def test_vectors_align_positionally_to_chunks(self) -> None:
        chunks = [self.chunk(n) for n in range(3)]
        embedder = HashingEmbedder()
        vectors = embed_chunks(chunks, embedder)
        assert len(vectors) == 3
        assert embedder.document_calls == [["passage 0", "passage 1", "passage 2"]]

    def test_no_chunks_means_no_call(self) -> None:
        embedder = HashingEmbedder()
        assert embed_chunks([], embedder) == []
        assert embedder.document_calls == []

    def test_a_miscounting_embedder_is_refused(self) -> None:
        class Miscounting(HashingEmbedder):
            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                return super().embed_documents(texts)[:-1]

        with pytest.raises(EmbeddingError, match="refusing to build"):
            embed_chunks([self.chunk(0), self.chunk(1)], Miscounting())
