"""Text into vectors, and the asymmetry the retrieval quality depends on.

**Documents and queries are not embedded the same way.** Voyage takes an
``input_type`` and applies a different prompt to each side: a passage is
embedded as something that *contains* answers, a question as something that
*seeks* them. Leaving it unset embeds both as neutral text, which still works
-- the vectors land in the same space -- and quietly costs retrieval quality
for nothing. It is free to get right and invisible to get wrong, so the two
sides are separate methods here rather than one method with a flag a caller
can forget.

**Why a Protocol rather than just the Voyage client.** Every retrieval test in
this suite needs vectors and none of them should need an API key or a network.
The sweep in ``docs/08`` §3.5 also re-embeds the corpus once per chunk size, so
the thing being swept has to be substitutable. :class:`Embedder` is the seam;
:class:`VoyageEmbedder` is the only production implementation.

**Batch size is Voyage's, not ours.** ``voyageai.VOYAGE_EMBED_BATCH_SIZE`` is
128, and the corpus is 144 chunks at ~26.5k tokens total (measured, not
estimated), so a full build is two requests. The per-request token ceiling is
not reachable from here: ingestion caps a chunk at ``target_chars`` plus one
overlap, so 128 of them cannot approach it.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover - import cycle only matters to type checkers
    from transit_rag.ingestion.chunks import Chunk

log = logging.getLogger("transit_rag.embeddings")

#: Voyage's documented per-request limit, mirrored so this module does not
#: import voyageai just to read a constant. Asserted against the real value in
#: the test suite, so a library change that moves it cannot pass silently.
VOYAGE_BATCH_SIZE = 128

#: A transient 429 partway through a build would otherwise throw away every
#: request already paid for. Voyage's client defaults this to 0.
DEFAULT_MAX_RETRIES = 3


class EmbeddingError(RuntimeError):
    """The embedding provider returned something unusable."""


@runtime_checkable
class Embedder(Protocol):
    """Anything that can turn text into vectors for this system.

    ``model`` and ``dimension`` are part of the interface because both are
    recorded in the index fingerprint: a collection built with one embedding
    model cannot be queried with another, and the failure is silent -- the
    vectors are still the right shape, the neighbours are just wrong.
    """

    @property
    def model(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def _batched(texts: list[str], size: int) -> list[list[str]]:
    return [texts[start : start + size] for start in range(0, len(texts), size)]


def _check_rectangular(vectors: list[list[float]], expected: int | None) -> int:
    """Confirm every vector has the same width, and return it.

    Chroma accepts the first vector's width as the collection's dimension and
    rejects every later mismatch with an error naming neither the document nor
    the cause. Catching it here names both.
    """
    if not vectors:
        raise EmbeddingError("the embedding provider returned no vectors")
    widths = {len(vector) for vector in vectors}
    if len(widths) != 1:
        raise EmbeddingError(f"inconsistent embedding widths in one response: {sorted(widths)}")
    width = widths.pop()
    if width == 0:
        raise EmbeddingError("the embedding provider returned zero-width vectors")
    if expected is not None and width != expected:
        raise EmbeddingError(
            f"embedding width changed mid-build: expected {expected}, got {width}. "
            f"An index mixing widths cannot be queried."
        )
    return width


class VoyageEmbedder:
    """:class:`Embedder` backed by the Voyage API.

    The client is constructed lazily so that importing this module -- which
    ``retrieval.index`` and the CLI both do at import time -- neither requires
    the ``rag`` extra nor reads an API key. A ``--dry-run`` build chunks the
    corpus without ever touching the network, and that only works if nothing
    here demands a key on the way in.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        max_retries: int = DEFAULT_MAX_RETRIES,
        timeout: float | None = 60.0,
        batch_size: int = VOYAGE_BATCH_SIZE,
    ) -> None:
        if not api_key.strip():
            raise EmbeddingError("VOYAGE_API_KEY is empty; see docs/05-setup-checklist.md")
        if batch_size < 1:
            raise EmbeddingError(f"batch_size must be at least 1, got {batch_size}")
        self._api_key = api_key
        self._model = model
        self._max_retries = max_retries
        self._timeout = timeout
        self._batch_size = batch_size
        self._client: object | None = None
        self._dimension: int | None = None

    @property
    def model(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        """Width of this model's vectors.

        Only known once something has been embedded: Voyage publishes it per
        model, but reading it from a real response is one fewer table to keep
        in sync with the provider.
        """
        if self._dimension is None:
            raise EmbeddingError(
                "embedding dimension is unknown until the first call; "
                "embed something before asking for it"
            )
        return self._dimension

    def _voyage(self) -> object:
        if self._client is None:
            import voyageai

            self._client = voyageai.Client(
                api_key=self._api_key,
                max_retries=self._max_retries,
                timeout=self._timeout,
            )
        return self._client

    def _embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        client = self._voyage()
        vectors: list[list[float]] = []
        batches = _batched(texts, self._batch_size)
        for number, batch in enumerate(batches, start=1):
            response = client.embed(  # type: ignore[attr-defined]
                batch,
                model=self._model,
                input_type=input_type,
            )
            produced = [list(vector) for vector in response.embeddings]
            if len(produced) != len(batch):
                raise EmbeddingError(
                    f"asked for {len(batch)} embeddings, got {len(produced)}; "
                    f"the batch cannot be aligned to its texts"
                )
            self._dimension = _check_rectangular(produced, self._dimension)
            log.debug("batch %d/%d — %d texts (%s)", number, len(batches), len(batch), input_type)
            vectors.extend(produced)
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed corpus passages. ``input_type="document"``."""
        if not texts:
            return []
        return self._embed(texts, "document")

    def embed_query(self, text: str) -> list[float]:
        """Embed one question. ``input_type="query"`` -- see the module docstring."""
        if not text.strip():
            raise EmbeddingError("cannot embed an empty query")
        return self._embed([text], "query")[0]


def embed_chunks(chunks: list[Chunk], embedder: Embedder) -> list[list[float]]:
    """Embed chunk text in the order given, so vectors align to ``chunks``.

    Alignment is positional all the way into Chroma's ``add``, which takes ids,
    documents, metadatas and embeddings as four parallel lists and cannot
    detect a shuffle between them -- it would simply attach every passage to
    the wrong vector and still answer queries.
    """
    if not chunks:
        return []
    vectors = embedder.embed_documents([chunk.text for chunk in chunks])
    if len(vectors) != len(chunks):
        raise EmbeddingError(
            f"{len(chunks)} chunks produced {len(vectors)} vectors; refusing to build "
            f"an index whose passages and vectors cannot be aligned"
        )
    return vectors
