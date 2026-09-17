"""Embedders that behave like the real one without a key or a network.

:class:`HashingEmbedder` is not a stub returning noise: it hashes tokens into a
normalised bag-of-words vector, so cosine similarity between a question and a
passage actually tracks word overlap. That is what lets the search tests assert
on *ordering* -- that the relevant passage comes back first -- rather than only
on plumbing, which is where the sign error this module guards against would
otherwise hide.
"""

from __future__ import annotations

import hashlib
import math
import re
from types import SimpleNamespace
from typing import Any

_TOKEN = re.compile(r"[a-z0-9]+")

DIMENSION = 64


class HashingEmbedder:
    """Deterministic, offline, and similarity-preserving enough to rank with."""

    def __init__(self, dimension: int = DIMENSION, model: str = "fake-embed-1") -> None:
        self._dimension = dimension
        self._model = model
        self.document_calls: list[list[str]] = []
        self.query_calls: list[str] = []

    @property
    def model(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        return self._dimension

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self._dimension
        for token in _TOKEN.findall(text.lower()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            vector[digest[0] % self._dimension] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            # An all-punctuation passage still needs a legal vector; Chroma
            # rejects a zero vector under cosine.
            vector[0] = 1.0
            norm = 1.0
        return [value / norm for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls.append(list(texts))
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.query_calls.append(text)
        return self._vector(text)


class RecordingVoyageClient:
    """Stands in for ``voyageai.Client``, recording how it was called.

    Only ``embed`` is used by :class:`~transit_rag.retrieval.embeddings.VoyageEmbedder`,
    and only its signature matters here: the point is to assert on batching and
    on ``input_type``, both of which are invisible in the returned vectors.
    """

    def __init__(self, dimension: int = 8, widths: list[int] | None = None) -> None:
        self.dimension = dimension
        self.widths = widths
        self.calls: list[SimpleNamespace] = []

    def embed(
        self,
        texts: list[str],
        model: str | None = None,
        input_type: str | None = None,
    ) -> Any:
        self.calls.append(SimpleNamespace(texts=list(texts), model=model, input_type=input_type))
        width = self.widths[len(self.calls) - 1] if self.widths else self.dimension
        return SimpleNamespace(embeddings=[[0.1] * width for _ in texts])


class ShortCountClient(RecordingVoyageClient):
    """Returns fewer vectors than it was given texts."""

    def embed(
        self,
        texts: list[str],
        model: str | None = None,
        input_type: str | None = None,
    ) -> Any:
        result = super().embed(texts, model=model, input_type=input_type)
        result.embeddings = result.embeddings[:-1]
        return result
