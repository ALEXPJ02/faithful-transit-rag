"""Querying the index, and handing back passages that can be cited.

**Similarity, not distance.** Chroma returns cosine *distance*, where smaller
is better. Every consumer downstream ranks and thresholds on "score", where
larger is better -- the harness's Hit Rate@k, the agent's decision about
whether retrieval found anything worth answering from, a human reading
``transit-index query``. Handing a distance to any of them is a sign error
waiting to be written, and it fails silently by returning the *worst* passages
first. The conversion happens once, here.

**A passage without a citation cannot be returned.** ``ingestion.chunks``
enforces this when a chunk is built; :class:`RetrievedPassage` enforces it
again on the way out, because between the two sits a database whose metadata
is untyped and separately writable. The judge in ``docs/08`` §3.4 scores claims
against the passage they came from, so a passage that cannot be attributed is
not weak evidence -- it is unusable, and quietly returning one would put an
uncitable passage into the answers the whole evaluation rests on.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from transit_rag.ingestion.chunks import UnattributableChunkError
from transit_rag.retrieval.embeddings import Embedder
from transit_rag.retrieval.index import DEFAULT_COLLECTION, open_collection

log = logging.getLogger("transit_rag.search")

#: Starting point for the k sweep in ``docs/08`` §3.5, which freezes k on a
#: development subset before scoring. The plan's retrieval metric is recall@5.
DEFAULT_K = 5


@dataclass(frozen=True)
class RetrievedPassage:
    """One passage the retriever returned, with what an answer would cite."""

    chunk_id: str
    text: str
    document_key: str
    document_title: str
    page: int
    citation: str
    #: Cosine similarity in [-1, 1]; larger is more relevant.
    score: float
    #: 1-based position in the returned set, so Hit Rate@k and MRR read directly.
    rank: int
    #: Where in the source the passage sits when a page cannot say: the alert
    #: ids an incident passage was built from. Empty for PDF chunks.
    locator: str = ""
    #: Everything else stored with the vector -- an incident's cause, lines and
    #: first-seen time -- for callers that need more than the citation.
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # A page for a PDF, a locator for an alert: either can be cited, and
        # neither may be faked. An alert has no page, and a page of 1 standing
        # in for "none" would be a citation to nothing.
        located = self.page >= 1 or bool(self.locator.strip())
        if not self.document_title.strip() or not located:
            raise UnattributableChunkError(
                f"passage {self.chunk_id!r} came back from the index without a usable "
                f"citation (title={self.document_title!r}, page={self.page}, "
                f"locator={self.locator!r}); the index is malformed and should be rebuilt"
            )


def _similarity(distance: float) -> float:
    """Cosine distance to cosine similarity."""
    return 1.0 - float(distance)


def _passage_from_hit(
    chunk_id: str,
    text: str,
    metadata: dict[str, Any] | None,
    distance: float,
    rank: int,
) -> RetrievedPassage:
    meta = metadata or {}
    return RetrievedPassage(
        chunk_id=chunk_id,
        text=text,
        document_key=str(meta.get("document_key", "")),
        document_title=str(meta.get("document_title", "")),
        page=int(meta.get("page", 0)),
        citation=str(meta.get("citation", "")),
        score=_similarity(distance),
        rank=rank,
        locator=str(meta.get("locator", "")),
        metadata=dict(meta),
    )


class Retriever:
    """Embeds a question and returns the passages nearest to it.

    Holds an open collection rather than a path: the harness runs hundreds of
    queries against one index, and reopening Chroma per query would dominate
    the measurement it is there to take.
    """

    def __init__(self, collection: Any, embedder: Embedder) -> None:
        self._collection = collection
        self._embedder = embedder

    @classmethod
    def open(
        cls,
        persist_dir: Path,
        embedder: Embedder,
        collection_name: str = DEFAULT_COLLECTION,
    ) -> Retriever:
        return cls(open_collection(persist_dir, collection_name), embedder)

    def count(self) -> int:
        return int(self._collection.count())

    def search(
        self,
        question: str,
        k: int = DEFAULT_K,
        *,
        document_key: str | None = None,
        seen_before: datetime | None = None,
        exclude_incident: str | None = None,
    ) -> list[RetrievedPassage]:
        """Top-``k`` passages for ``question``, best first.

        ``document_key`` restricts to one source document. The agent does not
        use it -- picking the document is the retriever's job -- but the
        ingestion sweep needs per-document recall to see which of the three
        PDFs a chunk size helps or hurts.

        ``seen_before`` and ``exclude_incident`` are the leakage guards in
        ``docs/08`` §3.5, for the alert index. Explaining an incident first
        seen at *T* may only draw on incidents first seen before *T* -- the
        future was not available when the question was asked -- and never on
        the incident itself, whose cause is the answer being scored.
        """
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        if not question.strip():
            raise ValueError("cannot search for an empty question")

        # Asking for more than the collection holds is an error in Chroma, not
        # a short result — and a caller sweeping k ∈ {3, 5, 10} over a small
        # development subset hits it routinely.
        available = self.count()
        if available == 0:
            raise ValueError("the collection is empty; build it first with `transit-index build`")
        n_results = min(k, available)

        query_vector = self._embedder.embed_query(question)
        where = _where(document_key, seen_before, exclude_incident)
        result = self._collection.query(
            query_embeddings=[query_vector],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        ids = result["ids"][0]
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]

        passages = [
            _passage_from_hit(chunk_id, text, metadata, distance, rank)
            for rank, (chunk_id, text, metadata, distance) in enumerate(
                zip(ids, documents, metadatas, distances, strict=True), start=1
            )
        ]
        log.debug(
            "%r — %d passages, best score %.3f",
            question,
            len(passages),
            passages[0].score if passages else float("nan"),
        )
        return passages


def _where(
    document_key: str | None,
    seen_before: datetime | None,
    exclude_incident: str | None,
) -> dict[str, Any] | None:
    """A Chroma metadata filter from whichever restrictions were asked for.

    Chroma rejects an ``$and`` with fewer than two clauses, so one condition
    is passed bare and none is passed as no filter at all.
    """
    clauses: list[dict[str, Any]] = []
    if document_key:
        clauses.append({"document_key": document_key})
    if seen_before is not None:
        if seen_before.tzinfo is None:
            # A naive time is ambiguous by ten or eleven hours here; guessing
            # would move the leakage boundary silently.
            raise ValueError("seen_before must be timezone-aware")
        clauses.append({"first_seen_ts": {"$lt": int(seen_before.timestamp())}})
    if exclude_incident:
        clauses.append({"incident_id": {"$ne": exclude_incident}})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def format_passages(passages: list[RetrievedPassage], *, width: int = 220) -> str:
    """Human-readable results, for the CLI and for eyeballing a sweep."""
    if not passages:
        return "no passages returned"
    lines: list[str] = []
    for passage in passages:
        snippet = " ".join(passage.text.split())
        if len(snippet) > width:
            snippet = f"{snippet[:width]}…"
        lines.append(f"{passage.rank}. [{passage.score:+.3f}] {passage.citation}")
        lines.append(f"   {snippet}")
    return "\n".join(lines)
