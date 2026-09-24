"""Chunks into a persisted Chroma collection, and the record of what built it.

**Why an index carries a fingerprint.** ``docs/08`` §3.5 sweeps chunk size and k
on a development subset, then freezes both before the test set is scored. A
retrieval number is therefore only meaningful next to the configuration that
produced it -- and the index on disk is the one artefact that outlives the
command that built it. Without a fingerprint, "Hit Rate@5 = 0.82" is a number
nobody can reproduce, including the person who measured it a week earlier.

**Why the corpus hash is part of it.** ``ingestion.corpus`` pins each document
to a content hash precisely because TfNSW revises these PDFs without notice.
That pin protects the *files*; it does nothing for an index built from a
superseded revision and still sitting on disk. A stale index is the worst kind
of wrong here -- it answers every query confidently, with citations, from a
document that is no longer the one the write-up names. :func:`stale_reasons`
exists so that condition is detectable rather than assumed away.

**Why a rebuild is destructive.** The whole corpus is 144 chunks and ~26.5k
tokens, so a full re-embed costs about 0.013% of Voyage's free tier and takes
two requests. Incremental upsert would buy nothing measurable and cost the one
property that matters: that the collection contains exactly the chunks the
current parameters produce, with no survivors from a previous chunk size.

Chroma stores collection metadata as flat scalars, so the fingerprint is
flattened on the way in and parsed back on the way out.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from transit_rag.ingestion.chunks import Chunk
from transit_rag.ingestion.corpus import DOCUMENTS, CorpusDocument

log = logging.getLogger("transit_rag.index")

#: Chroma requires 3-512 characters from ``[a-zA-Z0-9._-]``, starting and
#: ending alphanumeric. Short names like "opal" are rejected at create time.
DEFAULT_COLLECTION = "opal_policy"

#: Cosine, because embedding models are trained with a cosine objective and the
#: vectors are not unit-normalised by contract. Chroma's default is L2, which
#: would rank by magnitude as much as by direction.
DISTANCE_SPACE = "cosine"

#: Chroma writes in one transaction per ``add``; a few hundred chunks at a time
#: keeps a rebuild to a couple of writes without holding the whole corpus in a
#: single call.
WRITE_BATCH_SIZE = 256

_FINGERPRINT_PREFIX = "fingerprint."


def corpus_fingerprint(documents: tuple[CorpusDocument, ...] = DOCUMENTS) -> str:
    """A single hash standing for "these documents at these revisions".

    Order-independent by construction, so reordering :data:`DOCUMENTS` does not
    read as a corpus change, while a revised PDF or an added document does.
    """
    material = "\n".join(sorted(f"{document.key}:{document.sha256}" for document in documents))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class IndexFingerprint:
    """Everything needed to say whether a stored index is the one you meant."""

    embedding_model: str
    embedding_dimension: int
    target_chars: int
    overlap_chars: int
    corpus_hash: str
    document_keys: tuple[str, ...]
    chunk_count: int
    built_at: str

    @classmethod
    def create(
        cls,
        *,
        embedding_model: str,
        embedding_dimension: int,
        target_chars: int,
        overlap_chars: int,
        chunk_count: int,
        documents: tuple[CorpusDocument, ...] = DOCUMENTS,
    ) -> IndexFingerprint:
        return cls(
            embedding_model=embedding_model,
            embedding_dimension=embedding_dimension,
            target_chars=target_chars,
            overlap_chars=overlap_chars,
            corpus_hash=corpus_fingerprint(documents),
            document_keys=tuple(sorted(document.key for document in documents)),
            chunk_count=chunk_count,
            built_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )

    def as_metadata(self) -> dict[str, Any]:
        """Flatten for Chroma, which stores only scalars in collection metadata."""
        return {
            "hnsw:space": DISTANCE_SPACE,
            f"{_FINGERPRINT_PREFIX}embedding_model": self.embedding_model,
            f"{_FINGERPRINT_PREFIX}embedding_dimension": self.embedding_dimension,
            f"{_FINGERPRINT_PREFIX}target_chars": self.target_chars,
            f"{_FINGERPRINT_PREFIX}overlap_chars": self.overlap_chars,
            f"{_FINGERPRINT_PREFIX}corpus_hash": self.corpus_hash,
            f"{_FINGERPRINT_PREFIX}document_keys": ",".join(self.document_keys),
            f"{_FINGERPRINT_PREFIX}chunk_count": self.chunk_count,
            f"{_FINGERPRINT_PREFIX}built_at": self.built_at,
        }

    @classmethod
    def from_metadata(cls, metadata: dict[str, Any] | None) -> IndexFingerprint | None:
        """Parse a stored fingerprint, or ``None`` if this index predates them."""
        if not metadata:
            return None
        fields = {
            key[len(_FINGERPRINT_PREFIX) :]: value
            for key, value in metadata.items()
            if key.startswith(_FINGERPRINT_PREFIX)
        }
        required = {
            "embedding_model",
            "embedding_dimension",
            "target_chars",
            "overlap_chars",
            "corpus_hash",
            "document_keys",
            "chunk_count",
            "built_at",
        }
        if not required.issubset(fields):
            return None
        keys = str(fields["document_keys"])
        return cls(
            embedding_model=str(fields["embedding_model"]),
            embedding_dimension=int(fields["embedding_dimension"]),
            target_chars=int(fields["target_chars"]),
            overlap_chars=int(fields["overlap_chars"]),
            corpus_hash=str(fields["corpus_hash"]),
            document_keys=tuple(part for part in keys.split(",") if part),
            chunk_count=int(fields["chunk_count"]),
            built_at=str(fields["built_at"]),
        )


def stale_reasons(
    stored: IndexFingerprint | None,
    *,
    embedding_model: str,
    target_chars: int,
    overlap_chars: int,
    documents: tuple[CorpusDocument, ...] = DOCUMENTS,
) -> list[str]:
    """Why the stored index does not match what is being asked for.

    Returned as prose rather than a bool because each mismatch has a different
    remedy, and "the index is stale" alone sends the reader to the wrong one.
    An empty list means the index is usable as-is.
    """
    if stored is None:
        return ["the collection carries no fingerprint, so it cannot be identified"]

    reasons: list[str] = []
    if stored.embedding_model != embedding_model:
        # The dangerous one: query vectors from a different model land in a
        # different space, so every neighbour is wrong and nothing errors.
        reasons.append(
            f"embedded with {stored.embedding_model!r}, now configured for {embedding_model!r}"
        )
    if stored.target_chars != target_chars:
        reasons.append(f"chunked at {stored.target_chars} chars, now {target_chars}")
    if stored.overlap_chars != overlap_chars:
        reasons.append(f"overlap {stored.overlap_chars} chars, now {overlap_chars}")

    current_hash = corpus_fingerprint(documents)
    if stored.corpus_hash != current_hash:
        reasons.append(
            f"built from a different corpus revision ({stored.corpus_hash[:12]}… vs "
            f"{current_hash[:12]}…); re-fetch and rebuild, or eval numbers cite documents "
            f"that are no longer pinned"
        )
    return reasons


def open_client(persist_dir: Path) -> Any:
    """A persistent Chroma client rooted at ``persist_dir``."""
    import chromadb

    persist_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(persist_dir))


def build_index(
    chunks: list[Chunk],
    vectors: list[list[float]],
    *,
    persist_dir: Path,
    fingerprint: IndexFingerprint,
    collection_name: str = DEFAULT_COLLECTION,
) -> Any:
    """Replace ``collection_name`` with exactly these chunks, and return it.

    Destructive by design -- see the module docstring. The delete-then-create
    pair means a rebuild at a new chunk size cannot leave passages from the old
    one behind to be retrieved and cited.
    """
    if not chunks:
        raise ValueError("refusing to build an index from zero chunks")
    if len(chunks) != len(vectors):
        raise ValueError(
            f"{len(chunks)} chunks and {len(vectors)} vectors; these are parallel lists "
            f"and a mismatch would attach passages to the wrong vectors"
        )

    ids = [chunk.chunk_id for chunk in chunks]
    duplicates = len(ids) - len(set(ids))
    if duplicates:
        # Chroma would silently keep one of each. A collision means chunk_id is
        # no longer unique, which breaks gold-evidence matching in the harness.
        raise ValueError(f"{duplicates} duplicate chunk ids; ids must be unique within a build")

    client = open_client(persist_dir)
    try:
        client.delete_collection(name=collection_name)
        log.info("replaced existing collection %r", collection_name)
    except Exception:
        log.debug("no existing collection %r to replace", collection_name)

    collection = client.create_collection(name=collection_name, metadata=fingerprint.as_metadata())

    for start in range(0, len(chunks), WRITE_BATCH_SIZE):
        window = slice(start, start + WRITE_BATCH_SIZE)
        batch = chunks[window]
        collection.add(
            ids=[chunk.chunk_id for chunk in batch],
            embeddings=vectors[window],
            documents=[chunk.text for chunk in batch],
            metadatas=[chunk.metadata() for chunk in batch],
        )

    log.info(
        "built %r — %d chunks, %d-d vectors from %s",
        collection_name,
        collection.count(),
        fingerprint.embedding_dimension,
        fingerprint.embedding_model,
    )
    return collection


def open_collection(
    persist_dir: Path,
    collection_name: str = DEFAULT_COLLECTION,
) -> Any:
    """Open an existing collection, with an error that says how to create it."""
    client = open_client(persist_dir)
    try:
        return client.get_collection(name=collection_name)
    except Exception as exc:
        raise FileNotFoundError(
            f"no collection {collection_name!r} in {persist_dir}. Build it first: "
            f"`transit-index build`."
        ) from exc


def describe(collection: Any) -> IndexFingerprint | None:
    """The fingerprint stored on a collection, if it has one."""
    return IndexFingerprint.from_metadata(collection.metadata)
