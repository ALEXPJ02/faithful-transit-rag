"""``transit-index`` -- build, inspect and query the policy retrieval index.

    transit-index build                     # chunk, embed, persist
    transit-index build --dry-run           # chunk and report; no API calls
    transit-index build --target-chars 600  # one point in the docs/08 §4 sweep
    transit-index status                    # what is on disk, and is it stale
    transit-index query "how does a daily cap work" --k 5

``build`` is destructive: it replaces the collection rather than updating it,
so a rebuild at a new chunk size cannot leave passages from the old one behind
(``retrieval.index``). The whole corpus is two Voyage requests, so this costs
about as much as thinking about doing it incrementally would.

``--dry-run`` exists for the sweep. Chunk size is tuned by running the chunker
many times and looking at what comes out; only the point that wins needs to be
embedded, and a dry run needs neither a key nor a network.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from transit_rag.config import ConfigError, ModelConfig, chroma_persist_dir
from transit_rag.ingestion.chunks import (
    DEFAULT_OVERLAP_CHARS,
    DEFAULT_TARGET_CHARS,
    Chunk,
    chunk_corpus,
)
from transit_rag.ingestion.corpus import DEFAULT_CORPUS_DIR
from transit_rag.retrieval.embeddings import Embedder, VoyageEmbedder, embed_chunks
from transit_rag.retrieval.index import (
    DEFAULT_COLLECTION,
    IndexFingerprint,
    build_index,
    describe,
    open_collection,
    stale_reasons,
)
from transit_rag.retrieval.search import DEFAULT_K, Retriever, format_passages

log = logging.getLogger("transit_rag.index_cli")


def _summarise(chunks: list[Chunk]) -> str:
    per_document: dict[str, int] = {}
    for chunk in chunks:
        per_document[chunk.document_key] = per_document.get(chunk.document_key, 0) + 1
    lengths = sorted(len(chunk.text) for chunk in chunks)
    median = lengths[len(lengths) // 2] if lengths else 0
    breakdown = ", ".join(f"{key} {count}" for key, count in sorted(per_document.items()))
    return (
        f"{len(chunks)} chunks ({breakdown})\n"
        f"chars: min {lengths[0]}, median {median}, max {lengths[-1]}\n"
        f"total {sum(lengths):,} chars (~{sum(lengths) // 4:,} tokens to embed)"
    )


def _embedder(args: argparse.Namespace) -> Embedder:
    config = ModelConfig.from_env()
    model = args.embedding_model or config.embedding_model
    return VoyageEmbedder(api_key=config.voyage_api_key, model=model)


def _configured_model(args: argparse.Namespace) -> str:
    """The embedding model in play, without requiring an API key.

    ``status`` must work on a machine that has an index but no Voyage key --
    the point of it is to report what is on disk.
    """
    if args.embedding_model:
        return args.embedding_model
    try:
        return ModelConfig.from_env().embedding_model
    except ConfigError:
        import os

        return os.environ.get("VOYAGE_EMBEDDING_MODEL", "voyage-4-lite").strip() or "voyage-4-lite"


def command_build(args: argparse.Namespace) -> int:
    chunks = chunk_corpus(
        corpus_dir=args.corpus_dir,
        target_chars=args.target_chars,
        overlap_chars=args.overlap_chars,
    )
    print(_summarise(chunks))

    if args.dry_run:
        print("\n--dry-run: nothing embedded, nothing written.")
        return 0

    embedder = _embedder(args)
    print(f"\nembedding {len(chunks)} chunks with {embedder.model}…")
    vectors = embed_chunks(chunks, embedder)

    fingerprint = IndexFingerprint.create(
        embedding_model=embedder.model,
        embedding_dimension=len(vectors[0]),
        target_chars=args.target_chars,
        overlap_chars=args.overlap_chars,
        chunk_count=len(chunks),
    )
    collection = build_index(
        chunks,
        vectors,
        persist_dir=args.persist_dir,
        fingerprint=fingerprint,
        collection_name=args.collection,
    )
    print(
        f"built {args.collection!r} in {args.persist_dir}: "
        f"{collection.count()} passages, {fingerprint.embedding_dimension}-d"
    )
    return 0


def command_status(args: argparse.Namespace) -> int:
    try:
        collection = open_collection(args.persist_dir, args.collection)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    fingerprint = describe(collection)
    print(f"collection : {args.collection}")
    print(f"location   : {args.persist_dir}")
    print(f"passages   : {collection.count()}")

    if fingerprint is None:
        print("fingerprint: absent — this index cannot be identified; rebuild it")
        return 1

    print(f"model      : {fingerprint.embedding_model} ({fingerprint.embedding_dimension}-d)")
    print(f"chunking   : {fingerprint.target_chars} chars, {fingerprint.overlap_chars} overlap")
    print(f"corpus     : {fingerprint.corpus_hash[:16]}… ({len(fingerprint.document_keys)} docs)")
    print(f"built      : {fingerprint.built_at}")

    reasons = stale_reasons(
        fingerprint,
        embedding_model=_configured_model(args),
        target_chars=args.target_chars,
        overlap_chars=args.overlap_chars,
    )
    if reasons:
        print("\nSTALE — this index does not match the current configuration:")
        for reason in reasons:
            print(f"  - {reason}")
        print("\nRebuild with `transit-index build`.")
        return 1

    print("\nup to date with the pinned corpus and the configured chunking.")
    return 0


def command_query(args: argparse.Namespace) -> int:
    try:
        retriever = Retriever.open(args.persist_dir, _embedder(args), args.collection)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1
    passages = retriever.search(args.question, k=args.k, document_key=args.document)
    print(format_passages(passages))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="transit-index",
        description="Build and query the Opal policy retrieval index.",
    )
    parser.add_argument("--verbose", action="store_true", help="debug logging")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--persist-dir",
        type=Path,
        default=chroma_persist_dir(),
        help="where the Chroma index lives (default: $CHROMA_PERSIST_DIR or .chroma)",
    )
    common.add_argument("--collection", default=DEFAULT_COLLECTION)
    common.add_argument(
        "--embedding-model",
        default=None,
        help="override $VOYAGE_EMBEDDING_MODEL",
    )
    # On build these choose the chunking; on status they are what the stored
    # index is compared against, which is why they sit on the shared parser.
    common.add_argument("--target-chars", type=int, default=DEFAULT_TARGET_CHARS)
    common.add_argument("--overlap-chars", type=int, default=DEFAULT_OVERLAP_CHARS)

    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser(
        "build", parents=[common], help="chunk, embed and persist the corpus"
    )
    build.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    build.add_argument(
        "--dry-run",
        action="store_true",
        help="chunk and report only — no API calls, nothing written",
    )
    build.set_defaults(handler=command_build)

    status = subparsers.add_parser(
        "status", parents=[common], help="what is on disk, and whether it is stale"
    )
    status.set_defaults(handler=command_status)

    query = subparsers.add_parser("query", parents=[common], help="search the index")
    query.add_argument("question")
    query.add_argument("--k", type=int, default=DEFAULT_K)
    query.add_argument("--document", default=None, help="restrict to one document key")
    query.set_defaults(handler=command_query)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        result: int = args.handler(args)
        return result
    except (ConfigError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
