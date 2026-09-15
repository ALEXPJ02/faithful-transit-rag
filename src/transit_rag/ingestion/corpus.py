"""``python -m transit_rag.ingestion.corpus`` — fetch and verify the Opal corpus.

    python -m transit_rag.ingestion.corpus --fetch     # download the three PDFs
    python -m transit_rag.ingestion.corpus --verify    # check what is on disk
    python -m transit_rag.ingestion.corpus --fetch --accept-new-versions

Three public documents make up the retrieval corpus (``docs/03-data-sources.md``
§2): the Opal Fares Business Rules, the Opal Terms of Use, and the Public
Transport Fares and Ticketing brochure. The structured Opal fares CSVs are
deliberately excluded — they are tabular, and a faithfulness judge needs prose
it can cite.

**Why the content hash is pinned in the source.** These documents are revised
without notice: the April 2026 Terms of Use already 404s while the June 2026
one serves, and three versions of the Business Rules are live simultaneously.
``docs/03`` §5 therefore requires pinning a version so eval results stay
reproducible. A silently updated corpus would change retrieval results without
changing a line of code, and every number measured against it would quietly
stop meaning what the write-up says it means. So a hash mismatch is a failure,
not a warning, and accepting a new revision is a deliberate act that edits this
file.

Not run on a schedule, so like ``collection.routes`` it stays out of
``[project.scripts]``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from transit_rag.config import PROJECT_ROOT

log = logging.getLogger("transit_rag.corpus")

DEFAULT_CORPUS_DIR = PROJECT_ROOT / "data" / "corpus"
#: Committed, unlike the PDFs themselves: it is what makes a corpus
#: reproducible without putting several megabytes of binary into the repo.
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "corpus_manifest.json"

#: transport.nsw.gov.au returns 403 to a default requests/curl User-Agent and
#: 200 to a browser one. Not an authentication step -- the documents are public
#: and unauthenticated -- just a filter that a plain client trips over.
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


@dataclass(frozen=True)
class CorpusDocument:
    """One pinned source document.

    ``title`` is what a citation says, so it must read as the document's real
    name: the faithfulness judge scores claims against retrieved passages, and
    a passage attributed to "doc3.pdf" is unusable evidence.
    """

    key: str
    title: str
    version: str
    url: str
    sha256: str
    pages: int

    @property
    def filename(self) -> str:
        return f"{self.key}.pdf"


#: Verified against the live URLs on 2026-09-15.
DOCUMENTS: tuple[CorpusDocument, ...] = (
    CorpusDocument(
        key="opal-fares-business-rules",
        title="Opal Fares Business Rules and Information",
        version="v1.6",
        url=(
            "https://opendata.transport.nsw.gov.au/data/dataset/"
            "a0f8e50d-aa28-4c5a-8aca-818d920244c9/resource/"
            "50613361-c929-4cb1-9e67-c99329281180/download/"
            "opal-fares-business-rules-and-information-v1.6-.pdf"
        ),
        sha256="91edd1cbbffa30248fd22e3a215873efb86bbe9e19ad724402f6f0d54d925d0c",
        pages=17,
    ),
    CorpusDocument(
        key="opal-terms-of-use",
        title="Opal Terms of Use",
        version="June 2026",
        url="https://transportnsw.info/document/2114/opal-terms-of-use-june-2026.pdf",
        sha256="89862e742edba493daed26be75cdb279cd05bb18ecb80aae5f23e0aacee021bc",
        pages=32,
    ),
    CorpusDocument(
        key="fares-and-ticketing-brochure",
        title="Public Transport Fares and Ticketing",
        version="2026",
        url=(
            "https://www.transport.nsw.gov.au/system/files/media/documents/2026/"
            "public-transport-fares-and-ticketing-brochure.pdf"
        ),
        sha256="28303ec7c10e06ff0d3b745ce1caaa1c1863f030a6f8a1dcd851e6de018c96c8",
        pages=7,
    ),
)


class CorpusError(RuntimeError):
    """A document could not be fetched, or is not the one that was pinned."""


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download(document: CorpusDocument, timeout: int = 60) -> bytes:
    """Fetch one document's bytes, or raise :class:`CorpusError`."""
    import requests

    try:
        response = requests.get(document.url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise CorpusError(f"Could not fetch {document.title} from {document.url}: {exc}") from exc

    if not response.content.startswith(b"%PDF-"):
        # A portal serving an HTML error page with a 200 is the failure this
        # catches: without it the "PDF" lands on disk and only fails much
        # later, inside the parser, with a far less obvious message.
        raise CorpusError(
            f"{document.title} did not return a PDF — the first bytes were "
            f"{response.content[:16]!r}. The document may have moved."
        )
    return response.content


def fetch(
    corpus_dir: Path = DEFAULT_CORPUS_DIR,
    manifest_path: Path = DEFAULT_MANIFEST,
    accept_new_versions: bool = False,
) -> list[dict[str, object]]:
    """Download every pinned document, verify it, and write the manifest."""
    corpus_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, object]] = []

    for document in DOCUMENTS:
        data = download(document)
        digest = sha256_of(data)

        if digest != document.sha256:
            message = (
                f"{document.title} has changed: expected sha256 {document.sha256}, "
                f"got {digest}. TfNSW revises these without notice, and an unannounced "
                f"change would silently alter every retrieval result measured against "
                f"this corpus."
            )
            if not accept_new_versions:
                raise CorpusError(
                    f"{message}\nRe-run with --accept-new-versions once you have "
                    f"decided to move to the new revision, then update the pin in "
                    f"{__name__} and say so in the write-up."
                )
            log.warning("%s Accepting it because --accept-new-versions was given.", message)

        destination = corpus_dir / document.filename
        destination.write_bytes(data)
        log.info("%s — %s, %s bytes", document.title, document.version, f"{len(data):,}")

        entries.append(
            {
                "key": document.key,
                "title": document.title,
                "version": document.version,
                "url": document.url,
                "sha256": digest,
                "bytes": len(data),
                "pages": document.pages,
                "filename": document.filename,
            }
        )

    manifest = {
        "fetched_at_utc": datetime.now(UTC).isoformat(),
        "documents": entries,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log.info("wrote %s", manifest_path)
    return entries


def verify(corpus_dir: Path = DEFAULT_CORPUS_DIR) -> list[str]:
    """Check the files on disk against the pins. Returns a list of problems."""
    problems: list[str] = []
    for document in DOCUMENTS:
        path = corpus_dir / document.filename
        if not path.exists():
            problems.append(f"missing: {path}")
            continue
        digest = sha256_of(path.read_bytes())
        if digest != document.sha256:
            problems.append(f"changed: {path} has sha256 {digest}, pinned {document.sha256}")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--fetch", action="store_true", help="Download the corpus")
    parser.add_argument(
        "--verify", action="store_true", help="Check the files on disk against the pins"
    )
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--accept-new-versions",
        action="store_true",
        help="Store a document whose hash no longer matches the pin, instead of failing",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s"
    )

    if args.verify:
        problems = verify(args.corpus_dir)
        if problems:
            for problem in problems:
                print(f"  {problem}")
            print(f"\n{len(problems)} problem(s). Run --fetch to download the pinned corpus.")
            sys.exit(1)
        print(f"All {len(DOCUMENTS)} documents present and matching their pinned hashes.")
        return

    if not args.fetch:
        parser.print_help()
        return

    try:
        entries = fetch(args.corpus_dir, args.manifest, args.accept_new_versions)
    except CorpusError as exc:
        log.error("%s", exc)
        sys.exit(1)

    total_pages = sum(document.pages for document in DOCUMENTS)
    print(f"\n{len(entries)} documents, {total_pages} pages, in {args.corpus_dir}")


if __name__ == "__main__":
    main()
