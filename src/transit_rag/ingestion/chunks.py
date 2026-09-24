"""PDFs into chunks that can be cited.

The contract, from ``docs/04-implementation-plan.md`` §2 and ``docs/08`` §3.5:
**a chunk carries its document title and page number, or it does not exist.**
The faithfulness judge scores each claim against the passage it came from, so a
chunk that cannot be attributed is not weak evidence -- it is unusable, and
storing it would put unattributable passages into the retrieval results the
whole evaluation rests on. That is why the check lives in ``__post_init__``
rather than in whichever function happens to build one.

**Chunks never cross a page boundary.** A chunk spanning pages 4 and 5 has no
honest citation: half its claims are on one page and half on another, and the
judge cannot tell which. Splitting within the page keeps every citation exact.
It costs a little size uniformity -- the corpus runs 1,400-2,100 characters per
page -- and buys attribution that is right by construction.

**Chunk size is a swept parameter, not a decision.** ``docs/08`` §3.5 fixes chunk
size and k by sweeping them once on a development subset and freezing them
before the test set is scored, so nothing here hardcodes a preference.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from transit_rag.ingestion.corpus import DOCUMENTS, CorpusDocument

log = logging.getLogger("transit_rag.chunks")

#: Starting points for the sweep, not settled values.
DEFAULT_TARGET_CHARS = 1000
DEFAULT_OVERLAP_CHARS = 150
#: A page yielding less than this is furniture -- a divider, or a cover with
#: the title as artwork. The brochure has one such page at 37 characters.
#: Keeping them adds rows that can never answer anything but can still be
#: retrieved, which costs precision at every k.
MIN_PAGE_CHARS = 120

_WHITESPACE = re.compile(r"[ \t\r\f\v]+")
_BLANK_LINES = re.compile(r"\n\s*\n+")
#: Sentence end followed by a capital or digit. Deliberately conservative: it
#: is only a fallback for a paragraph too long to fit whole.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


class UnattributableChunkError(ValueError):
    """A chunk was built without the provenance a citation needs."""


@dataclass(frozen=True)
class Chunk:
    """One retrievable passage, and where it came from."""

    document_key: str
    document_title: str
    page: int
    chunk_index: int
    text: str

    def __post_init__(self) -> None:
        if not self.document_title.strip():
            raise UnattributableChunkError(
                f"chunk {self.chunk_index} of {self.document_key!r} has no document title"
            )
        if self.page < 1:
            # 1-indexed, because a citation is read by a human against a
            # printed page. A 0 here means nobody set it.
            raise UnattributableChunkError(
                f"chunk {self.chunk_index} of {self.document_key!r} has page {self.page}; "
                f"pages are 1-indexed and a citation needs a real one"
            )
        if not self.text.strip():
            raise UnattributableChunkError(
                f"chunk {self.chunk_index} of {self.document_key!r} is empty"
            )

    @property
    def citation(self) -> str:
        """What an answer cites, and what the judge checks a claim against."""
        return f"{self.document_title}, p. {self.page}"

    @property
    def chunk_id(self) -> str:
        return f"{self.document_key}:p{self.page}:{self.chunk_index}"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def metadata(self) -> dict[str, Any]:
        """Vector-store metadata. The citation fields travel with the vector."""
        return {
            "document_key": self.document_key,
            "document_title": self.document_title,
            "page": self.page,
            "chunk_index": self.chunk_index,
            "citation": self.citation,
        }


def normalise(text: str) -> str:
    """Collapse the spacing a PDF extractor leaves behind, keeping paragraphs."""
    text = _WHITESPACE.sub(" ", text)
    text = _BLANK_LINES.sub("\n\n", text)
    return "\n".join(line.strip() for line in text.splitlines()).strip()


def extract_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """``(page_number, text)`` for each page, 1-indexed."""
    from pypdf import PdfReader

    reader = PdfReader(pdf_path)
    return [
        (number, normalise(page.extract_text() or ""))
        for number, page in enumerate(reader.pages, start=1)
    ]


def _split_oversized(paragraph: str, target_chars: int) -> list[str]:
    """Break a paragraph too long to fit, preferring sentence boundaries."""
    sentences = _SENTENCE_END.split(paragraph)
    pieces: list[str] = []
    current = ""
    for sentence in sentences:
        if current and len(current) + len(sentence) + 1 > target_chars:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current)

    # A single sentence longer than the target still has to go somewhere; a
    # hard cut is ugly but it keeps the invariant that no chunk exceeds it.
    split: list[str] = []
    for piece in pieces:
        while len(piece) > target_chars:
            split.append(piece[:target_chars])
            piece = piece[target_chars:]
        if piece:
            split.append(piece)
    return split


def split_page(
    text: str,
    target_chars: int = DEFAULT_TARGET_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> list[str]:
    """Split one page's text into passages of roughly ``target_chars``.

    Paragraph boundaries first, sentence boundaries when a paragraph is too
    long. ``overlap_chars`` repeats the tail of each passage at the head of the
    next, so a claim straddling a boundary survives in one piece somewhere.
    """
    if not text.strip():
        return []

    paragraphs: list[str] = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        if len(block) > target_chars:
            paragraphs.extend(_split_oversized(block, target_chars))
        else:
            paragraphs.append(block)

    passages: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if current and len(current) + len(paragraph) + 2 > target_chars:
            passages.append(current)
            tail = current[-overlap_chars:] if overlap_chars > 0 else ""
            current = f"{tail}\n\n{paragraph}".strip() if tail else paragraph
        else:
            current = f"{current}\n\n{paragraph}".strip() if current else paragraph
    if current.strip():
        passages.append(current)
    return passages


def chunk_document(
    document: CorpusDocument,
    corpus_dir: Path,
    target_chars: int = DEFAULT_TARGET_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
    min_page_chars: int = MIN_PAGE_CHARS,
) -> list[Chunk]:
    """Chunk one document, page by page."""
    pdf_path = corpus_dir / document.filename
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"{pdf_path} is missing. Fetch the corpus first: "
            f"`python -m transit_rag.ingestion.corpus --fetch`."
        )

    chunks: list[Chunk] = []
    for page_number, page_text in extract_pages(pdf_path):
        if len(page_text) < min_page_chars:
            log.debug("%s p.%d: %d chars, skipped", document.key, page_number, len(page_text))
            continue
        for index, passage in enumerate(split_page(page_text, target_chars, overlap_chars)):
            chunks.append(
                Chunk(
                    document_key=document.key,
                    document_title=document.title,
                    page=page_number,
                    chunk_index=index,
                    text=passage,
                )
            )
    return chunks


def chunk_corpus(
    corpus_dir: Path,
    target_chars: int = DEFAULT_TARGET_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
    documents: tuple[CorpusDocument, ...] = DOCUMENTS,
) -> list[Chunk]:
    """Chunk every pinned document."""
    chunks: list[Chunk] = []
    for document in documents:
        produced = chunk_document(document, corpus_dir, target_chars, overlap_chars)
        log.info("%s — %d chunks", document.title, len(produced))
        chunks.extend(produced)
    return chunks
