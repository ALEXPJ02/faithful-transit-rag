# Retrieval

How the ingested Opal corpus becomes a searchable index, and the two properties
that layer is responsible for. The numbers in §1 were measured by running the
chunker over the pinned corpus on 2026-09-17, not estimated.

## 1. What the index actually contains

`transit-index build --dry-run` over the three pinned PDFs:

| | |
| --- | --- |
| Chunks | **144** |
| — Opal Terms of Use (32 pp.) | 93 |
| — Opal Fares Business Rules (17 pp.) | 35 |
| — Fares and Ticketing brochure (7 pp.) | 16 |
| Chunk length | min 163, median 782, max 1,152 characters |
| Corpus size | 106,238 characters, **~26,600 tokens to embed** |
| Requests per build | **2** (Voyage's batch limit is 128) |

Two consequences follow from how small that is, and both are design decisions
rather than conveniences:

**A rebuild is destructive.** ~26.6k tokens is about 0.013% of Voyage's 200M
free tier, so `build` deletes the collection and recreates it rather than
upserting into it. Incremental update would buy nothing measurable and would
cost the property that matters: that the collection holds exactly the chunks
the current parameters produce, with no survivors from a previous chunk size
still sitting there waiting to be retrieved and cited.

**The brochure is the odd one out.** 2.7 MB of PDF yielding 16 chunks — it is
image-heavy marketing collateral, and most of its pages are artwork with a
caption. `MIN_PAGE_CHARS` in `ingestion/chunks.py` drops the emptiest of them,
because a page that cannot answer anything can still be retrieved, and doing so
costs precision at every k.

## 2. Two properties this layer owns

### Every returned passage can be cited

`ingestion/chunks.py` enforces this when a chunk is built. `search.py` enforces
it *again* on the way out, and the duplication is deliberate: between those two
points sits a database whose metadata is untyped and separately writable. The
faithfulness judge ([`08-evaluation-plan.md`](./08-evaluation-plan.md) §5a)
scores each claim against the passage it came from, so a passage that cannot be
attributed is not weak evidence — it is unusable, and returning one quietly
would put an uncitable passage into the answers the whole evaluation rests on.

Citation metadata (`document_key`, `document_title`, `page`, `citation`) travels
with the vector into Chroma, so a retrieval result carries its own provenance
rather than requiring a second lookup that could silently fail to match.

### The index says what built it

[`08-evaluation-plan.md`](./08-evaluation-plan.md) §4 sweeps chunk size and k on
a development subset and freezes both before the test QA set is scored. A
retrieval number is therefore only meaningful alongside the configuration that
produced it — and the index on disk is the one artefact that outlives the
command that built it.

Every collection carries an `IndexFingerprint` in its Chroma metadata:

| Field | Why it is there |
| --- | --- |
| `embedding_model` | Query vectors from another model land in a different space. Every neighbour is wrong and **nothing errors** |
| `embedding_dimension` | Recorded from the first real response, not a table to keep in sync |
| `target_chars`, `overlap_chars` | The swept parameters. Without them "Hit Rate@5 = 0.82" names no configuration |
| `corpus_hash` | SHA-256 over each document's pinned content hash |
| `document_keys`, `chunk_count`, `built_at` | What was indexed, and when |

`transit-index status` compares the stored fingerprint against the current
configuration and **exits non-zero** when they disagree, so it works as a
pre-eval check rather than only as a thing to read.

**Why the corpus hash is in there.** `ingestion/corpus.py` pins each document to
a content hash because TfNSW revises these PDFs without notice — three versions
of the Business Rules are live simultaneously. That pin protects the *files*; it
does nothing for an index built from a superseded revision and still sitting on
disk. A stale index is the worst failure available here: it answers every query
confidently, with citations, from a document that is no longer the one the
write-up names.

## 3. Decisions worth recording

### Documents and queries are embedded differently

Voyage takes an `input_type`, and applies a different prompt to each side: a
passage is embedded as something that *contains* answers, a question as
something that *seeks* them. Leaving it unset still works — the vectors land in
the same space — and quietly costs retrieval quality for nothing.

It is free to get right and invisible to get wrong, which is why
`embed_documents` and `embed_query` are separate methods rather than one method
with a flag a caller can forget, and why the test suite asserts on the
`input_type` each side is sent with. Nothing in the returned vectors would
reveal the mistake.

### Cosine, not Chroma's default

Chroma defaults to L2. Embedding models are trained against a cosine objective
and the vectors are not unit-normalised by contract, so L2 would rank partly by
magnitude. `hnsw:space` is set to `cosine` at collection creation — it cannot be
changed afterwards, so getting it wrong means a silent rebuild later.

### Similarity is returned, never distance

Chroma returns cosine *distance*, where smaller is better. Everything downstream
ranks on "score", where larger is better — Hit Rate@k, MRR, the agent's decision
about whether retrieval found anything worth answering from, a human reading the
CLI. Handing a distance to any of them is a sign error that **fails silently by
returning the worst passages first**. The conversion happens once, in
`search.py`, and the test suite asserts on which passage ranks top rather than
only on the plumbing — an ordering assertion is the only kind that catches it.

### The embedder is a Protocol

Every retrieval test needs vectors and none of them should need an API key or a
network, and the chunk-size sweep re-embeds the corpus once per configuration.
`Embedder` is the seam. The suite substitutes a hashing embedder whose cosine
similarity tracks word overlap, so the ordering tests assert on real ranking
behaviour rather than on mocks returning canned neighbours.

## 4. Commands

```bash
transit-index build --dry-run          # chunk and report; no API calls, nothing written
transit-index build                    # chunk, embed, persist
transit-index build --target-chars 600 # one point in the docs/08 §4 sweep
transit-index status                   # what is on disk, and whether it is stale
transit-index query "how does a daily cap work" --k 5
```

`--dry-run` needs neither a key nor a network: chunk size is tuned by running
the chunker many times and looking at what comes out, and only the winning
configuration needs to be embedded. `status` likewise runs without a key —
reporting what is already on disk is the whole point of it.

The index lives at `$CHROMA_PERSIST_DIR` (default `.chroma`, gitignored). For
Phase 2 it is built into the Docker image at build time rather than created at
runtime, which is legitimate precisely because the corpus is static
([`02-tech-stack.md`](./02-tech-stack.md) §1).

## 5. Still to come

- **`VOYAGE_API_KEY` is not yet set**, so no index has been built against the
  real embedding model. Everything up to the Voyage call has been run end to end
  over the real 144 chunks — persistence, reopening from disk, the fingerprint
  round trip, staleness detection and ranked retrieval with correct citations.
  The Voyage call itself is covered by unit tests against a substituted client.
- **The chunk-size and k sweep** ([`08`](./08-evaluation-plan.md) §4). The knobs
  and the dry-run path exist; the QA set it would be swept against does not yet.
- **Retrieval as an agent tool** — `mcp_server/`, once the agent loop exists.
