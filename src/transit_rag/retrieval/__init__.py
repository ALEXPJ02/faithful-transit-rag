"""Vector retrieval over the ingested Opal corpus.

Chunks from ``ingestion`` are embedded with Voyage and persisted into a local
Chroma collection; :mod:`transit_rag.retrieval.search` queries it and returns
passages that carry their citation.

Two properties this layer is responsible for, both because the evaluation in
``docs/08`` rests on them:

* **Every returned passage can be cited.** Ingestion enforces it when a chunk
  is built and ``search`` enforces it again on the way out, with an untyped
  database in between.
* **The index says what built it.** Chunk size, k and the embedding model are
  swept and then frozen, so an index that cannot be identified makes every
  number measured against it unreproducible.
"""
