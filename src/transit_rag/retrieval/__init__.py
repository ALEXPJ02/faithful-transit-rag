"""Embedding and vector search over the static corpus.

Voyage ``voyage-4-lite`` embeddings into a persisted Chroma collection. The
corpus never changes, so the index is built once and baked into the
deployment image rather than created at runtime (docs/02-tech-stack.md).
"""
