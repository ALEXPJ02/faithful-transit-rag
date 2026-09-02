"""Agentic RAG for real-time Sydney public transport Q&A.

Three answer sources, one agent (see docs/01-architecture.md):

* ``retrieval``  — static Opal fare/policy documents, embedded into Chroma.
* ``realtime``   — live TfNSW GTFS-Realtime conditions, exposed as MCP tools.
* ``prediction`` — a bounded-accuracy XGBoost delay model for T1/T4.

``evaluation`` scores the combined system's faithfulness against a
static-retrieval-only baseline, which is the object of study.
"""

__version__ = "0.1.0"
