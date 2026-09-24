"""Predicting and explaining T1/T4 service disruptions from real-time GTFS.

Three workflow stages, one orchestrator (see ``docs/01-architecture.md``):

* ``realtime``   — live TfNSW GTFS-Realtime conditions, exposed as MCP tools.
  RQ1 Objective 1, and deterministic Python rather than an LLM.
* ``prediction`` — is T1/T4 disrupted, and by how much, with an interval the
  answer has to state. RQ1 Objective 2.
* ``retrieval``  — past T1/T4 service alerts, embedded into Chroma, so a cause
  can be given with the evidence it rests on. RQ1 Objective 3.

``evaluation`` scores each stage separately (RQ2). It is deliberately not an
end-to-end score: a right answer reached through a fabricated reason is not a
success, and one number cannot tell the two apart.

``ingestion`` still holds the Opal fare-policy PDF path. Opal left scope on
2026-09-22; the code is retained because it works and because a second document
type keeps the citation contract from quietly becoming alert-specific.
"""

__version__ = "0.1.0"
