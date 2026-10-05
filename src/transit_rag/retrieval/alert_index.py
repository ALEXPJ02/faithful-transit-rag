"""The alert index's name, and the check that it describes the snapshot in hand.

Shared by everything that retrieves past incidents: the reasons evaluation and the
agent's tools. A retrieval run against an index built from different incidents
would score on a corpus nobody chose, so both refuse it the same way, with the
command that fixes it (``docs/10`` §5).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from transit_rag.config import configured_embedding_model
from transit_rag.ingestion.alerts import Incident, incident_corpus_hash, incident_passages
from transit_rag.retrieval.index import describe, stale_reasons

ALERT_COLLECTION = "tfnsw_alerts"


def check_alert_index(collection: Any, incidents: Sequence[Incident], db: Path) -> None:
    """Refuse an alert index built from different incidents or a different model."""
    reasons = stale_reasons(
        describe(collection),
        embedding_model=configured_embedding_model(),
        target_chars=0,
        overlap_chars=0,
        expected_corpus_hash=incident_corpus_hash(incident_passages(incidents)),
    )
    if reasons:
        raise ValueError(
            "the alert index does not match this snapshot's incidents: "
            + "; ".join(reasons)
            + f". Rebuild it with `transit-index build --source alerts --db {db}`."
        )
