"""Incident passages in the index, and the leakage guards on searching them.

The guards are what ``docs/08`` §3.5 rests on: explaining an incident may only
draw on incidents first seen before it, and never on itself. They are tested
against a real Chroma collection, because a filter that Chroma silently
ignores would look exactly like one that works until the scores came back
too good.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from retrieval_fakes import HashingEmbedder
from transit_rag.ingestion.alerts import (
    ALERT_DOCUMENT_TITLE,
    AlertRecord,
    IncidentPassage,
    group_incidents,
    incident_corpus_hash,
    incident_passages,
)
from transit_rag.ingestion.chunks import UnattributableChunkError
from transit_rag.prediction.collection.store import SqliteObservationStore
from transit_rag.realtime.parsing import AlertScope, ServiceAlert
from transit_rag.retrieval.cli import main as index_main
from transit_rag.retrieval.embeddings import embed_chunks
from transit_rag.retrieval.index import IndexFingerprint, build_index, stale_reasons
from transit_rag.retrieval.search import RetrievedPassage, Retriever

pytest.importorskip("chromadb", reason="install the 'rag' extra: pip install -e '.[rag]'")

DAY = datetime(2026, 9, 21, 6, 0, tzinfo=UTC)


def _record(alert_id: str, cause: str, description: str, start: datetime) -> AlertRecord:
    return AlertRecord(
        alert_id=alert_id,
        cause=cause,
        effect="UNKNOWN_EFFECT",
        header_text="North Shore Line",
        description_text=description,
        first_seen=start,
        last_seen=start + timedelta(minutes=30),
        lines=("T1",),
    )


#: Three incidents a day apart, distinguishable by vocabulary.
RECORDS = [
    _record("repairs-1", "TECHNICAL_PROBLEM", "urgent train repairs at North Sydney", DAY),
    _record("police-2", "POLICE_ACTIVITY", "police operation at Edgecliff", DAY + timedelta(1)),
    _record(
        "repairs-3", "TECHNICAL_PROBLEM", "urgent train repairs at Chatswood", DAY + timedelta(2)
    ),
]


def _passages() -> list[IncidentPassage]:
    return incident_passages(group_incidents(RECORDS))


@pytest.fixture
def retriever(tmp_path: Path) -> Retriever:
    passages = _passages()
    embedder = HashingEmbedder()
    vectors = embed_chunks(passages, embedder)
    fingerprint = IndexFingerprint.create(
        embedding_model=embedder.model,
        embedding_dimension=embedder.dimension,
        target_chars=0,
        overlap_chars=0,
        chunk_count=len(passages),
        corpus_hash=incident_corpus_hash(passages),
        document_keys=("tfnsw_alerts",),
    )
    collection = build_index(
        passages,
        vectors,
        persist_dir=tmp_path / "chroma",
        fingerprint=fingerprint,
        collection_name="tfnsw_alerts",
    )
    return Retriever(collection, embedder)


class TestIncidentPassage:
    def test_cites_every_alert_and_when_it_was_first_seen(self) -> None:
        [passage] = incident_passages(group_incidents(RECORDS[:1]))
        # 06:00 UTC is 16:00 in Sydney.
        assert passage.citation == "TfNSW alert repairs- (first seen 2026-09-21 16:00 Sydney time)"
        assert passage.locator == "alerts repairs-"

    def test_metadata_carries_what_the_leakage_filter_reads(self) -> None:
        [passage] = incident_passages(group_incidents(RECORDS[:1]))
        metadata = passage.metadata()
        assert metadata["first_seen_ts"] == int(DAY.timestamp())
        assert metadata["incident_id"] == passage.incident_id
        assert metadata["cause_group"] == "technical"
        assert metadata["document_title"] == ALERT_DOCUMENT_TITLE

    def test_a_passage_without_alerts_cannot_exist(self) -> None:
        with pytest.raises(UnattributableChunkError):
            IncidentPassage("inc-x", (), ("T1",), "UNKNOWN_CAUSE", "other_unknown", DAY, DAY, "t")

    def test_a_naive_time_is_refused(self) -> None:
        naive = datetime(2026, 9, 21, 16, 0)
        with pytest.raises(UnattributableChunkError, match="naive"):
            IncidentPassage(
                "inc-x", ("a",), ("T1",), "UNKNOWN_CAUSE", "other_unknown", naive, naive, "t"
            )


class TestCorpusHash:
    def test_order_does_not_matter(self) -> None:
        passages = _passages()
        assert incident_corpus_hash(passages) == incident_corpus_hash(list(reversed(passages)))

    def test_changed_text_is_a_different_corpus(self) -> None:
        changed = [_record("repairs-1", "TECHNICAL_PROBLEM", "a different description", DAY)]
        assert incident_corpus_hash(incident_passages(group_incidents(changed))) != (
            incident_corpus_hash(incident_passages(group_incidents(RECORDS[:1])))
        )

    def test_a_stored_index_from_other_incidents_is_stale(self) -> None:
        stored = IndexFingerprint.create(
            embedding_model="m",
            embedding_dimension=4,
            target_chars=0,
            overlap_chars=0,
            chunk_count=1,
            corpus_hash="0" * 64,
            document_keys=("tfnsw_alerts",),
        )
        reasons = stale_reasons(
            stored,
            embedding_model="m",
            target_chars=0,
            overlap_chars=0,
            expected_corpus_hash="f" * 64,
        )
        assert len(reasons) == 1 and "different incidents" in reasons[0]


class TestLeakageGuards:
    def test_unfiltered_search_ranks_by_similarity(self, retriever: Retriever) -> None:
        passages = retriever.search("train repairs at Chatswood", k=3)
        assert "Chatswood" in passages[0].text
        assert passages[0].locator.startswith("alerts ")
        assert passages[0].page == 0

    def test_seen_before_excludes_every_later_incident(self, retriever: Retriever) -> None:
        """Explaining the Edgecliff incident: Chatswood, a day later, is the
        future and must not be retrievable, however similar."""
        passages = retriever.search(
            "train repairs at Chatswood", k=3, seen_before=DAY + timedelta(1)
        )
        assert [p.metadata["alert_ids"] for p in passages] == ["repairs-1"]

    def test_exclude_incident_removes_only_that_incident(self, retriever: Retriever) -> None:
        own = _passages()[2].incident_id
        passages = retriever.search("train repairs at Chatswood", k=3, exclude_incident=own)
        assert own not in {p.metadata["incident_id"] for p in passages}
        assert len(passages) == 2

    def test_both_guards_together(self, retriever: Retriever) -> None:
        own = _passages()[1].incident_id
        passages = retriever.search(
            "police operation", k=3, seen_before=DAY + timedelta(2), exclude_incident=own
        )
        assert [p.metadata["alert_ids"] for p in passages] == ["repairs-1"]

    def test_a_naive_time_is_refused_rather_than_guessed(self, retriever: Retriever) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            retriever.search("repairs", k=3, seen_before=datetime(2026, 9, 22, 16, 0))


def test_a_returned_passage_needs_a_page_or_a_locator() -> None:
    with pytest.raises(UnattributableChunkError):
        RetrievedPassage("id", "text", "tfnsw_alerts", ALERT_DOCUMENT_TITLE, 0, "c", 0.5, 1)
    RetrievedPassage("id", "text", "tfnsw_alerts", ALERT_DOCUMENT_TITLE, 0, "c", 0.5, 1, "alerts a")


def _snapshot(tmp_path: Path) -> Path:
    db = tmp_path / "snapshot.db"
    seen = "2026-09-21T06:11:00+00:00"
    with SqliteObservationStore(db) as store:
        store.record_alerts(
            [
                ServiceAlert("a1", "TECHNICAL_PROBLEM", "UNKNOWN_EFFECT", "UNKNOWN_SEVERITY",
                             "North Shore Line", "Allow extra travel time due to repairs.", "", seen),
            ],
            [AlertScope("a1", "NSN_2a", "T1", 0, "", 0, 0, seen)],
        )  # fmt: skip
    return db


class TestCli:
    def test_dry_run_counts_incident_passages_without_a_key(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = index_main(
            ["build", "--source", "alerts", "--db", str(_snapshot(tmp_path)), "--dry-run"]
        )
        assert code == 0
        assert "1 incident passages" in capsys.readouterr().out

    def test_the_alerts_source_needs_a_snapshot(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert index_main(["build", "--source", "alerts", "--dry-run"]) == 1
        assert "--db" in capsys.readouterr().err
