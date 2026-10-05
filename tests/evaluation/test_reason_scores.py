"""Tests for scoring the reasons stage (``evaluation/reasons.py``) and its command.

The command test runs end to end on a snapshot written by the real store and an
index built in Chroma by the offline hashing embedder. It checks the parts a
reader cannot see in a printed score: that retrieval is guarded, that the index
is checked against the snapshot, and that every answer is kept.
"""

from __future__ import annotations

import json
import re
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from retrieval_fakes import HashingEmbedder
from transit_rag.agent.reasons import CauseGroup, ReasonAnswer
from transit_rag.agent.situation import Situation
from transit_rag.evaluation import reasons_cli
from transit_rag.evaluation.reasons import (
    ReasonCase,
    build_cases,
    most_common_cause,
    retrieval_recall,
    score_reasons,
)
from transit_rag.ingestion.alerts import (
    AlertRecord,
    Incident,
    audit,
    incident_corpus_hash,
    incident_passages,
    load_alerts,
    load_overrides,
)
from transit_rag.prediction.collection.store import SqliteObservationStore
from transit_rag.realtime.parsing import AlertScope, ServiceAlert, StopDelayObservation
from transit_rag.retrieval import alert_index
from transit_rag.retrieval.alert_index import check_alert_index
from transit_rag.retrieval.embeddings import embed_chunks
from transit_rag.retrieval.index import IndexFingerprint, build_index

DAY = datetime(2026, 9, 21, 6, 0, tzinfo=UTC)  # 16:00 Sydney time


def _incident(alert_id: str, cause: str, start: datetime) -> Incident:
    return Incident(
        alerts=(
            AlertRecord(
                alert_id=alert_id,
                cause=cause,
                effect="UNKNOWN_EFFECT",
                header_text="North Shore Line",
                description_text="Allow extra travel time.",
                first_seen=start,
                last_seen=start + timedelta(minutes=30),
                lines=("T1",),
            ),
        )
    )


def _case(incident: Incident) -> ReasonCase:
    empty = Situation(incident.lines, incident.first_seen, timedelta(minutes=30), 0, 0, None, ())
    return ReasonCase(incident=incident, situation=empty)


class TestScores:
    def test_macro_f1_averages_only_the_groups_that_occur(self) -> None:
        scores = score_reasons(
            ["technical", "technical", "network_incident"],
            ["technical", "weather_external", "network_incident"],
        )
        assert scores.groups == ("technical", "network_incident")
        assert scores.per_group_f1 == {"technical": pytest.approx(2 / 3), "network_incident": 1.0}
        assert scores.macro_f1 == pytest.approx(5 / 6)
        assert scores.accuracy == pytest.approx(2 / 3)

    def test_mismatched_lengths_are_refused(self) -> None:
        with pytest.raises(ValueError):
            score_reasons(["technical"], [])

    def test_retrieval_recall_counts_any_shared_group(self) -> None:
        recall = retrieval_recall(
            ["technical", "network_incident"], [["other_unknown", "technical"], []]
        )
        assert recall == 0.5


class TestMostCommonCause:
    def test_it_counts_only_incidents_before_the_one_explained(self) -> None:
        incidents = [
            _incident("a", "TECHNICAL_PROBLEM", DAY),
            _incident("b", "POLICE_ACTIVITY", DAY + timedelta(days=1)),
            _incident("c", "POLICE_ACTIVITY", DAY + timedelta(days=2)),
            _incident("d", "POLICE_ACTIVITY", DAY + timedelta(days=3)),
        ]
        # Only "a" came before "b": the two later police incidents are the future.
        assert most_common_cause(_case(incidents[1]), incidents) == "technical"

    def test_ties_go_to_the_first_listed_group_and_no_history_to_unknown(self) -> None:
        incidents = [
            _incident("a", "POLICE_ACTIVITY", DAY),
            _incident("b", "TECHNICAL_PROBLEM", DAY + timedelta(hours=1)),
            _incident("c", "OTHER_CAUSE", DAY + timedelta(hours=2)),
        ]
        assert most_common_cause(_case(incidents[2]), incidents) == "technical"
        assert most_common_cause(_case(incidents[0]), incidents) == "other_unknown"


def test_cases_start_from_a_sydney_date() -> None:
    incidents = [
        _incident("a", "TECHNICAL_PROBLEM", DAY),
        _incident("b", "POLICE_ACTIVITY", DAY + timedelta(days=8)),
    ]
    events = pd.DataFrame(
        columns=[
            "line",
            "service_date",
            "trip_id",
            "stop_id",
            "stops_ahead",
            "delay_s",
            "observed_at",
        ]
    ).astype({"delay_s": "Float64", "observed_at": "datetime64[ns, UTC]", "stops_ahead": int})
    cases = build_cases(incidents, events, {}, since="2026-09-29")
    assert [case.incident.incident_id for case in cases] == [incidents[1].incident_id]
    assert cases[0].truth == "network_incident"


def test_an_index_from_other_incidents_is_refused_with_the_rebuild_command() -> None:
    collection = SimpleNamespace(metadata={})
    with pytest.raises(
        ValueError, match=re.escape("transit-index build --source alerts --db snap.db")
    ):
        check_alert_index(collection, [_incident("a", "TECHNICAL_PROBLEM", DAY)], Path("snap.db"))


# --- the command, end to end -----------------------------------------------------------------

REPAIRS = "Allow extra travel time due to urgent train repairs at Central earlier."
POLICE = "Allow extra travel time due to police activity at Central earlier."


def _snapshot(tmp_path: Path) -> Path:
    """Two T1 incidents a day apart, each preceded by late trains at Central."""
    db = tmp_path / "snapshot.db"
    with SqliteObservationStore(db) as store:
        for day, (alert_id, cause, text) in enumerate(
            (("repairs", "TECHNICAL_PROBLEM", REPAIRS), ("police", "POLICE_ACTIVITY", POLICE))
        ):
            first = DAY + timedelta(days=day)
            for minutes in (0, 30):
                seen = (first + timedelta(minutes=minutes)).isoformat()
                store.record_alerts(
                    [
                        ServiceAlert(
                            alert_id,
                            cause,
                            "UNKNOWN_EFFECT",
                            "UNKNOWN_SEVERITY",
                            "North Shore Line",
                            text,
                            "",
                            seen,
                        )
                    ],
                    [AlertScope(alert_id, "NSN_1a", "T1", 0, "", 0, 0, seen)],
                )
            before = first - timedelta(minutes=10)
            store.record_observations(
                [
                    StopDelayObservation(
                        "2026-09-21", f"d{day}-t{i}", "2000331", "NSN_1a", "T1", None, 0,
                        600, None, "SCHEDULED", before.isoformat(),
                    )
                    for i in range(3)
                ]
            )  # fmt: skip
    return db


def _bundle(tmp_path: Path) -> Path:
    path = tmp_path / "bundle.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "stops.txt",
            "stop_id,stop_name,parent_station\n200060,Central Station,\n2000331,Central Station Platform 1,200060\n",
        )
    return path


def _index(db: Path, persist_dir: Path, overrides: Path) -> None:
    _, incidents = audit(load_alerts(db), load_overrides(overrides))
    passages = incident_passages(incidents)
    embedder = HashingEmbedder()
    build_index(
        passages,
        embed_chunks(passages, embedder),
        persist_dir=persist_dir,
        fingerprint=IndexFingerprint.create(
            embedding_model=embedder.model,
            embedding_dimension=embedder.dimension,
            target_chars=0,
            overlap_chars=0,
            chunk_count=len(passages),
            corpus_hash=incident_corpus_hash(passages),
            document_keys=("tfnsw_alerts",),
        ),
        collection_name="tfnsw_alerts",
    )


class _FakeReasoner:
    """Answers "technical" whenever repairs are mentioned anywhere it can see."""

    def __init__(self, model: str, client: object = None) -> None:
        self.model = model
        self.questions: list[str] = []
        self.input_tokens = self.output_tokens = 0

    def answer(self, system: str, user: str) -> ReasonAnswer:
        self.questions.append(user)
        group: CauseGroup = "technical" if "repairs" in user else "other_unknown"
        return ReasonAnswer(
            cause_group=group, explanation="x", cited_alert_ids=["repairs", "invented"]
        )


def test_the_command_guards_retrieval_scores_three_systems_and_keeps_every_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    db, persist = _snapshot(tmp_path), tmp_path / "chroma"
    overrides = tmp_path / "none.csv"
    overrides.write_text("alert_id,is_incident,reason\n", encoding="utf-8")
    _index(db, persist, overrides)
    reasoners: list[_FakeReasoner] = []

    def fake_reasoner(model: str, client: object = None) -> _FakeReasoner:
        reasoners.append(_FakeReasoner(model, client))
        return reasoners[-1]

    monkeypatch.setattr(reasons_cli, "VoyageEmbedder", lambda **_: HashingEmbedder())
    monkeypatch.setattr(
        reasons_cli.VoyageConfig, "from_env", classmethod(lambda cls: SimpleNamespace(api_key="k"))
    )
    monkeypatch.setattr(reasons_cli, "configured_embedding_model", lambda: "fake-embed-1")
    monkeypatch.setattr(alert_index, "configured_embedding_model", lambda: "fake-embed-1")
    monkeypatch.setattr(reasons_cli, "ClaudeReasoner", fake_reasoner)
    monkeypatch.setattr(
        reasons_cli.ModelConfig,
        "from_env",
        classmethod(
            lambda cls: SimpleNamespace(generation_model="fake-model", anthropic_api_key="k")
        ),
    )
    out = tmp_path / "answers.jsonl"
    argv = ["--db", str(db), "--overrides", str(overrides), "--persist-dir", str(persist),
            "--bundle", str(_bundle(tmp_path)), "--model", "--out", str(out)]  # fmt: skip
    assert reasons_cli.main(argv) == 0

    printed = capsys.readouterr().out
    assert "Late services were seen at Central (3)." in printed
    assert "retrieved: nothing earlier" in printed  # the first incident has no past
    assert "most common cause (time-aware)" in printed
    assert "model + retrieval (fake-model)" in printed

    records = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(records) == 4  # two incidents x two model systems
    police = [r for r in records if r["truth"] == "network_incident"]
    without = next(r for r in police if r["system"] == "model, no retrieval")
    with_retrieval = next(r for r in police if r["system"] == "model + retrieval")
    # The police incident's own alert never reaches the model; the earlier
    # repairs incident does, through retrieval only.
    assert "police" not in " ".join(reasoners[0].questions).lower()
    assert with_retrieval["retrieved"] and not without["retrieved"]
    assert with_retrieval["unsupported_citations"] == ["invented"]
