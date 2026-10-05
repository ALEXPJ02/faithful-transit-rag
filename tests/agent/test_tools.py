"""Tests for the orchestrator's tools and the snapshot they read (``agent/tools.py``, ``agent/feed.py``).

Every tool answers as of one moment. These tests pin that it cannot see past
it: alerts by the last poll before it, retrieval by first sighting before it,
risk from the last complete window before it.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from retrieval_fakes import HashingEmbedder
from transit_rag.agent.feed import SnapshotFeed
from transit_rag.agent.tools import TOOL_SPECS, ToolBox
from transit_rag.ingestion.alerts import (
    AlertRecord,
    group_incidents,
    incident_corpus_hash,
    incident_passages,
)
from transit_rag.prediction.disruption.features import FEATURE_COLUMNS
from transit_rag.prediction.disruption.models import FittedDetector
from transit_rag.retrieval.embeddings import embed_chunks
from transit_rag.retrieval.index import IndexFingerprint, build_index
from transit_rag.retrieval.search import Retriever

AT = datetime(2026, 10, 1, 22, 50, tzinfo=UTC)  # 08:50 on 2 Oct in Sydney
Q = datetime(2026, 10, 1, 22, 45, tzinfo=UTC)  # the quarter-hour before it


def _alert(alert_id: str, cause: str, first: datetime, last: datetime, text: str) -> AlertRecord:
    return AlertRecord(
        alert_id, cause, "UNKNOWN_EFFECT", "Western Line", text, first, last, ("T1",)
    )


ON_NOW = _alert(
    "aaaa1111-x", "POLICE_ACTIVITY", AT - timedelta(minutes=20), AT + timedelta(hours=1),
    "Allow extra travel time due to a person on the tracks at Central.",
)  # fmt: skip
GONE = _alert(
    "bbbb2222-x", "TECHNICAL_PROBLEM", AT - timedelta(hours=5), AT - timedelta(hours=4),
    "Allow extra travel time due to urgent train repairs at Strathfield earlier.",
)  # fmt: skip
LATER = _alert(
    "cccc3333-x", "MEDICAL_EMERGENCY", AT + timedelta(minutes=10), AT + timedelta(minutes=40),
    "Allow extra travel time due to a medical emergency at Redfern.",
)  # fmt: skip


def _feed() -> SnapshotFeed:
    events = pd.DataFrame(
        {
            "line": "T1",
            "service_date": "2026-10-02",
            "trip_id": [f"t{i}" for i in range(6)],
            "stop_id": "2000331",
            "stops_ahead": 0,
            "delay_s": pd.array([600, 420, 0, 0, 0, 0], dtype="Float64"),
            "observed_at": pd.to_datetime([AT - timedelta(minutes=10)] * 6, utc=True),
        }
    )
    # Every half hour for the six hours to AT, as the collector polls alerts.
    polls = pd.Series(pd.date_range(AT - timedelta(hours=6), AT, freq="30min"))
    return SnapshotFeed(
        Path("unused.db"), events, [ON_NOW, GONE, LATER], polls, {"2000331": "Central"}
    )


class TestFeed:
    def test_an_alert_counts_only_if_the_last_poll_before_now_carried_it(self) -> None:
        assert [a.alert_id for a in _feed().alerts_at(AT, "T1")] == [ON_NOW.alert_id]

    def test_an_alert_is_seen_while_it_was_carried_and_not_after(self) -> None:
        assert _feed().alerts_at(AT - timedelta(hours=4, minutes=30), "T1") == [GONE]
        assert _feed().alerts_at(AT - timedelta(hours=3), "T1") == []

    def test_before_any_alert_poll_there_are_no_alerts(self) -> None:
        assert _feed().alerts_at(AT - timedelta(days=1), "T1") == []


def test_every_tool_is_strict_and_takes_only_t1_or_t4() -> None:
    for spec in TOOL_SPECS:
        assert spec["strict"] is True
        schema = spec["input_schema"]
        assert schema["additionalProperties"] is False
        assert schema["properties"]["line"]["enum"] == ["T1", "T4"]


class TestLineStatus:
    def test_it_reports_the_feed_and_the_alerts_in_it_now(self) -> None:
        output, is_error = ToolBox(feed=_feed(), at=AT).call("line_status", {"line": "T1"})
        status = json.loads(output)
        assert not is_error
        assert status["as_of"] == "2026-10-02 08:50"
        assert status["services_observed_last_30_min"] == 6
        assert status["more_than_5_min_late_when_last_reported"] == 2
        assert status["more_than_5_min_late_at_some_point_by_station"] == {"Central": 2}
        assert [alert["alert_id"] for alert in status["alerts_in_feed"]] == ["aaaa1111"]


class TestCall:
    def test_an_unknown_tool_or_line_is_an_error_result_not_an_exception(self) -> None:
        tools = ToolBox(feed=_feed(), at=AT)
        assert tools.call("teleport", {"line": "T1"})[1] is True
        output, is_error = tools.call("line_status", {"line": "T8"})
        assert is_error and "T1, T4" in json.loads(output)["error"]

    def test_a_failing_tool_says_what_failed(self) -> None:
        output, is_error = ToolBox(feed=_feed(), at=AT).call("disruption_risk", {"line": "T1"})
        assert is_error and "no disruption detector" in json.loads(output)["error"]

    def test_a_naive_moment_is_refused(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            ToolBox(feed=_feed(), at=datetime(2026, 10, 2, 8, 50))


class _Constant:
    """A model whose probability is fixed, to see which row the tool scores."""

    def __init__(self, by_hour: dict[int, float]) -> None:
        self.by_hour = by_hour

    def predict_proba(self, matrix: pd.DataFrame) -> np.ndarray:
        p = np.array([self.by_hour[int(h)] for h in matrix["hour_local"]])
        return np.column_stack([1 - p, p])


def _rows() -> pd.DataFrame:
    starts = pd.to_datetime([Q - timedelta(minutes=30), Q - timedelta(minutes=15), Q], utc=True)
    frame = pd.DataFrame({column: 0.0 for column in FEATURE_COLUMNS}, index=range(3))
    frame["line"] = "T1"
    frame["window_start_utc"] = starts
    frame["service_date"] = "2026-10-02"
    frame["hour_local"] = [1, 2, 3]  # identifies the window to the fake model
    return frame


class TestDisruptionRisk:
    def test_it_scores_the_last_complete_window_and_states_the_threshold(self) -> None:
        detector = FittedDetector(
            "xgboost", _Constant({1: 0.1, 2: 0.4, 3: 0.9}), FEATURE_COLUMNS, {}, 0.63, 0.25
        )
        tools = ToolBox(
            feed=_feed(),
            at=AT,  # 08:50: the 08:30-08:45 window is the last complete one
            detector=detector,
            detector_provenance={"train_dates": ["2026-09-30"], "validation_dates": ["2026-10-01"]},
            rows=_rows(),
        )
        risk = json.loads(tools.call("disruption_risk", {"line": "T1"})[0])
        assert risk["probability_disrupted_next_30_min"] == 0.4
        assert risk["flagged"] is True and risk["threshold"] == 0.25
        assert risk["window"] == "2026-10-02 08:30 to 08:45"
        assert risk["this_date_was_in_training"] is False

    def test_it_says_when_the_detector_has_seen_the_date(self) -> None:
        detector = FittedDetector("xgboost", _Constant({2: 0.4}), FEATURE_COLUMNS, {}, 0.6, 0.5)
        tools = ToolBox(
            feed=_feed(),
            at=AT,
            detector=detector,
            detector_provenance={"train_dates": ["2026-10-02"]},
            rows=_rows(),
        )
        assert json.loads(tools.call("disruption_risk", {"line": "T1"})[0])[
            "this_date_was_in_training"
        ]


def test_past_incidents_never_include_one_first_seen_after_now(tmp_path: Path) -> None:
    passages = incident_passages(group_incidents([ON_NOW, GONE, LATER]))
    embedder = HashingEmbedder()
    collection = build_index(
        passages,
        embed_chunks(passages, embedder),
        persist_dir=tmp_path / "chroma",
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
    tools = ToolBox(feed=_feed(), at=AT, retriever=Retriever(collection, embedder))
    found = json.loads(tools.call("similar_past_incidents", {"line": "T1"})[0])["incidents"]
    ids = {alert_id for incident in found for alert_id in incident["alert_ids"]}
    assert ids == {"aaaa1111", "bbbb2222"}  # Redfern, first seen ten minutes later, is not there
