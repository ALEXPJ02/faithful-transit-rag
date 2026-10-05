"""The orchestrator's tools: one per workflow stage, all answering as of one moment.

Each tool is a JSON-schema definition for the model and a handler over a
:class:`~transit_rag.agent.feed.SnapshotFeed`. The schemas are ``strict``, so
the model cannot call a tool with a line that is not T1 or T4. A handler never
raises into the loop. A failure comes back as an error result the model can read
and say, because an answer built on a tool that silently failed is the
unfaithfulness the evaluation looks for.

| Tool | Stage | What it returns |
| --- | --- | --- |
| ``line_status`` | O1 | Services observed and late in the last 30 minutes, the worst delay, where late trains were seen, and the operator's alerts in the feed |
| ``disruption_risk`` | O2 | The detector's probability that the line is disrupted in the next 30 minutes, and its threshold |
| ``similar_past_incidents`` | O3 | Past incidents most like the present, first seen before now, with the alert ids to cite |
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

from transit_rag.agent.feed import SnapshotFeed
from transit_rag.ingestion.alerts import Incident, sydney_time
from transit_rag.prediction.disruption.features import detection_table
from transit_rag.prediction.disruption.labels import WINDOW, label_windows, load_coverage
from transit_rag.prediction.disruption.models import FittedDetector
from transit_rag.retrieval.search import Retriever

LINES: tuple[str, ...] = ("T1", "T4")


def _line_tool(name: str, description: str) -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "line": {"type": "string", "enum": list(LINES), "description": "T1 or T4"}
            },
            "required": ["line"],
            "additionalProperties": False,
        },
    }


TOOL_SPECS: list[dict[str, Any]] = [
    _line_tool(
        "line_status",
        "What the live feed shows on a line now: services observed in the last 30 minutes, "
        "how many are more than five minutes late, the worst delay, the stations where late "
        "trains were seen, and the operator's alerts currently in the feed for the line.",
    ),
    _line_tool(
        "disruption_risk",
        "The disruption detector's probability that the line will be disrupted in the next 30 "
        "minutes, from the last complete 15-minute window, with the threshold above which it "
        "flags a disruption. A model estimate, not an observation.",
    ),
    _line_tool(
        "similar_past_incidents",
        "Past disruptions most similar to what the line shows now, each with its cause and the "
        "alert ids to cite. Only incidents first reported before now are returned.",
    ),
]


def detection_rows(feed: SnapshotFeed, incidents: list[Incident]) -> pd.DataFrame:
    """The snapshot's detection table: every window's features, built once per run."""
    labels = label_windows(feed.events, incidents, load_coverage(feed.db_path))
    return detection_table(labels, feed.events, feed.alerts, feed.alert_polls)


@dataclass
class ToolBox:
    """The tools bound to one snapshot and one moment."""

    feed: SnapshotFeed
    at: datetime
    detector: FittedDetector | None = None
    detector_provenance: dict[str, Any] = field(default_factory=dict)
    rows: pd.DataFrame | None = None
    retriever: Retriever | None = None
    k: int = 5

    def __post_init__(self) -> None:
        if self.at.tzinfo is None:
            raise ValueError("the moment the tools answer for must be timezone-aware")

    def call(self, name: str, arguments: dict[str, Any]) -> tuple[str, bool]:
        """Run a tool. Returns its JSON result, and whether it is an error."""
        handlers: dict[str, Callable[[str], dict[str, Any]]] = {
            "line_status": self.line_status,
            "disruption_risk": self.disruption_risk,
            "similar_past_incidents": self.similar_past_incidents,
        }
        if name not in handlers:
            return json.dumps({"error": f"there is no tool named {name!r}"}), True
        line = arguments.get("line")
        if line not in LINES:
            return json.dumps({"error": f"line must be one of {', '.join(LINES)}"}), True
        try:
            return json.dumps(handlers[name](line)), False
        except Exception as exc:  # reported to the model, never swallowed
            return json.dumps({"error": f"{type(exc).__name__}: {exc}"}), True

    def line_status(self, line: str) -> dict[str, Any]:
        situation = self.feed.situation(line, self.at)
        return {
            "line": line,
            "as_of": sydney_time(self.at),
            "summary": situation.describe(),
            "services_observed_last_30_min": situation.services,
            "more_than_5_min_late": situation.late,
            "worst_delay_minutes": (
                None if situation.max_delay_s is None else round(situation.max_delay_s / 60)
            ),
            "late_at_stations": dict(situation.delayed_stations),
            "alerts_in_feed": [
                {
                    "alert_id": alert.short_id,
                    "cause": alert.cause,
                    "header": alert.header_text,
                    "description": alert.description_text[:500],
                    "first_seen": sydney_time(alert.first_seen),
                }
                for alert in self.feed.alerts_at(self.at, line)
            ],
        }

    def disruption_risk(self, line: str) -> dict[str, Any]:
        if self.detector is None or self.rows is None:
            raise ValueError(
                "no disruption detector is loaded; save one with transit-detect --save"
            )
        window_end = pd.Timestamp(self.at).floor(WINDOW)
        window_start = window_end - WINDOW
        row = self.rows[
            (self.rows["line"] == line) & (self.rows["window_start_utc"] == window_start)
        ]
        if row.empty:
            raise ValueError("the snapshot holds no complete window ending before this moment")
        probability = float(self.detector.score(row).iloc[0])
        date = str(row["service_date"].iloc[0])
        seen = self.detector_provenance
        return {
            "line": line,
            "window": f"{sydney_time(window_start)} to {sydney_time(window_end)[-5:]}",
            "probability_disrupted_next_30_min": round(probability, 3),
            "flagged": probability >= self.detector.threshold,
            "threshold": round(self.detector.threshold, 3),
            "detector": self.detector.name,
            "detector_validation_average_precision": round(self.detector.validation_ap, 3),
            "this_date_was_in_training": date in seen.get("train_dates", [])
            or date in seen.get("validation_dates", []),
            "note": "a model estimate of the next 30 minutes, not an observation",
        }

    def similar_past_incidents(self, line: str) -> dict[str, Any]:
        if self.retriever is None:
            raise ValueError("no alert index is open")
        query = self.feed.situation(line, self.at).describe()
        passages = self.retriever.search(query, self.k, seen_before=self.at)
        return {
            "line": line,
            "searched_for": query,
            "incidents": [
                {
                    "alert_ids": [
                        alert_id[:8]
                        for alert_id in str(passage.metadata.get("alert_ids", "")).split(",")
                        if alert_id
                    ],
                    "cause": passage.metadata.get("cause"),
                    "cause_group": passage.metadata.get("cause_group"),
                    "text": passage.text,
                }
                for passage in passages
            ],
        }
