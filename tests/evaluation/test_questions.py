"""Tests for the scored question set (``evaluation/questions.py``).

They pin the rules that choose the questions: the peaks on every test date and line,
incidents a quarter of an hour after their first sighting, by their service date, and
the templates in turn. The runner's tests check that each question is asked as of its
own moment and that a scored run is never written over another.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from transit_rag.agent.feed import SnapshotFeed
from transit_rag.agent.tools import ToolBox
from transit_rag.evaluation import questions
from transit_rag.evaluation.questions import (
    TEMPLATES,
    Question,
    question_set,
    read_questions,
    write_questions,
)
from transit_rag.ingestion.alerts import AlertRecord, Incident
from transit_rag.realtime.parsing import SYDNEY

TEST_DATES = ["2026-10-17", "2026-10-18"]


def _incident(alert_id: str, first_seen: datetime, lines: tuple[str, ...]) -> Incident:
    return Incident(
        alerts=(
            AlertRecord(
                alert_id=alert_id,
                cause="POLICE_ACTIVITY",
                effect="UNKNOWN_EFFECT",
                header_text="North Shore Line",
                description_text="Allow extra travel time.",
                first_seen=first_seen,
                last_seen=first_seen + timedelta(minutes=30),
                lines=lines,
            ),
        )
    )


def _sydney(text: str) -> datetime:
    return datetime.fromisoformat(text).replace(tzinfo=SYDNEY)


INCIDENTS = [
    _incident("before", _sydney("2026-10-16 12:00"), ("T1",)),  # not a test date
    _incident("both", _sydney("2026-10-17 18:05"), ("T1", "T4")),
    # 01:30 on the 19th is still service date 18 October: the day turns over at 03:00.
    _incident("late", _sydney("2026-10-19 01:30"), ("T4",)),
    _incident("bus", _sydney("2026-10-18 12:00"), ("BMT",)),  # not a tracked line
]


class TestTheSet:
    def test_the_moments_are_the_peaks_and_the_test_date_incidents(self) -> None:
        asked = question_set(TEST_DATES, INCIDENTS)
        peaks = [item for item in asked if item.why == "peak"]
        assert len(peaks) == 2 * 2 * 2  # dates x peaks x lines
        assert {(item.at.astimezone(SYDNEY).strftime("%d %H:%M"), item.line) for item in peaks} == {
            (f"{day} {clock}", line)
            for day in ("17", "18")
            for clock in ("08:30", "17:30")
            for line in ("T1", "T4")
        }
        incident_moments = {
            (item.why, item.line, item.at.astimezone(SYDNEY).strftime("%d %H:%M"))
            for item in asked
            if item.why != "peak"
        }
        assert incident_moments == {
            (f"incident {INCIDENTS[1].incident_id}", "T1", "17 18:20"),
            (f"incident {INCIDENTS[1].incident_id}", "T4", "17 18:20"),
            (f"incident {INCIDENTS[2].incident_id}", "T4", "19 01:45"),
        }

    def test_questions_run_in_time_order_and_the_templates_take_turns(self) -> None:
        asked = question_set(TEST_DATES, INCIDENTS)
        assert [item.at for item in asked] == sorted(item.at for item in asked)
        for index, item in enumerate(asked):
            assert item.question == TEMPLATES[index % len(TEMPLATES)].format(line=item.line)
        assert question_set(TEST_DATES, INCIDENTS) == asked

    def test_the_set_survives_its_file(self, tmp_path: Path) -> None:
        asked = question_set(TEST_DATES, INCIDENTS)
        path = tmp_path / "questions.csv"
        write_questions(asked, path)
        back = read_questions(path)
        assert back == asked
        assert all(item.at.utcoffset() is not None for item in back)

    def test_a_moment_without_its_offset_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "questions.csv"
        path.write_text("at,line,question,why\n2026-10-17T08:30,T1,q,peak\n", encoding="utf-8")
        with pytest.raises(ValueError, match="no UTC offset"):
            read_questions(path)


def test_writing_uses_the_detection_splits_test_dates_and_is_done_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Ten dates: the split keeps the last two (15%, rounded) for test.
    dates = [f"2026-10-{day:02d}" for day in range(9, 19)]
    table = pd.DataFrame({"service_date": dates})
    monkeypatch.setattr(questions, "build", lambda db, overrides, rule: (table, INCIDENTS))
    out = tmp_path / "questions.csv"
    argv = ["write", "--db", str(tmp_path / "frozen.db"), "--out", str(out)]
    assert questions.main(argv) == 0
    assert "Test dates: 2026-10-17, 2026-10-18" in capsys.readouterr().out
    assert len(read_questions(out)) == 8 + 3
    assert questions.main(argv) == 1
    assert "written once" in capsys.readouterr().err


class _Answers:
    """Answers every question at once, recording the moment the tools were set to."""

    def __init__(self) -> None:
        self.messages = self

    def create(self, **_: Any) -> SimpleNamespace:
        return SimpleNamespace(
            stop_reason="end_turn",
            content=[SimpleNamespace(type="text", text="No tool was needed.")],
            usage=SimpleNamespace(input_tokens=10, output_tokens=5),
        )


def _toolbox(at: datetime, k: int = 5) -> ToolBox:
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
    )
    feed = SnapshotFeed(
        Path("unused.db"), events, [], pd.Series([], dtype="datetime64[ns, UTC]"), {}
    )
    return ToolBox(feed=feed, at=at, k=k)


def test_each_question_is_asked_as_of_its_own_moment_and_runs_never_mix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    asked = [
        Question(datetime(2026, 10, 16, 21, 30, tzinfo=UTC), "T1", "q one", "peak"),
        Question(datetime(2026, 10, 17, 6, 30, tzinfo=UTC), "T4", "q two", "incident x"),
    ]
    path = tmp_path / "questions.csv"
    write_questions(asked, path)
    built: list[datetime] = []

    def fake_toolbox(args: Any, at: datetime | None = None) -> ToolBox:
        assert at is not None
        built.append(at)
        return _toolbox(at, k=args.k)

    monkeypatch.setattr(questions, "build_toolbox", fake_toolbox)
    monkeypatch.setattr(
        questions.ModelConfig,
        "from_env",
        classmethod(lambda cls: SimpleNamespace(generation_model="fake", anthropic_api_key="k")),
    )
    import anthropic

    monkeypatch.setattr(anthropic, "Anthropic", lambda api_key: _Answers())
    out_dir = tmp_path / "transcripts"
    argv = [
        "ask",
        "--questions",
        str(path),
        "--out-dir",
        str(out_dir),
        "--db",
        str(tmp_path / "x.db"),
        "--k",
        "3",
    ]
    assert questions.main(argv) == 0
    assert built == [asked[0].at]  # built once, then moved
    written = sorted(out_dir.glob("*.json"))
    assert [p.name for p in written] == ["001_T1_20261017_0830.json", "002_T4_20261017_1730.json"]
    records = [json.loads(p.read_text(encoding="utf-8")) for p in written]
    assert [r["as_of"] for r in records] == [item.at.isoformat() for item in asked]
    assert [r["k"] for r in records] == [3, 3]  # the frozen k, not the default 5
    assert [(r["question"], r["why"]) for r in records] == [
        ("q one", "peak"),
        ("q two", "incident x"),
    ]
    assert "2 transcript(s)" in capsys.readouterr().out

    assert questions.main(argv) == 1
    assert "written once" in capsys.readouterr().err
