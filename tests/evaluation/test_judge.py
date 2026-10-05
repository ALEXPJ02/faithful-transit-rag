"""Tests for the faithfulness judge and its validation (``evaluation/judge.py``, ``kappa.py``).

No test calls a model. These pin how answers become statements, what evidence each
is judged against, how citations are counted, how a failed verdict is kept out of
the rate, and the κ arithmetic the judge is accepted or rejected on.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from transit_rag.evaluation import judge_cli
from transit_rag.evaluation.judge import (
    Answer,
    ClaudeJudge,
    JudgedStatement,
    JudgeVerdict,
    citation_coverage,
    cited_alert_ids,
    faithfulness_rate,
    judge_answer,
    prompt_fingerprint,
    reasons_answers,
    split_statements,
    transcript_answer,
)
from transit_rag.evaluation.kappa import cohens_kappa, read_labels, stratified_sample


class TestStatements:
    def test_sentences_split_but_decimals_and_ids_do_not(self) -> None:
        text = (
            "Caringbah 15.0 min [12.4, 17.6]. The cause is a person on the tracks, per "
            "alert ad28913f. 6 of 31 were late!"
        )
        assert split_statements(text) == [
            "Caringbah 15.0 min [12.4, 17.6].",
            "The cause is a person on the tracks, per alert ad28913f.",
            "6 of 31 were late!",
        ]

    def test_abbreviations_do_not_end_a_sentence_and_brackets_do_not_hide_one(self) -> None:
        text = (
            "Delays at St. Leonards and Mt. Druitt (e.g. Central, i.e. The City) took approx. 5 "
            "min vs. 2 Sunday. (Per ad28913f.) Then 2."
        )
        assert split_statements(text) == [
            "Delays at St. Leonards and Mt. Druitt (e.g. Central, i.e. The City) took approx. 5 "
            "min vs. 2 Sunday.",
            "(Per ad28913f.)",
            "Then 2.",
        ]

    def test_each_list_item_is_its_own_statement(self) -> None:
        assert split_statements("Late at:\n- Central, 9 min\n\n- Redfern, 6 min") == [
            "Late at:",
            "- Central, 9 min",
            "- Redfern, 6 min",
        ]

    def test_alert_ids_are_eight_hex_digits(self) -> None:
        assert cited_alert_ids("per ad28913f and 7e669567, not abc1234 or ad28913f00") == {
            "ad28913f",
            "7e669567",
        }


class _Fixed:
    model = "fake-judge"

    def __init__(self, *verdicts: bool | Exception) -> None:
        self.verdicts = list(verdicts)
        self.seen: list[tuple[str, str]] = []

    def verdict(self, evidence: str, statement: str) -> JudgeVerdict:
        self.seen.append((evidence, statement))
        outcome = self.verdicts.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return JudgeVerdict(supported=outcome, rationale="because")


ANSWER = Answer(
    answer_id="agent|q1",
    system="agent",
    text="Alert ad28913f says a person is on the tracks. Trains are 9 minutes late. It may rain.",
    evidence="[line_status] ... ad28913f ...",
    shown_ids=frozenset({"ad28913f"}),
)


class TestJudgeAnswer:
    def test_each_statement_is_judged_against_the_same_evidence(self) -> None:
        judge = _Fixed(True, True, False)
        statements = judge_answer(ANSWER, judge)
        assert [s.supported for s in statements] == [True, True, False]
        assert [statement for _, statement in judge.seen] == split_statements(ANSWER.text)
        assert {evidence for evidence, _ in judge.seen} == {ANSWER.evidence}
        assert [s.statement_id for s in statements] == ["agent|q1#0", "agent|q1#1", "agent|q1#2"]

    def test_only_a_shown_alert_counts_as_a_citation(self) -> None:
        statements = judge_answer(ANSWER, _Fixed(True, True, True))
        assert [s.cites_shown_alert for s in statements] == [True, False, False]
        invented = Answer("a", "agent", "See alert deadbeef.", "e", frozenset({"ad28913f"}))
        assert judge_answer(invented, _Fixed(False))[0].cites_shown_alert is False

    def test_a_failed_verdict_is_kept_out_of_the_rate_and_counted(self) -> None:
        statements = judge_answer(ANSWER, _Fixed(True, RuntimeError("timeout"), False))
        assert statements[1].supported is None
        assert "judge failed" in statements[1].rationale
        assert faithfulness_rate(statements) == 0.5  # one of the two judged
        assert citation_coverage(statements) == pytest.approx(1 / 3)  # over all three

    def test_nothing_judged_is_undefined_not_zero(self) -> None:
        assert math.isnan(faithfulness_rate([]))
        assert math.isnan(citation_coverage([]))


class TestReasonsAnswers:
    RECORD: dict[str, Any] = {  # noqa: RUF012 -- read, never mutated
        "system": "model + retrieval",
        "incident_id": "inc-1",
        "repeat": 0,
        "explanation": "Like the North Sydney repairs. A signal fault is likely.",
        "cited_alert_ids": ["ef6fc9dc"],
        "prompt": "Situation: ...\n\n[alerts ef6fc9dc, fdd1c2cd] cause group: technical\n...",
    }

    def test_judged_against_the_verbatim_prompt_with_the_ids_it_showed(self) -> None:
        failed = {**self.RECORD, "explanation": None, "error": "RefusedError: no"}
        [answer] = reasons_answers([self.RECORD, failed])
        assert answer.answer_id == "model + retrieval|inc-1|0"
        assert answer.evidence == self.RECORD["prompt"]
        assert answer.shown_ids == {"ef6fc9dc", "fdd1c2cd"}
        # A reasons answer cites in a field, so each sentence carries those citations.
        statements = judge_answer(answer, _Fixed(True, True))
        assert [s.cites_shown_alert for s in statements] == [True, True]

    def test_an_answer_without_its_prompt_is_refused_not_skipped(self) -> None:
        legacy = {k: v for k, v in self.RECORD.items() if k != "prompt"}
        with pytest.raises(ValueError, match="no prompt"):
            reasons_answers([legacy])


TRANSCRIPT: dict[str, Any] = {
    "question": "Is T1 disrupted?",
    "as_of": "2026-10-01T08:50:00+10:00",
    "answer": "Yes, per ad28913f.",
    "steps": [
        {"tool": "line_status", "input": {"line": "T1"}, "output": json.dumps({"alerts_in_feed": [{"alert_id": "ad28913f"}]})},
        {"tool": "similar_past_incidents", "input": {"line": "T1"}, "output": json.dumps({"incidents": [{"alert_ids": ["0fac0883"]}]})},
        {"tool": "disruption_risk", "input": {"line": "T1"}, "output": "error: no detector"},
    ],
}  # fmt: skip


def test_an_agent_answer_is_judged_against_the_moment_and_every_tool_output() -> None:
    transcript = TRANSCRIPT
    answer = transcript_answer(transcript, "q1")
    assert answer.shown_ids == {"ad28913f", "0fac0883"}
    # The moment worded as the system prompt words it, then the question.
    assert answer.evidence.startswith(
        "It is 08:50 on Thursday 1 October 2026 in Sydney.\nQuestion: Is T1 disrupted?\n\n"
    )
    assert all(step["output"] in answer.evidence for step in transcript["steps"])


class _Messages:
    def __init__(self, stop_reason: str = "end_turn") -> None:
        self.kwargs: dict[str, Any] = {}
        self.stop_reason = stop_reason

    def parse(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        verdict = JudgeVerdict(supported=True, rationale="r")
        return SimpleNamespace(
            stop_reason=self.stop_reason,
            parsed_output=None if self.stop_reason == "max_tokens" else verdict,
            usage=SimpleNamespace(input_tokens=50, output_tokens=5),
        )


class TestClaudeJudge:
    def test_temperature_zero_goes_where_the_model_accepts_it(self) -> None:
        messages = _Messages()
        judge = ClaudeJudge("claude-haiku-4-5", client=SimpleNamespace(messages=messages))
        assert judge.verdict("the evidence", "the statement").supported is True
        assert messages.kwargs["extra_body"] == {"temperature": 0}
        assert messages.kwargs["output_format"] is JudgeVerdict
        assert "temperature" not in messages.kwargs  # the 1.x signatures have no such keyword
        content = messages.kwargs["messages"][0]["content"]
        assert "<evidence>\nthe evidence\n</evidence>" in content
        assert "<statement>\nthe statement\n</statement>" in content
        judge.verdict("the evidence", "the statement")
        assert (judge.input_tokens, judge.output_tokens) == (100, 10)

    def test_a_model_that_rejects_sampling_gets_none(self) -> None:
        messages = _Messages()
        judge = ClaudeJudge("claude-sonnet-5", client=SimpleNamespace(messages=messages))
        judge.verdict("evidence", "statement")
        assert "extra_body" not in messages.kwargs and judge.deterministic is False

    @pytest.mark.parametrize("stop_reason", ["refusal", "max_tokens"])
    def test_a_refusal_or_a_cut_off_verdict_is_an_error_the_caller_counts(
        self, stop_reason: str
    ) -> None:
        judge = ClaudeJudge(
            "claude-haiku-4-5", client=SimpleNamespace(messages=_Messages(stop_reason))
        )
        with pytest.raises(RuntimeError, match=f"no verdict \\({stop_reason}\\)"):
            judge.verdict("evidence", "statement")


class TestKappa:
    def test_known_values(self) -> None:
        assert cohens_kappa([True, False, True, False], [True, False, True, False]) == 1.0
        # Agreement 3/4; chance 0.5 x 0.25 + 0.5 x 0.75 = 0.5; kappa (0.75 - 0.5) / 0.5.
        assert cohens_kappa([True, True, False, False], [True, False, False, False]) == 0.5
        # Agreement 3/5; chance 0.6 x 0.6 + 0.4 x 0.4 = 0.52; kappa (0.6 - 0.52) / 0.48.
        assert cohens_kappa(
            [True, True, True, False, False], [True, True, False, True, False]
        ) == pytest.approx(1 / 6)

    def test_one_label_for_everything_is_undefined_not_perfect(self) -> None:
        assert math.isnan(cohens_kappa([True, True], [True, True]))
        assert math.isnan(cohens_kappa([], []))

    def test_the_sample_takes_every_stratum_and_leaves_out_failures(self) -> None:
        statements = [
            JudgedStatement(f"s{i:02d}", "a", system, "x", supported, "r", False)
            for i, (system, supported) in enumerate(
                [("agent", True)] * 21 + [("agent", False)] * 2 + [("rag", True)] * 15 + [("rag", None)] * 3
            )
        ]  # fmt: skip
        sample = stratified_sample(statements, share=0.2, seed=1)
        strata = {(s.system, s.supported) for s in sample}
        assert strata == {("agent", True), ("agent", False), ("rag", True)}
        # 20% rounded up, per stratum: 4.2 -> 5, 0.4 -> 1, and exactly 3 of 15.
        assert len(sample) == 5 + 1 + 3
        assert sample == stratified_sample(list(reversed(statements)), share=0.2, seed=1)
        assert sample != stratified_sample(statements, share=0.2, seed=2)
        for share in (0, 1.5):
            with pytest.raises(ValueError, match="share"):
                stratified_sample(statements, share=share)

    def test_a_share_of_a_stratum_is_rounded_up_exactly(self) -> None:
        statements = [
            JudgedStatement(f"s{i:03d}", "a", "agent", "x", True, "r", False) for i in range(100)
        ]
        assert len(stratified_sample(statements, share=0.07)) == 7  # not ceil(7.000000000000001)


def _judged_file(tmp_path: Path, verdicts: list[bool]) -> Path:
    path = tmp_path / "judged.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for i, supported in enumerate(verdicts):
            statement = JudgedStatement(
                f"agent|q#{i}", "agent|q", "agent", f"s{i}", supported, "r", False
            )
            handle.write(json.dumps({**statement.as_dict(), "evidence": "the evidence"}) + "\n")
    return path


def _labelled(sheet: list[dict[str, str]], labels: dict[str, str], path: Path) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sheet[0]))
        writer.writeheader()
        for row in sheet:
            writer.writerow({**row, "human_supported": labels[row["statement_id"]]})
    return path


def test_the_blind_sheet_holds_no_verdict_and_kappa_gates_the_judge(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    judged = _judged_file(tmp_path, [True, True, False, False, True])
    out = tmp_path / "sample.csv"
    command = ["sample", "--judged", str(judged), "--out", str(out), "--share", "1"]
    assert judge_cli.main(command) == 0
    with out.open(newline="", encoding="utf-8") as handle:
        sheet = list(csv.DictReader(handle))
    assert set(sheet[0]) == {"statement_id", "system", "statement", "evidence", "human_supported"}
    assert all(row["human_supported"] == "" for row in sheet)
    assert all(row["evidence"] == "the evidence" for row in sheet)
    capsys.readouterr()

    # One disagreement, on q#4. Agreement 4/5; chance 0.6 x 0.4 + 0.4 x 0.6 = 0.48;
    # kappa (0.8 - 0.48) / 0.52 = 0.62, which passes.
    author = {f"agent|q#{i}": label for i, label in enumerate(["yes", "yes", "no", "no", "no"])}
    labels = _labelled(sheet, author, tmp_path / "labelled.csv")
    assert read_labels(labels) == {
        "agent|q#0": True,
        "agent|q#1": True,
        "agent|q#2": False,
        "agent|q#3": False,
        "agent|q#4": False,
    }
    assert judge_cli.main(["kappa", "--judged", str(judged), "--labels", str(labels)]) == 0
    printed = capsys.readouterr().out
    assert "Cohen's κ = 0.62 over 5 statement(s); raw agreement 80%" in printed
    assert "the judge passes" in printed
    assert "disagree agent|q#4: judge=True author=False" in printed
    assert "disagree agent|q#3" not in printed

    # A second disagreement: agreement 3/5, chance 0.52, kappa 0.17, below the floor.
    author["agent|q#3"] = "yes"
    labels = _labelled(sheet, author, tmp_path / "labelled.csv")
    assert judge_cli.main(["kappa", "--judged", str(judged), "--labels", str(labels)]) == 2
    assert "Cohen's κ = 0.17" in capsys.readouterr().out


def test_kappa_says_when_a_sample_cannot_validate_the_judge(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    judged = _judged_file(tmp_path, [True, True])
    sheet = [{"statement_id": f"agent|q#{i}", "human_supported": ""} for i in range(2)]
    labels = _labelled(sheet, {"agent|q#0": "yes", "agent|q#1": "yes"}, tmp_path / "l.csv")
    assert judge_cli.main(["kappa", "--judged", str(judged), "--labels", str(labels)]) == 2
    assert "undefined" in capsys.readouterr().out

    empty = tmp_path / "empty.csv"
    empty.write_text("statement_id,system,statement,evidence,human_supported\n")
    assert judge_cli.main(["kappa", "--judged", str(judged), "--labels", str(empty)]) == 1

    stranger = _labelled(
        [{"statement_id": "agent|other#0", "human_supported": ""}],
        {"agent|other#0": "no"},
        tmp_path / "s.csv",
    )
    assert judge_cli.main(["kappa", "--judged", str(judged), "--labels", str(stranger)]) == 1


def test_an_unlabelled_sheet_is_refused(tmp_path: Path) -> None:
    sheet = tmp_path / "sheet.csv"
    sheet.write_text("statement_id,system,statement,evidence,human_supported\nx,agent,s,e,\n")
    with pytest.raises(ValueError, match="yes or no"):
        read_labels(sheet)


def test_the_judge_writes_rows_that_report_and_sample_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    reasons = tmp_path / "reasons.jsonl"
    failed = {**TestReasonsAnswers.RECORD, "explanation": None, "error": "RefusedError: no"}
    reasons.write_text(f"{json.dumps(TestReasonsAnswers.RECORD)}\n{json.dumps(failed)}\n")
    transcripts = tmp_path / "transcripts"
    transcripts.mkdir()
    (transcripts / "q1.json").write_text(json.dumps(TRANSCRIPT))
    fake = _Fixed(True, False, True)  # the reasons answer's two statements, then the agent's
    monkeypatch.setattr(
        judge_cli,
        "ClaudeJudge",
        lambda model, client: SimpleNamespace(
            model=model, verdict=fake.verdict, deterministic=True, input_tokens=7, output_tokens=1
        ),
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_JUDGE_MODEL", "judge-under-test")
    out = tmp_path / "judged.jsonl"
    command = ["judge", "--reasons", str(reasons), "--transcripts", str(transcripts)]
    assert judge_cli.main([*command, "--out", str(out)]) == 0
    assert "judged 3 statement(s) from 2 answer(s)" in capsys.readouterr().out
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [row["statement_id"] for row in rows] == [
        "model + retrieval|inc-1|0#0",
        "model + retrieval|inc-1|0#1",
        "agent|q1#0",
    ]
    assert rows[0]["evidence"] == TestReasonsAnswers.RECORD["prompt"]
    assert rows[2]["evidence"].startswith("It is 08:50 on Thursday 1 October 2026")
    assert {(row["judge_model"], row["judge_prompt"]) for row in rows} == {
        ("judge-under-test", prompt_fingerprint())
    }

    assert judge_cli.main(["report", "--judged", str(out)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == f"judged by judge-under-test, judge prompt {prompt_fingerprint()}"
    agent = next(line for line in lines if line.strip().startswith("agent"))
    reasons_line = next(line for line in lines if "model + retrieval" in line)
    assert agent.split()[-4:] == ["1", "100%", "100%", "0"]
    assert reasons_line.split()[-4:] == ["2", "50%", "100%", "0"]

    sheet = tmp_path / "sample.csv"
    assert judge_cli.main(["sample", "--judged", str(out), "--out", str(sheet)]) == 0
    with sheet.open(newline="", encoding="utf-8") as handle:
        assert {row["statement_id"] for row in csv.DictReader(handle)} == {
            row["statement_id"] for row in rows
        }  # every stratum here holds one statement, and each gets at least one


def test_a_file_from_two_judges_is_refused_and_an_old_prompt_is_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    judged = _judged_file(tmp_path, [True, False])
    rows = [json.loads(line) for line in judged.read_text(encoding="utf-8").splitlines()]
    for row, prompt in zip(rows, ["aaaaaaaaaaaa", "bbbbbbbbbbbb"], strict=True):
        row.update(judge_model="claude-haiku-4-5", judge_prompt=prompt)
    judged.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    assert judge_cli.main(["report", "--judged", str(judged)]) == 1
    assert "mixes verdicts from 2 judges" in capsys.readouterr().err

    rows[1]["judge_prompt"] = "aaaaaaaaaaaa"
    judged.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    assert judge_cli.main(["report", "--judged", str(judged)]) == 0
    assert capsys.readouterr().out.splitlines()[0] == (
        "judged by claude-haiku-4-5, judge prompt aaaaaaaaaaaa; "
        f"the current judge prompt is {prompt_fingerprint()}"
    )


def test_the_fingerprint_follows_the_instructions(monkeypatch: pytest.MonkeyPatch) -> None:
    from transit_rag.evaluation import judge

    before = prompt_fingerprint()
    assert len(before) == 12 and before == prompt_fingerprint()
    monkeypatch.setattr(judge, "JUDGE_SYSTEM", judge.JUDGE_SYSTEM + " Be lenient.")
    assert prompt_fingerprint() != before
    monkeypatch.undo()
    monkeypatch.setattr(judge, "JUDGE_MESSAGE", "<e>{evidence}</e><s>{statement}</s>")
    assert prompt_fingerprint() != before
