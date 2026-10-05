"""Tests for the orchestrator loop (``agent/loop.py``).

The client is scripted: each test fixes what the "model" returns turn by turn
and checks what the loop sent back. That is where a hand-rolled loop goes wrong:
a tool result attached to the wrong call, an error not marked as one, or an
answer taken from a turn that never finished.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

from transit_rag.agent.loop import MAX_TURNS, SYSTEM_TEMPLATE, ask, system_prompt
from transit_rag.agent.tools import TOOL_SPECS

AT = datetime(2026, 10, 1, 22, 50, tzinfo=UTC)  # 08:50 on Friday 2 October in Sydney


def _use(call_id: str, name: str, line: str) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=call_id, name=name, input={"line": line})


def _text(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def _response(stop_reason: str, *content: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(
        stop_reason=stop_reason,
        content=list(content),
        usage=SimpleNamespace(input_tokens=100, output_tokens=20),
    )


class _ScriptedClient:
    def __init__(self, *responses: SimpleNamespace) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []
        self.messages = self

    def create(self, **kwargs: Any) -> SimpleNamespace:
        # Copy the list: the loop keeps appending to the same one.
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        return self.responses.pop(0)


class _Tools:
    """Stands in for ToolBox: answers line_status, fails anything else."""

    at = AT

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def call(self, name: str, arguments: dict[str, Any]) -> tuple[str, bool]:
        self.calls.append((name, arguments))
        if name == "line_status":
            return json.dumps({"line": arguments["line"], "late": 3}), False
        return json.dumps({"error": "no detector"}), True


def test_tool_results_go_back_to_the_calls_that_asked_for_them() -> None:
    client = _ScriptedClient(
        _response(
            "tool_use",
            _text("Checking."),
            _use("call-1", "line_status", "T1"),
            _use("call-2", "disruption_risk", "T1"),
        ),
        _response("end_turn", _text("T1 has 3 late services; the risk could not be checked.")),
    )
    tools = _Tools()
    transcript = ask("Is T1 disrupted?", tools, client, "fake-model")  # type: ignore[arg-type]

    assert transcript.answer == "T1 has 3 late services; the risk could not be checked."
    assert transcript.stop_reason == "end_turn" and transcript.turns == 2
    assert (transcript.input_tokens, transcript.output_tokens) == (200, 40)

    # The second request carries the assistant turn whole, then every result in one message.
    second = client.requests[1]["messages"]
    assert second[1]["role"] == "assistant" and len(second[1]["content"]) == 3
    results = second[2]["content"]
    assert [r["tool_use_id"] for r in results] == ["call-1", "call-2"]
    assert "is_error" not in results[0] and results[1]["is_error"] is True
    assert [step.tool for step in transcript.steps] == ["line_status", "disruption_risk"]
    assert transcript.steps[1].is_error


def test_every_request_has_the_system_prompt_and_the_tools() -> None:
    client = _ScriptedClient(_response("end_turn", _text("ok")))
    ask("q", _Tools(), client, "fake-model")  # type: ignore[arg-type]
    request = client.requests[0]
    assert request["tools"] == TOOL_SPECS
    assert request["system"] == system_prompt(AT)
    assert "temperature" not in request


def test_the_moment_is_told_in_sydney_time() -> None:
    assert "It is 08:50 on Friday 2 October 2026 in Sydney." in system_prompt(AT)
    assert "{as_of}" in SYSTEM_TEMPLATE


def test_a_refusal_ends_the_loop_and_is_recorded_as_it_stands() -> None:
    client = _ScriptedClient(_response("refusal"))
    transcript = ask("q", _Tools(), client, "fake-model")  # type: ignore[arg-type]
    assert (transcript.stop_reason, transcript.answer, transcript.turns) == ("refusal", "", 1)


def test_a_loop_that_never_answers_stops_at_the_turn_limit() -> None:
    endless = [_response("tool_use", _use(f"c{i}", "line_status", "T4")) for i in range(MAX_TURNS)]
    transcript = ask("q", _Tools(), _ScriptedClient(*endless), "fake-model")  # type: ignore[arg-type]
    assert transcript.stop_reason == "max_turns"
    assert transcript.turns == MAX_TURNS and transcript.answer == ""
    assert len(transcript.steps) == MAX_TURNS


def test_the_transcript_keeps_tool_output_verbatim() -> None:
    client = _ScriptedClient(
        _response("tool_use", _use("c1", "line_status", "T4")),
        _response("end_turn", _text("done")),
    )
    transcript = ask("q", _Tools(), client, "fake-model")  # type: ignore[arg-type]
    stored = transcript.to_dict()["steps"][0]
    assert stored["output"] == json.dumps({"line": "T4", "late": 3})
    assert stored["input"] == {"line": "T4"}
