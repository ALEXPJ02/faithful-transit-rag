"""Tests for the reasons stage (``agent/reasons.py``).

No test calls a model. The prompt and the answer's contract are this package's;
the model is behind :class:`~transit_rag.agent.reasons.LanguageModel`, and the
real client is checked only for how it is called.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from transit_rag.agent.reasons import (
    SYSTEM_PROMPT,
    ClaudeReasoner,
    ReasonAnswer,
    RefusedError,
    explain,
    user_prompt,
)
from transit_rag.agent.situation import Situation
from transit_rag.retrieval.search import RetrievedPassage

AT = datetime(2026, 9, 25, 7, 19, tzinfo=UTC)
SITUATION = Situation(("T1",), AT, timedelta(minutes=30), 63, 4, 420, (("Lindfield", 2),))


def _passage(alert_ids: str, group: str, text: str) -> RetrievedPassage:
    return RetrievedPassage(
        chunk_id="inc-" + alert_ids[:6],
        text=text,
        document_key="tfnsw_alerts",
        document_title="TfNSW service alerts",
        page=0,
        citation="TfNSW alerts",
        score=0.8,
        rank=1,
        locator="alerts " + alert_ids,
        metadata={"alert_ids": alert_ids, "cause_group": group},
    )


PASSAGES = [
    _passage(
        "0f013e51-aaaa,ce4a88bd-bbbb",
        "network_incident",
        "T1 and T4, first seen 2026-09-21 16:11. Cause: police activity.",
    ),
    _passage(
        "56d10520-cccc", "technical", "T4, first seen 2026-09-22 10:23. Cause: technical problem."
    ),
]


class TestPrompt:
    def test_past_incidents_are_shown_with_their_ids_and_group(self) -> None:
        text = user_prompt(SITUATION, PASSAGES)
        assert text.startswith("Situation: T1, 17:19 on Friday 25 September 2026")
        assert "[alerts 0f013e51, ce4a88bd] cause group: network_incident" in text
        assert "Cause: technical problem." in text

    def test_the_no_retrieval_system_is_told_so(self) -> None:
        assert "No past incidents are provided." in user_prompt(SITUATION, None)

    def test_an_empty_retrieval_is_not_the_same_as_none_asked(self) -> None:
        assert "none was found" in user_prompt(SITUATION, [])

    def test_the_four_groups_are_described_to_the_model(self) -> None:
        for group in ("technical", "network_incident", "weather_external", "other_unknown"):
            assert f"- {group}:" in SYSTEM_PROMPT


class TestAnswer:
    def test_a_cause_outside_the_four_groups_cannot_be_returned(self) -> None:
        with pytest.raises(ValidationError):
            ReasonAnswer(cause_group="signal_failure", explanation="x", cited_alert_ids=[])  # type: ignore[arg-type]

    def test_citations_of_alerts_never_shown_are_named(self) -> None:
        answer = ReasonAnswer(
            cause_group="technical",
            explanation="Like the 22 Sep repairs.",
            cited_alert_ids=["56d10520", "deadbeef"],
        )
        assert answer.unsupported_citations(PASSAGES) == ["deadbeef"]
        assert answer.unsupported_citations([]) == ["56d10520", "deadbeef"]


class _Recording:
    """A language model that records what it was asked."""

    model = "fake-model"

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def answer(self, system: str, user: str) -> ReasonAnswer:
        self.calls.append((system, user))
        return ReasonAnswer(cause_group="technical", explanation="x", cited_alert_ids=[])


def test_explain_sends_the_system_prompt_and_the_question() -> None:
    model = _Recording()
    explain(SITUATION, model, PASSAGES)
    [(system, user)] = model.calls
    assert system == SYSTEM_PROMPT
    assert user == user_prompt(SITUATION, PASSAGES)


class _FakeMessages:
    def __init__(self, response: Any) -> None:
        self.response = response
        self.kwargs: dict[str, Any] = {}

    def parse(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        return self.response


def _client(stop_reason: str, parsed: ReasonAnswer | None) -> Any:
    usage = SimpleNamespace(input_tokens=120, output_tokens=40)
    response = SimpleNamespace(
        stop_reason=stop_reason, parsed_output=parsed, stop_details=None, usage=usage
    )
    return SimpleNamespace(messages=_FakeMessages(response))


class TestClaudeReasoner:
    def test_it_asks_for_the_answer_schema_with_the_configured_model(self) -> None:
        expected = ReasonAnswer(cause_group="network_incident", explanation="x", cited_alert_ids=[])
        client = _client("end_turn", expected)
        reasoner = ClaudeReasoner(model="claude-sonnet-5", client=client)
        assert reasoner.answer("system", "user") == expected
        kwargs = client.messages.kwargs
        assert kwargs["model"] == "claude-sonnet-5"
        assert kwargs["output_format"] is ReasonAnswer
        assert kwargs["system"] == "system"
        assert kwargs["messages"] == [{"role": "user", "content": "user"}]
        assert "temperature" not in kwargs  # the current models refuse sampling parameters

    def test_it_keeps_a_running_total_of_tokens(self) -> None:
        expected = ReasonAnswer(cause_group="technical", explanation="x", cited_alert_ids=[])
        reasoner = ClaudeReasoner(model="m", client=_client("end_turn", expected))
        reasoner.answer("s", "u")
        reasoner.answer("s", "u")
        assert (reasoner.input_tokens, reasoner.output_tokens) == (240, 80)

    def test_a_refusal_is_raised_not_guessed_around(self) -> None:
        reasoner = ClaudeReasoner(model="m", client=_client("refusal", None))
        with pytest.raises(RefusedError):
            reasoner.answer("system", "user")

    def test_no_parseable_answer_is_an_error(self) -> None:
        reasoner = ClaudeReasoner(model="m", client=_client("max_tokens", None))
        with pytest.raises(RuntimeError, match="no parseable answer"):
            reasoner.answer("system", "user")
