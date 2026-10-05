"""RQ1 Objective 3: the likely cause of a disruption, grounded in past alerts.

The reasons stage is given a :class:`~transit_rag.agent.situation.Situation`,
which holds what the delay feed showed and never the incident's own alert. It
may also be given past incidents retrieved from TfNSW's alert history, under the
leakage guards of ``docs/08`` §3.5. From these it names a cause group and gives a
one- or two-sentence reason citing the alerts it relied on. RQ2 compares it with
the same model given no past incidents, and with the most common cause
(``docs/08`` §3.3).

**The answer's shape is enforced, not hoped for.** The model answers through
structured output, validated against :class:`ReasonAnswer`, so a cause outside
the four groups cannot be returned. Citations are then checked against the
passages actually shown. :meth:`ReasonAnswer.unsupported_citations` names any
alert id that was not retrieved, because an invented id is the failure the
faithfulness metric exists to catch.

**Sampling cannot be fixed on the current models.** ``docs/08`` §3.5 asks for
three runs "at fixed temperature". The configured generation model takes no
sampling parameters, so repeats measure the variance of its default sampling,
and the write-up says so.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

from transit_rag.agent.situation import Situation
from transit_rag.retrieval.search import RetrievedPassage

CauseGroup = Literal["technical", "network_incident", "weather_external", "other_unknown"]

#: The groups of ``docs/08`` §3.2, in the order they are described to the model.
CAUSE_GROUPS: tuple[str, ...] = (
    "technical",
    "network_incident",
    "weather_external",
    "other_unknown",
)

SYSTEM_PROMPT = """\
You explain service disruptions on Sydney Trains' T1 and T4 lines.

You are told what the live delay feed showed on a line in the half hour before a \
disruption was reported. Sometimes you are also given past incidents from \
Transport for NSW's alert history, each published before this disruption. Decide \
the most likely cause group:

- technical: train, signal, track or power faults, and urgent repairs
- network_incident: accidents, police activity, medical emergencies, people on \
the tracks
- weather_external: weather, strikes, demonstrations, construction, holidays
- other_unknown: anything else, or when nothing indicates a cause

Use only the evidence given. If you rely on a past incident, cite its alert ids \
exactly as they are written. If no past incident bears on this one, or none is \
given, cite none and say what the delay pattern alone suggests. Explain in one or \
two sentences."""


class ReasonAnswer(BaseModel):
    """What the reasons stage returns, validated against this schema."""

    cause_group: CauseGroup
    explanation: str = Field(description="One or two sentences.")
    cited_alert_ids: list[str] = Field(
        description="Alert ids of the past incidents relied on, exactly as given; empty if none."
    )

    def unsupported_citations(self, passages: Sequence[RetrievedPassage]) -> list[str]:
        """Cited ids that were not among the passages shown."""
        shown = {alert_id for passage in passages for alert_id in _alert_ids(passage)}
        return [alert_id for alert_id in self.cited_alert_ids if alert_id not in shown]


def _alert_ids(passage: RetrievedPassage) -> list[str]:
    """The short alert ids a passage can be cited by, as its citation prints them."""
    return [
        alert_id[:8]
        for alert_id in str(passage.metadata.get("alert_ids", "")).split(",")
        if alert_id
    ]


def user_prompt(situation: Situation, passages: Sequence[RetrievedPassage] | None) -> str:
    """The question: the situation, and the past incidents if this system gets them."""
    parts = [f"Situation: {situation.describe()}"]
    if passages is None:
        parts.append("No past incidents are provided.")
    elif not passages:
        parts.append("Past incidents were searched for, and none was found.")
    else:
        parts.append("Past incidents, most similar first:")
        for passage in passages:
            ids = ", ".join(_alert_ids(passage))
            group = passage.metadata.get("cause_group", "unknown")
            parts.append(f"[alerts {ids}] cause group: {group}\n{passage.text}")
    return "\n\n".join(parts)


class LanguageModel(Protocol):
    """Anything that can answer a reasons question. Tests use a fake."""

    @property
    def model(self) -> str: ...

    def answer(self, system: str, user: str) -> ReasonAnswer: ...


class RefusedError(RuntimeError):
    """The model declined to answer; recorded, never guessed around."""


@dataclass
class ClaudeReasoner:
    """The reasons stage on Claude, through the SDK's structured output."""

    model: str
    client: Any = None
    max_tokens: int = 4000
    #: Running totals, so a run can report what it spent (``docs/04`` §3's budget risk).
    input_tokens: int = 0
    output_tokens: int = 0

    def __post_init__(self) -> None:
        if self.client is None:
            import anthropic

            self.client = anthropic.Anthropic()

    def answer(self, system: str, user: str) -> ReasonAnswer:
        response = self.client.messages.parse(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_format=ReasonAnswer,
        )
        usage = getattr(response, "usage", None)
        if usage is not None:
            self.input_tokens += int(usage.input_tokens)
            self.output_tokens += int(usage.output_tokens)
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            raise RefusedError(f"{self.model} declined: {details}")
        parsed = response.parsed_output
        if not isinstance(parsed, ReasonAnswer):
            raise RuntimeError(
                f"{self.model} returned no parseable answer ({response.stop_reason})"
            )
        return parsed


def explain(
    situation: Situation,
    model: LanguageModel,
    passages: Sequence[RetrievedPassage] | None = None,
) -> ReasonAnswer:
    """Ask ``model`` for the cause. ``passages=None`` is the no-retrieval system."""
    return model.answer(SYSTEM_PROMPT, user_prompt(situation, passages))
