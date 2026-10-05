"""The orchestrator: a hand-rolled tool-use loop over the three workflow stages.

``docs/01`` §1 and ``CLAUDE.md``'s settled decision: one Claude loop calls each
stage as a tool and writes the answer. It is written by hand rather than with a
framework, so every request, tool call and result is visible and recorded.

**What the evaluation reads is recorded here.** A :class:`Transcript` keeps the
question, the moment, every tool call with its exact output, and the answer.
Tool-faithfulness (``docs/08`` §2, O1) compares the answer's claims about live
conditions with those outputs, so the outputs are stored verbatim, as the model
saw them.

**The prompt states the checked rules from the first version**, never bolted on
(``docs/08`` §7). Live conditions only as a tool reported them. A probability as
the estimate the tool returned, unnarrowed. A cause only with the alert ids it
rests on, or else said to be unknown.

The loop follows the SDK's documented shape. The assistant's whole ``content`` is
appended each turn, which keeps any thinking blocks intact. All tool results for
a turn go back in one user message. A refusal or a truncated turn stops the loop
and is recorded, never retried blindly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

from transit_rag.agent.tools import TOOL_SPECS, ToolBox
from transit_rag.realtime.parsing import SYDNEY

#: A question needs at most one call per stage per line, and a follow-up or two.
MAX_TURNS = 8

SYSTEM_TEMPLATE = """\
You answer riders' questions about Sydney Trains' T1 and T4 lines. It is {as_of} in \
Sydney.

Use the tools for anything about current conditions, the risk of disruption, or \
causes. These rules are checked:
- State live conditions only as a tool reported them.
- A disruption probability is a model estimate. Give the number the tool returned, \
say it is an estimate, and do not round it into certainty.
- Give a cause only with the alert ids it rests on, whether a current alert or a past \
incident. If nothing supports a cause, say the cause is not known.
- If a tool returned an error, say what could not be checked.

Answer in at most five sentences."""


@dataclass(frozen=True)
class Step:
    """One tool call, and exactly what came back."""

    tool: str
    input: dict[str, Any]
    output: str
    is_error: bool


@dataclass
class Transcript:
    """Everything the evaluation needs about one answered question."""

    question: str
    as_of: str
    model: str
    steps: list[Step] = field(default_factory=list)
    answer: str = ""
    stop_reason: str = ""
    turns: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def system_prompt(at: datetime) -> str:
    local = at.astimezone(SYDNEY)
    return SYSTEM_TEMPLATE.format(as_of=f"{local:%H:%M} on {local:%A} {local.day} {local:%B %Y}")


def _text(content: Any) -> str:
    return "\n".join(block.text for block in content if getattr(block, "type", None) == "text")


def ask(
    question: str,
    tools: ToolBox,
    client: Any,
    model: str,
    *,
    max_turns: int = MAX_TURNS,
    max_tokens: int = 16000,
) -> Transcript:
    """Answer ``question`` as of ``tools.at``, calling the tools as the model asks."""
    transcript = Transcript(question=question, as_of=tools.at.isoformat(), model=model)
    messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
    system = system_prompt(tools.at)

    for _ in range(max_turns):
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            tools=TOOL_SPECS,
            messages=messages,
        )
        transcript.turns += 1
        usage = getattr(response, "usage", None)
        if usage is not None:
            transcript.input_tokens += int(usage.input_tokens)
            transcript.output_tokens += int(usage.output_tokens)
        transcript.stop_reason = str(response.stop_reason)

        if response.stop_reason != "tool_use":
            # end_turn is the answer. A refusal or max_tokens is recorded as it
            # stands, so the evaluation can count it rather than see a retry.
            transcript.answer = _text(response.content)
            return transcript

        messages.append({"role": "assistant", "content": response.content})
        results = []
        for block in response.content:
            if getattr(block, "type", None) != "tool_use":
                continue
            output, is_error = tools.call(block.name, dict(block.input))
            transcript.steps.append(Step(block.name, dict(block.input), output, is_error))
            result: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": output,
            }
            if is_error:
                result["is_error"] = True
            results.append(result)
        messages.append({"role": "user", "content": results})

    transcript.stop_reason = "max_turns"
    return transcript
