"""The faithfulness judge: is each statement supported by what the system was shown?

``docs/08`` §3.4 defines two metrics. **Faithfulness rate** is the share of
statements supported by a retrieved alert or a tool output (Papageorgiou et al.,
2025). **Citation coverage** is the share of statements citing at least one alert
that was actually shown (Huang et al., 2026).

**Per statement, never per answer.** An answer is split into sentences, and each is
judged on its own. One invented clause inside an otherwise grounded answer is
exactly what an answer-level verdict would forgive. The split is deterministic, so
the same answer always yields the same statements.

**The judge sees exactly the system's evidence, and never the ground truth.** For a
reasons answer that is the verbatim prompt it was given. For an agent answer it is
the moment the agent was told it is, the question, and every tool output, verbatim
from the transcript. Neither system prompt holds a fact about the line, the delays or
any incident: one holds instructions and cause-group definitions, the other
instructions and that moment. The judge asks *does the evidence support this?*, not *is this
right?*: a faithful wrong answer and an unfaithful right one are both possible, and
the evaluation must tell them apart (``docs/01`` §2).

**A cheaper model, validated before it is trusted.** The judge is
``ANTHROPIC_JUDGE_MODEL`` (``claude-haiku-4-5``). An unvalidated judge is an
unvalidated instrument, so ``kappa.py`` checks it against the author's own labels on
a blind, stratified 20% sample. A κ below 0.6 sends the prompt back for revision
before any full run (``docs/08`` §3.5).

**Temperature 0, where the model still honours it.** The SDK's 1.x signatures dropped
sampling parameters, but the API did not. Haiku 4.5 and the 4.5/4.6 line accept
them, so the judge sends ``temperature: 0`` through ``extra_body``, as the SDK's
upgrade guide prescribes for code that depends on it. Newer models reject any
sampling parameter with a 400, so for them it is omitted, and
:attr:`ClaudeJudge.deterministic` says so.

**Citation coverage needs no judge.** Whether a statement names a shown alert id is a
string check against the ids the evidence actually contained.

**Every verdict names the rules it applied.** :func:`prompt_fingerprint` hashes the
judge's instructions, message and answer schema, and each judged row carries it with
the judge model. A κ validates one fingerprint, and a rate computed under another is
not covered by it.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Protocol

from pydantic import BaseModel, Field

from transit_rag.ingestion.alerts import sydney_moment

#: A sentence ends at terminal punctuation, perhaps closed by a bracket or quote,
#: followed by space and a capital, digit, quote or opening bracket. Decimals
#: ("15.0") and ids do not end one, and nor do the abbreviations an answer about
#: Sydney's stations is likely to use ("St. Leonards", "e.g. Central").
_SENTENCE_END = re.compile(
    r"(?:(?<=[.!?])|(?<=[.!?][)\]\"']))"
    r"(?<!\be\.g\.)(?<!\bi\.e\.)(?<!\bSt\.)(?<!\bMt\.)(?<!\bvs\.)(?<![Aa]pprox\.)"
    r"\s+(?=[A-Z0-9\"'(\[])"
)

#: How an alert is cited everywhere in this system: the first 8 hex digits.
_ALERT_ID = re.compile(r"\b[0-9a-f]{8}\b")

#: Revised twice on 2026-10-05, on development verdicts and before any author label
#: existed (``docs/15`` §3). The first version misread maxima ("up to 9 minutes")
#: and applied its outside-knowledge rule to some station geography but not the
#: rest. The second over-applied its new cause rule to statements that no cause is
#: known, and had no rule for a cause carried by an alert or a past incident, which
#: is what retrieval exists to supply.
JUDGE_SYSTEM = """\
You check whether one statement from a train-disruption assistant is supported by \
the evidence the assistant was given.

Supported means every factual claim in the statement is in the evidence or follows \
directly from it. Numbers, times, places, causes and alert ids must match.
- Read each claim in its plain sense, as a careful reader would. A paraphrase that \
keeps the evidence's meaning is supported: "up to 9 minutes late", "no more than 9 \
minutes late" and "the most delayed was 9 minutes behind" say the same thing.
- A hedged claim ("likely", "may", "suggests") is supported only if the evidence \
supports it at the strength claimed.
- Delay figures show how many services were late, by how much, and where. They do \
not show why. Attributing the delays to a cause that nothing in the evidence links \
them to, such as a signal fault, congestion or police activity, is unsupported \
however it is hedged. Saying that nothing points to a cause attributes none.
- An alert or a past incident in the evidence can link a cause to the delays. The \
statement must say what connects them, such as the same stations, that connection \
must be in the evidence, and a past incident's cause must be offered as possible, \
not as certain.
- A statement that something is unknown or cannot be determined is supported when \
the evidence does not determine it.
- A statement with no factual claim, such as framing or a restated question, is \
supported.
- A claim the evidence does not contain is unsupported, even if it may be true. That \
includes facts about the network, such as which line or area a station is on, unless \
the evidence states them.

Judge only against the evidence. Use no outside knowledge."""

#: The message each verdict is asked with.
JUDGE_MESSAGE = "<evidence>\n{evidence}\n</evidence>\n\n<statement>\n{statement}\n</statement>"


def split_statements(text: str) -> list[str]:
    """An answer's sentences, deterministically. A line break ends one too, so each
    item of a list is judged on its own."""
    return [
        part.strip()
        for line in text.splitlines()
        for part in _SENTENCE_END.split(line.strip())
        if part.strip()
    ]


def cited_alert_ids(text: str) -> set[str]:
    """Every 8-hex-digit alert id named in ``text``."""
    return set(_ALERT_ID.findall(text))


class JudgeVerdict(BaseModel):
    """The judge's answer for one statement, validated against this schema."""

    supported: bool
    rationale: str = Field(
        description="One sentence: what in the evidence supports it, or what is missing."
    )


def prompt_fingerprint() -> str:
    """The first 12 hex digits of a sha256 over everything that tells the judge what to do."""
    schema = json.dumps(JudgeVerdict.model_json_schema(), sort_keys=True)
    payload = "\n\n".join([JUDGE_SYSTEM, JUDGE_MESSAGE, schema])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


class Judge(Protocol):
    """Anything that can judge a statement against evidence. Tests use a fake."""

    @property
    def model(self) -> str: ...

    def verdict(self, evidence: str, statement: str) -> JudgeVerdict: ...


#: Models that still honour a sampling parameter. Claude Opus 4.7 and later, and
#: Claude Sonnet 5 and later, reject one.
ACCEPTS_TEMPERATURE: tuple[str, ...] = (
    "claude-haiku-4-5",
    "claude-sonnet-4-6",
    "claude-opus-4-6",
    "claude-sonnet-4-5",
    "claude-opus-4-5",
)


@dataclass
class ClaudeJudge:
    """The judge on Claude, through the SDK's structured output, at temperature 0 if allowed."""

    model: str
    client: Any = None
    max_tokens: int = 1024
    input_tokens: int = 0
    output_tokens: int = 0

    def __post_init__(self) -> None:
        if self.client is None:
            import anthropic

            self.client = anthropic.Anthropic()

    @property
    def deterministic(self) -> bool:
        """Whether the judge runs at temperature 0, which depends on the model."""
        return self.model.startswith(ACCEPTS_TEMPERATURE)

    def verdict(self, evidence: str, statement: str) -> JudgeVerdict:
        sampling = {"extra_body": {"temperature": 0}} if self.deterministic else {}
        response = self.client.messages.parse(
            model=self.model,
            max_tokens=self.max_tokens,
            system=JUDGE_SYSTEM,
            messages=[
                {
                    "role": "user",
                    "content": JUDGE_MESSAGE.format(evidence=evidence, statement=statement),
                }
            ],
            output_format=JudgeVerdict,
            **sampling,
        )
        usage = getattr(response, "usage", None)
        if usage is not None:
            self.input_tokens += int(usage.input_tokens)
            self.output_tokens += int(usage.output_tokens)
        parsed = response.parsed_output
        if response.stop_reason == "refusal" or not isinstance(parsed, JudgeVerdict):
            raise RuntimeError(f"{self.model} gave no verdict ({response.stop_reason})")
        return parsed


@dataclass(frozen=True)
class JudgedStatement:
    """One statement, the judge's verdict on it, and whether it cites a shown alert."""

    statement_id: str
    answer_id: str
    system: str
    statement: str
    #: ``None`` when the judge failed: excluded from the rate, and counted.
    supported: bool | None
    rationale: str
    cites_shown_alert: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Answer:
    """One answer to judge: its text, the evidence it was given, and the ids shown."""

    answer_id: str
    system: str
    text: str
    evidence: str
    shown_ids: frozenset[str]
    #: Ids the answer cites outside its text. A reasons answer lists them in a field.
    cited_outside_text: frozenset[str] = frozenset()


def judge_answer(answer: Answer, judge: Judge) -> list[JudgedStatement]:
    """Split ``answer`` and judge each statement against its evidence."""
    judged = []
    for index, statement in enumerate(split_statements(answer.text)):
        cited = cited_alert_ids(statement) | answer.cited_outside_text
        try:
            verdict = judge.verdict(answer.evidence, statement)
            supported: bool | None = verdict.supported
            rationale = verdict.rationale
        except Exception as exc:  # recorded and counted, never guessed
            supported, rationale = None, f"judge failed: {type(exc).__name__}: {exc}"
        judged.append(
            JudgedStatement(
                statement_id=f"{answer.answer_id}#{index}",
                answer_id=answer.answer_id,
                system=answer.system,
                statement=statement,
                supported=supported,
                rationale=rationale,
                cites_shown_alert=bool(cited & answer.shown_ids),
            )
        )
    return judged


def faithfulness_rate(statements: Sequence[JudgedStatement]) -> float:
    """Supported statements over judged ones. A failed verdict is in neither."""
    judged = [s for s in statements if s.supported is not None]
    return sum(bool(s.supported) for s in judged) / len(judged) if judged else float("nan")


def citation_coverage(statements: Sequence[JudgedStatement]) -> float:
    """Statements citing at least one alert that was shown, over all statements."""
    if not statements:
        return float("nan")
    return sum(s.cites_shown_alert for s in statements) / len(statements)


def reasons_answers(records: Iterable[dict[str, Any]]) -> list[Answer]:
    """Reasons-stage answers from ``transit-reasons --out``, with their verbatim prompts.

    The ids shown are read from the prompt's ``[alerts …]`` headers, because that is
    how the model saw them. A reasons answer lists its citations in a field, so they
    apply to each sentence of its explanation. A failed answer has nothing to judge,
    and ``transit-reasons`` has already counted it.
    """
    answers = []
    for record in records:
        if not record.get("explanation"):
            continue
        if not record.get("prompt"):
            # A prompt rebuilt now could differ from the one sent, if the index has changed.
            raise ValueError(
                f"the {record['system']!r} answer for {record['incident_id']} has no prompt; "
                "re-run transit-reasons, which records each prompt verbatim"
            )
        shown = {
            alert_id
            for header in re.findall(r"\[alerts ([0-9a-f, ]+)\]", record["prompt"])
            for alert_id in re.split(r",\s*", header)
            if alert_id
        }
        answers.append(
            Answer(
                answer_id=f"{record['system']}|{record['incident_id']}|{record['repeat']}",
                system=record["system"],
                text=record["explanation"],
                evidence=record["prompt"],
                shown_ids=frozenset(shown),
                cited_outside_text=frozenset(record.get("cited_alert_ids", [])),
            )
        )
    return answers


def transcript_answer(transcript: dict[str, Any], name: str) -> Answer:
    """An orchestrator answer, judged against the moment, the question and every tool output."""
    shown: set[str] = set()
    for step in transcript["steps"]:
        try:
            output = json.loads(step["output"])
        except json.JSONDecodeError:
            continue
        shown |= {alert["alert_id"] for alert in output.get("alerts_in_feed", [])}
        for incident in output.get("incidents", []):
            shown |= set(incident.get("alert_ids", []))
    outputs = "\n\n".join(
        f"[{step['tool']}({json.dumps(step['input'])})]\n{step['output']}"
        for step in transcript["steps"]
    )
    # Worded as the system prompt words it, so "this morning" can be checked.
    moment = sydney_moment(datetime.fromisoformat(transcript["as_of"]))
    return Answer(
        answer_id=f"agent|{name}",
        system="agent",
        text=transcript["answer"],
        evidence=f"It is {moment} in Sydney.\nQuestion: {transcript['question']}\n\n{outputs}",
        shown_ids=frozenset(shown),
    )
