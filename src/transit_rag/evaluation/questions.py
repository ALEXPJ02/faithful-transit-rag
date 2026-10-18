"""``transit-questions`` -- the fixed question set the orchestrator is scored on.

    transit-questions write --db data/evaluation_20261018.db --out data/questions_test.csv
    transit-questions ask --questions data/questions_test.csv --out-dir data/transcripts_test \\
        --db data/evaluation_20261018.db --detector models/detector_20261018.joblib \\
        --delay-model models/delay_model_20261018.joblib

``docs/14`` §7 scores the orchestrator on a fixed set of questions asked on the test
dates. RQ1 Objective 1's tool-faithfulness, and the agent's faithfulness and citation
coverage, come from judging those answers (``docs/15``). The set is fixed by rules
written before the test dates existed, so no question can be picked for how well it
was answered:

- **When:** each test date and line at 08:30 and 17:30 Sydney time, the two peaks. Also
  each incident on a test date, a quarter of an hour after it was first seen, while its
  alert is in the feed.
- **What:** three fixed questions, assigned in turn through the moments in time order,
  so each is asked about as often as the others.

The test dates are the detection split's: ``time_based_split`` over the detection
table, exactly as ``transit-detect`` draws it. The agent's risk tool is a detector
trained on the other dates, so it never answers for a date it learned from.

``ask`` builds the tools once and moves them to each question's moment. It writes one
transcript per question, and refuses a folder that already holds some, because a scored
run must not mix with another.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime, time, timedelta
from pathlib import Path

from transit_rag.agent.cli import add_toolbox_arguments, build_toolbox
from transit_rag.agent.loop import ask
from transit_rag.config import ConfigError, ModelConfig
from transit_rag.ingestion.alerts import DEFAULT_LINES, DEFAULT_OVERRIDES, Incident, sydney_time
from transit_rag.prediction.disruption.detect_cli import build
from transit_rag.prediction.disruption.labels import LabelRule
from transit_rag.prediction.features.quality import time_based_split
from transit_rag.realtime.parsing import SYDNEY, service_day

TEMPLATES: tuple[str, ...] = (
    "Is the {line} disrupted right now, and why?",
    "How late are {line} trains running, and how sure are you?",
    "Should I expect delays on the {line} in the next half hour?",
)

#: The morning and evening peaks, in Sydney time.
PEAKS = (time(8, 30), time(17, 30))

#: Long enough after an incident's first sighting for its alert to be in the feed:
#: the agent's tools see an alert only once a poll before the moment carried it.
AFTER_FIRST_SEEN = timedelta(minutes=15)

COLUMNS = ("at", "line", "question", "why")


@dataclass(frozen=True)
class Question:
    """One question, the moment it is asked as of, and the rule that chose it."""

    at: datetime
    line: str
    question: str
    why: str


def question_set(
    test_dates: Iterable[str], incidents: Sequence[Incident], lines: Sequence[str] = DEFAULT_LINES
) -> list[Question]:
    """The scored questions for ``test_dates``, by the rules in the module docstring."""
    dates = sorted(set(test_dates))
    moments: list[tuple[datetime, str, str]] = [
        (datetime.combine(date.fromisoformat(day), clock, tzinfo=SYDNEY), line, "peak")
        for day in dates
        for clock in PEAKS
        for line in lines
    ]
    for incident in incidents:
        # The service date turns over at 03:00, so a 01:30 incident is the day before's.
        if service_day(incident.first_seen.isoformat()) in dates:
            at = (incident.first_seen + AFTER_FIRST_SEEN).astimezone(SYDNEY)
            moments += [
                (at, line, f"incident {incident.incident_id}")
                for line in incident.lines
                if line in lines
            ]
    moments.sort(key=lambda moment: (moment[0], moment[1]))
    return [
        Question(at, line, TEMPLATES[index % len(TEMPLATES)].format(line=line), why)
        for index, (at, line, why) in enumerate(moments)
    ]


def write_questions(questions: Sequence[Question], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        for item in questions:
            writer.writerow([item.at.isoformat(), item.line, item.question, item.why])


def read_questions(path: Path) -> list[Question]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    questions = []
    for number, row in enumerate(rows, start=2):
        at = datetime.fromisoformat(row["at"])
        if at.tzinfo is None:
            raise ValueError(f"{path}:{number}: the moment {row['at']!r} has no UTC offset")
        questions.append(Question(at, row["line"], row["question"], row["why"]))
    return questions


def command_write(args: argparse.Namespace) -> int:
    if args.out.exists():
        raise ValueError(f"{args.out} already exists; the question set is written once")
    table, incidents = build(args.db, args.overrides, LabelRule())
    test_dates = sorted(time_based_split(table).test["service_date"].unique())
    questions = question_set(test_dates, incidents)
    write_questions(questions, args.out)
    reasons = Counter("peak" if item.why == "peak" else "incident" for item in questions)
    print(f"Test dates: {', '.join(test_dates)}")
    print(
        f"wrote {len(questions)} question(s) to {args.out}: {reasons['peak']} at the peaks, "
        f"{reasons['incident']} at incidents"
    )
    return 0


def command_ask(args: argparse.Namespace) -> int:
    questions = read_questions(args.questions)
    if not questions:
        raise ValueError(f"{args.questions} holds no questions")
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise ValueError(f"{args.out_dir} already holds transcripts; a scored run is written once")
    tools = build_toolbox(args, at=questions[0].at)
    config = ModelConfig.from_env()
    import anthropic

    client = anthropic.Anthropic(api_key=config.anthropic_api_key)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tokens = [0, 0]
    for index, item in enumerate(questions, start=1):
        transcript = ask(item.question, replace(tools, at=item.at), client, config.generation_model)
        # k is frozen at the cut-off (docs/13 §7), and a transcript must show which it used.
        record = {**transcript.to_dict(), "why": item.why, "k": tools.k}
        name = f"{index:03d}_{item.line}_{item.at.astimezone(SYDNEY):%Y%m%d_%H%M}.json"
        (args.out_dir / name).write_text(json.dumps(record, indent=2), encoding="utf-8")
        tokens[0] += transcript.input_tokens
        tokens[1] += transcript.output_tokens
        print(
            f"[{index}/{len(questions)}] {item.line} {sydney_time(item.at)} ({item.why}): "
            f"{len(transcript.steps)} tool call(s), stop {transcript.stop_reason}"
        )
    print(
        f"\n{len(questions)} transcript(s) in {args.out_dir}; tokens {tokens[0]:,} in / {tokens[1]:,} out"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="transit-questions",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    write = sub.add_parser("write", help="write the question set for the test dates")
    write.add_argument("--db", type=Path, required=True, help="the frozen evaluation snapshot")
    write.add_argument("--overrides", type=Path, default=DEFAULT_OVERRIDES)
    write.add_argument("--out", type=Path, required=True)
    write.set_defaults(handler=command_write)

    run = sub.add_parser("ask", help="ask every question, one transcript each")
    run.add_argument("--questions", type=Path, required=True)
    run.add_argument("--out-dir", type=Path, required=True)
    add_toolbox_arguments(run, moment=False)
    run.set_defaults(handler=command_ask)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result: int = args.handler(args)
        return result
    except (FileNotFoundError, ValueError, ConfigError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
