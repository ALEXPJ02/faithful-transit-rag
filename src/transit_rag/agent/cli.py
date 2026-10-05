"""``transit-ask`` -- ask the orchestrator a question, as of a moment in a snapshot.

    transit-ask --db data/delay_observations_20261005.db --at 2026-10-01T08:45+10:00 \\
        "Is the T1 disrupted right now, and why?"
    transit-ask ... --detector models/detector_20261005.joblib --out data/transcript.json

The orchestrator answers as of ``--at``. It sees what the snapshot recorded before
then and nothing after (``docs/14``). ``--at`` must carry its UTC offset, because a
naive time is ten or eleven hours ambiguous in Sydney. Without ``--detector`` the
disruption tool reports that no detector is loaded, and the answer has to say so.
Every tool call is printed with its result, and ``--out`` keeps the whole
transcript for scoring.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from transit_rag.agent.delays import DelayModel
from transit_rag.agent.feed import SnapshotFeed
from transit_rag.agent.loop import ask
from transit_rag.agent.tools import ToolBox, detection_rows
from transit_rag.config import (
    ConfigError,
    ModelConfig,
    VoyageConfig,
    chroma_persist_dir,
    configured_embedding_model,
)
from transit_rag.ingestion.alerts import DEFAULT_OVERRIDES, audit, load_overrides
from transit_rag.prediction.collection.bundles import discover
from transit_rag.prediction.disruption.models import load_detector
from transit_rag.retrieval.alert_index import ALERT_COLLECTION, check_alert_index
from transit_rag.retrieval.embeddings import VoyageEmbedder
from transit_rag.retrieval.index import open_collection
from transit_rag.retrieval.search import DEFAULT_K, Retriever


def timetable_eras(given: Path | None, discovered: list[Path]) -> list[Path]:
    """The eras the delay tool may look trains up in: every archived one, and ``--bundle``.

    A bundle passed for station names is a timetable too. Leaving it out made a run
    whose only era was that file unable to find any train's next stop (found by
    Cursor Bugbot on #23).
    """
    eras = list(discovered)
    if given is not None and given.resolve() not in {era.resolve() for era in eras}:
        eras.append(given)
    return eras


def _moment(text: str) -> datetime:
    moment = datetime.fromisoformat(text)
    if moment.tzinfo is None:
        raise ValueError(
            f"--at {text!r} has no UTC offset; write it as e.g. 2026-10-01T08:45+10:00"
        )
    return moment


def build_toolbox(args: argparse.Namespace) -> ToolBox:
    """The tools for one snapshot and one moment, with whichever models were given.

    Shared with ``transit-mcp``, so the agent and an MCP client get the same tools.
    """
    at = _moment(args.at)
    bundle = args.bundle or max(discover(), key=lambda path: path.name)
    feed = SnapshotFeed.open(args.db, bundle)
    _, incidents = audit(feed.alerts, load_overrides(args.overrides))

    collection = open_collection(args.persist_dir, ALERT_COLLECTION)
    check_alert_index(collection, incidents, args.db)
    embedder = VoyageEmbedder(
        api_key=VoyageConfig.from_env().api_key, model=configured_embedding_model()
    )
    tools = ToolBox(feed=feed, at=at, retriever=Retriever(collection, embedder), k=args.k)
    if args.detector is not None:
        tools.detector, tools.detector_provenance = load_detector(args.detector)
        tools.rows = detection_rows(feed, incidents)
    if args.delay_model is not None:
        tools.delay_model = DelayModel.load(args.delay_model)
        tools.bundles = timetable_eras(args.bundle, discover())
    return tools


def command_ask(args: argparse.Namespace) -> int:
    tools = build_toolbox(args)
    config = ModelConfig.from_env()
    import anthropic

    client = anthropic.Anthropic(api_key=config.anthropic_api_key)
    transcript = ask(args.question, tools, client, config.generation_model)

    print(f"Q ({transcript.as_of}): {transcript.question}\n")
    for step in transcript.steps:
        flag = " [error]" if step.is_error else ""
        print(f"  -> {step.tool}({json.dumps(step.input)}){flag}")
        print(f"     {step.output[:300]}{'…' if len(step.output) > 300 else ''}")
    print(f"\nA: {transcript.answer}")
    print(
        f"\n({transcript.model}, {transcript.turns} turn(s), stop: {transcript.stop_reason}, "
        f"tokens {transcript.input_tokens:,} in / {transcript.output_tokens:,} out)"
    )
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(transcript.to_dict(), indent=2), encoding="utf-8")
        print(f"wrote the transcript to {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="transit-ask",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("question")
    add_toolbox_arguments(parser)
    parser.add_argument("--out", type=Path, default=None, help="write the transcript as JSON")
    parser.set_defaults(handler=command_ask)
    return parser


def add_toolbox_arguments(parser: argparse.ArgumentParser) -> None:
    """What :func:`build_toolbox` reads: the snapshot, the moment, and the models."""
    parser.add_argument("--db", type=Path, required=True, help="a collection snapshot (read-only)")
    parser.add_argument("--at", required=True, help="the moment, with its UTC offset")
    parser.add_argument("--detector", type=Path, default=None, help="from transit-detect --save")
    parser.add_argument(
        "--delay-model", type=Path, default=None, help="transit-train's artefact, with its interval"
    )
    parser.add_argument("--bundle", type=Path, default=None, help="for station names")
    parser.add_argument("--overrides", type=Path, default=DEFAULT_OVERRIDES)
    parser.add_argument("--persist-dir", type=Path, default=chroma_persist_dir())
    parser.add_argument("--k", type=int, default=DEFAULT_K)


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
