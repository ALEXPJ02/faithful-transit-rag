"""``transit-reasons`` -- explain each incident's cause, and score how well it was done.

    transit-reasons --db data/delay_observations_20261005.db             # cases, retrieval, baseline
    transit-reasons --db ... --model                                    # also ask Claude, with and without retrieval
    transit-reasons --db ... --model --since 2026-09-29 --repeats 3     # only the later incidents, three runs
    transit-reasons --db ... --model --out data/reasons_runs.jsonl      # keep every answer

For each incident it builds the situation the feed showed before the incident's
first alert, never the alert itself. It retrieves past incidents under both
leakage guards: first seen before, and never the incident itself (``docs/08``
§3.5). Three systems are then scored on cause (``docs/08`` §3.3): the time-aware
most common cause, Claude without retrieval, and Claude with it. Without
``--model`` no model is called, so the cases and retrieval can be checked for
free. The alert index must match the snapshot's incidents, or the run stops and
says how to rebuild it.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

from transit_rag.agent.reasons import ClaudeReasoner, LanguageModel, ReasonAnswer, explain
from transit_rag.agent.situation import station_names
from transit_rag.config import (
    ConfigError,
    ModelConfig,
    VoyageConfig,
    chroma_persist_dir,
    configured_embedding_model,
)
from transit_rag.evaluation.reasons import (
    ReasonCase,
    build_cases,
    most_common_cause,
    retrieval_recall,
    score_reasons,
)
from transit_rag.ingestion.alerts import (
    DEFAULT_OVERRIDES,
    audit,
    load_alerts,
    load_overrides,
    sydney_time,
)
from transit_rag.prediction.collection.bundles import discover
from transit_rag.prediction.disruption.labels import load_stop_events
from transit_rag.retrieval.alert_index import ALERT_COLLECTION, check_alert_index
from transit_rag.retrieval.embeddings import VoyageEmbedder
from transit_rag.retrieval.index import open_collection
from transit_rag.retrieval.search import DEFAULT_K, RetrievedPassage, Retriever


def _retrieve(retriever: Retriever, case: ReasonCase, k: int) -> list[RetrievedPassage]:
    return retriever.search(
        case.situation.describe(),
        k,
        seen_before=case.incident.first_seen,
        exclude_incident=case.incident.incident_id,
    )


def _ask(
    model: LanguageModel, case: ReasonCase, passages: list[RetrievedPassage] | None
) -> dict[str, Any]:
    try:
        answer: ReasonAnswer = explain(case.situation, model, passages)
    except Exception as exc:  # a refusal or an API failure is recorded, not hidden
        return {"cause_group": None, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "cause_group": answer.cause_group,
        "explanation": answer.explanation,
        "cited_alert_ids": answer.cited_alert_ids,
        "unsupported_citations": answer.unsupported_citations(passages or []),
    }


def _print_scores(name: str, truth: list[str], runs: list[list[str | None]]) -> None:
    """One line per system: mean and spread over repeats. A failed answer counts as wrong."""
    scored = [score_reasons(truth, [p or "none" for p in run]) for run in runs]
    accuracy = [s.accuracy for s in scored]
    macro = [s.macro_f1 for s in scored]

    def spread(values: list[float]) -> str:
        if len(values) == 1:
            return f"{values[0]:.2f}"
        return f"{statistics.mean(values):.2f} ± {statistics.stdev(values):.2f}"

    print(
        f"  {name:<28}{spread(accuracy):>14}{spread(macro):>14}   "
        f"over {', '.join(scored[0].groups)}"
    )


def command_reasons(args: argparse.Namespace) -> int:
    alerts = load_alerts(args.db)
    _, incidents = audit(alerts, load_overrides(args.overrides))
    collection = open_collection(args.persist_dir, ALERT_COLLECTION)
    check_alert_index(collection, incidents, args.db)

    embedder = VoyageEmbedder(
        api_key=VoyageConfig.from_env().api_key, model=configured_embedding_model()
    )
    retriever = Retriever(collection, embedder)
    bundle = args.bundle or max(discover(), key=lambda path: path.name)
    cases = build_cases(
        incidents, load_stop_events(args.db), station_names(bundle), since=args.since
    )
    if not cases:
        raise ValueError(f"no incident first seen on or after {args.since}")

    truth = [case.truth for case in cases]
    retrieved = [_retrieve(retriever, case, args.k) for case in cases]
    retrieved_groups = [
        [str(p.metadata.get("cause_group")) for p in passages] for passages in retrieved
    ]
    baseline = [most_common_cause(case, incidents) for case in cases]

    print(f"Reasons from {args.db}: {len(cases)} incident(s), k={args.k}")
    for case, groups, guess in zip(cases, retrieved_groups, baseline, strict=True):
        print(
            f"\n{case.incident.incident_id}  {sydney_time(case.incident.first_seen)}  truth {case.truth}"
        )
        print(f"  situation: {case.situation.describe()}")
        print(f"  retrieved: {', '.join(groups) or 'nothing earlier'}")
        print(f"  most common so far: {guess}")

    records: list[dict[str, Any]] = []
    systems: dict[str, list[list[str | None]]] = {
        "most common cause (time-aware)": [list(baseline)]
    }
    if args.model:
        config = ModelConfig.from_env()
        import anthropic

        model = ClaudeReasoner(
            model=config.generation_model,
            client=anthropic.Anthropic(api_key=config.anthropic_api_key),
        )
        for system, with_retrieval in (("model, no retrieval", False), ("model + retrieval", True)):
            runs: list[list[str | None]] = []
            for repeat in range(args.repeats):
                run: list[str | None] = []
                for case, passages in zip(cases, retrieved, strict=True):
                    result = _ask(model, case, passages if with_retrieval else None)
                    run.append(result["cause_group"])
                    records.append(
                        {
                            "system": system,
                            "model": model.model,
                            "repeat": repeat,
                            "incident_id": case.incident.incident_id,
                            "truth": case.truth,
                            "situation": case.situation.describe(),
                            "retrieved": [p.chunk_id for p in passages] if with_retrieval else [],
                            **result,
                        }
                    )
                runs.append(run)
            systems[f"{system} ({model.model})"] = runs

    print(f"\nRetrieval recall@{args.k}: {retrieval_recall(truth, retrieved_groups):.2f}")
    print(f"\n  {'system':<28}{'accuracy':>14}{'macro-F1':>14}")
    for name, runs in systems.items():
        _print_scores(name, truth, runs)
    unsupported = sum(len(r.get("unsupported_citations", [])) for r in records)
    errors = sum(1 for r in records if r.get("error"))
    if args.model:
        print(
            f"\n  answers: {len(records)}, failed: {errors}, citations of unshown alerts: {unsupported}"
        )
        print(f"  tokens: {model.input_tokens:,} in, {model.output_tokens:,} out")

    if args.out is not None and records:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
        print(f"wrote {len(records)} answers to {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="transit-reasons",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--db", type=Path, required=True, help="a collection snapshot (read-only)")
    parser.add_argument("--overrides", type=Path, default=DEFAULT_OVERRIDES)
    parser.add_argument("--persist-dir", type=Path, default=chroma_persist_dir())
    parser.add_argument("--bundle", type=Path, default=None, help="for station names")
    parser.add_argument(
        "--since", default=None, help="only incidents first seen from this Sydney date"
    )
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--model", action="store_true", help="ask the configured Claude model")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--out", type=Path, default=None, help="write every answer as JSONL")
    parser.set_defaults(handler=command_reasons)
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
