"""``transit-judge`` -- faithfulness and citation coverage, and the judge's own validation.

    transit-judge judge  --reasons data/reasons_runs.jsonl --transcripts data/transcripts \\
                         --out data/judged.jsonl                       # calls the judge model
    transit-judge report --judged data/judged.jsonl                    # rates per system
    transit-judge sample --judged data/judged.jsonl --out data/judge_sample.csv   # blind, to label
    transit-judge sample --judged data/judged.jsonl --exclude data/judge_sample.csv \\
                         --seed 1 --out data/judge_sample_2.csv   # a second round, none repeated
    transit-judge kappa  --judged data/judged.jsonl --labels data/judge_sample.csv
    transit-judge rescore --judged data/judged.jsonl --labels data/judge_sample.csv \\
                          --model claude-opus-4-6 --out data/rescored.jsonl   # the same sample, again

The order matters (``docs/08`` §3.5). Judge a development set, have the author label
the blind sample, and check κ. Only then trust ``report``. Below κ = 0.6 the judge is
revised and the sample re-scored, before the full run and never after seeing its
results. ``rescore`` re-judges exactly the labelled statements, with their evidence
verbatim, under the current rubric and the model given. ``kappa`` then reads its output.

A later round passes each earlier sheet to ``--exclude``. No statement is drawn twice,
so its κ is measured on statements that played no part in choosing the judge.

Every judged row records the judge model and :func:`prompt_fingerprint`. ``report``
and ``kappa`` print them, and refuse a file that mixes two, so a κ can be matched to
the rates it vouches for.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

from transit_rag.config import ConfigError, ModelConfig
from transit_rag.evaluation.judge import (
    Answer,
    ClaudeJudge,
    JudgedStatement,
    citation_coverage,
    faithfulness_rate,
    judge_answer,
    judge_statement,
    prompt_fingerprint,
    reasons_answers,
    transcript_answer,
)
from transit_rag.evaluation.kappa import (
    DEFAULT_SHARE,
    KAPPA_FLOOR,
    cohens_kappa,
    read_labels,
    read_statement_ids,
    stratified_sample,
    write_blind_sheet,
)


def _answers(reasons: Path | None, transcripts: Path | None) -> list[Answer]:
    answers: list[Answer] = []
    if reasons is not None:
        with reasons.open(encoding="utf-8") as handle:
            answers += reasons_answers(json.loads(line) for line in handle if line.strip())
    if transcripts is not None:
        for path in sorted(transcripts.glob("*.json")):
            answers.append(
                transcript_answer(json.loads(path.read_text(encoding="utf-8")), path.stem)
            )
    return answers


def _read_judged(path: Path) -> tuple[list[JudgedStatement], dict[str, str], str]:
    """The statements, each answer's evidence, and the one judge that produced them."""
    statements, evidence, judges = [], {}, set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            evidence[row["answer_id"]] = row.pop("evidence")
            model = row.pop("judge_model", "an unrecorded judge")
            prompt = row.pop("judge_prompt", "unrecorded")
            judges.add(f"{model}, judge prompt {prompt}")
            statements.append(JudgedStatement(**row))
    if len(judges) > 1:
        raise ValueError(f"{path} mixes verdicts from {len(judges)} judges: {sorted(judges)}")
    return statements, evidence, judges.pop() if judges else "no judge"


def _describe(judge: str) -> str:
    current = prompt_fingerprint()
    if judge.endswith(current):
        return f"judged by {judge}"
    return f"judged by {judge}; the current judge prompt is {current}"


def _judge(model: str | None) -> ClaudeJudge:
    """The judge on ``model``, or on ``ANTHROPIC_JUDGE_MODEL`` when none is named."""
    config = ModelConfig.from_env()
    import anthropic

    return ClaudeJudge(
        model=model or config.judge_model,
        client=anthropic.Anthropic(api_key=config.anthropic_api_key),
    )


def command_judge(args: argparse.Namespace) -> int:
    answers = _answers(args.reasons, args.transcripts)
    if not answers:
        raise ValueError("no answers to judge; pass --reasons and/or --transcripts")
    judge = _judge(args.model)
    provenance = {"judge_model": judge.model, "judge_prompt": prompt_fingerprint()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.out.open("w", encoding="utf-8") as handle:
        for answer in answers:
            for statement in judge_answer(answer, judge):
                row = {**statement.as_dict(), **provenance, "evidence": answer.evidence}
                handle.write(json.dumps(row) + "\n")
                count += 1
    sampling = "temperature 0" if judge.deterministic else "default sampling: re-runs may differ"
    print(
        f"judged {count} statement(s) from {len(answers)} answer(s) with {judge.model} "
        f"({sampling}), judge prompt {provenance['judge_prompt']}"
    )
    print(f"tokens: {judge.input_tokens:,} in, {judge.output_tokens:,} out; wrote {args.out}")
    return 0


def command_rescore(args: argparse.Namespace) -> int:
    if args.out.exists():
        raise ValueError(f"{args.out} already exists; a re-scored sample is written once")
    labels = read_labels(args.labels)
    if not labels:
        # Refused before anything is written: an empty re-score would block every later one.
        raise ValueError(f"{args.labels} holds no labels")
    statements, evidence, _ = _read_judged(args.judged)
    chosen = [s for s in statements if s.statement_id in labels]
    missing = sorted(set(labels) - {s.statement_id for s in chosen})
    if missing:
        raise ValueError(f"{len(missing)} labelled statement(s) are not in {args.judged}")
    judge = _judge(args.model)
    provenance = {"judge_model": judge.model, "judge_prompt": prompt_fingerprint()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for statement in chosen:
            supported, rationale = judge_statement(
                judge, evidence[statement.answer_id], statement.statement
            )
            again = replace(statement, supported=supported, rationale=rationale)
            row = {**again.as_dict(), **provenance, "evidence": evidence[statement.answer_id]}
            handle.write(json.dumps(row) + "\n")
    sampling = "temperature 0" if judge.deterministic else "default sampling: re-runs may differ"
    print(
        f"re-judged the {len(chosen)} labelled statement(s) with {judge.model} ({sampling}), "
        f"judge prompt {provenance['judge_prompt']}"
    )
    print(f"tokens: {judge.input_tokens:,} in, {judge.output_tokens:,} out; wrote {args.out}")
    return 0


def command_report(args: argparse.Namespace) -> int:
    statements, _, judge = _read_judged(args.judged)
    print(_describe(judge))
    by_system: dict[str, list[JudgedStatement]] = defaultdict(list)
    for statement in statements:
        by_system[statement.system].append(statement)
    print(f"  {'system':<24}{'statements':>11}{'faithful':>10}{'cited':>8}{'failed':>8}")
    for system, group in sorted(by_system.items()):
        failed = sum(s.supported is None for s in group)
        print(
            f"  {system:<24}{len(group):>11}{faithfulness_rate(group):>10.0%}"
            f"{citation_coverage(group):>8.0%}{failed:>8}"
        )
    print("\nTrust these only after `transit-judge kappa` has passed on a labelled sample.")
    return 0


def command_sample(args: argparse.Namespace) -> int:
    if args.out.exists():
        # The author may have labelled it in place, and drawing again would erase that.
        raise ValueError(f"{args.out} already exists; a drawn sample is written once")
    statements, evidence, _ = _read_judged(args.judged)
    drawn: set[str] = set()
    for sheet in args.exclude:
        drawn |= read_statement_ids(sheet)
    unknown = sorted(drawn - {s.statement_id for s in statements})
    if unknown:
        # Ignored, a wrong sheet would exclude nothing and an earlier round could repeat.
        raise ValueError(
            f"{len(unknown)} statement(s) in --exclude are not in {args.judged}, e.g. "
            f"{unknown[0]}: was that sheet drawn from other answers?"
        )
    pool = [s for s in statements if s.statement_id not in drawn]
    sample = stratified_sample(pool, args.share, args.seed)
    if not sample:
        raise ValueError(f"{args.judged} has no statement with a verdict left to draw")
    write_blind_sheet(sample, evidence, args.out)
    left_out = f" (after leaving out {len(drawn)} already drawn)" if drawn else ""
    print(
        f"wrote {len(sample)} of {len(pool)} statement(s){left_out} to {args.out}, without the "
        "judge's verdicts. Fill human_supported with yes or no, then run `transit-judge kappa`."
    )
    return 0


def command_kappa(args: argparse.Namespace) -> int:
    statements, _, judge = _read_judged(args.judged)
    print(_describe(judge))
    labels = read_labels(args.labels)
    if not labels:
        raise ValueError(f"{args.labels} holds no labels")
    judged = {s.statement_id: s for s in statements if s.supported is not None}
    missing = sorted(set(labels) - set(judged))
    if missing:
        raise ValueError(f"{len(missing)} labelled statement(s) have no verdict, e.g. {missing[0]}")
    ids = sorted(labels)
    judge_says = [bool(judged[i].supported) for i in ids]
    author_says = [labels[i] for i in ids]
    kappa = cohens_kappa(judge_says, author_says)
    if math.isnan(kappa):
        print(f"Cohen's κ is undefined: judge and author gave one label to all {len(ids)}")
        print("  this sample cannot validate the judge; label one holding both verdicts")
        return 2
    agreement = sum(a == b for a, b in zip(judge_says, author_says, strict=True)) / len(ids)
    print(f"Cohen's κ = {kappa:.2f} over {len(ids)} statement(s); raw agreement {agreement:.0%}")
    verdict = "passes" if kappa >= KAPPA_FLOOR else "FAILS -- revise the judge prompt, re-score"
    print(f"  the floor is {KAPPA_FLOOR} (docs/08 §3.5): the judge {verdict}")
    for statement_id in ids:
        s = judged[statement_id]
        if s.supported != labels[statement_id]:
            print(f"  disagree {statement_id}: judge={s.supported} author={labels[statement_id]}")
            print(f"    {s.statement[:160]}")
            print(f"    judge's reason: {s.rationale[:160]}")
    return 0 if kappa >= KAPPA_FLOOR else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="transit-judge",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    judge = sub.add_parser("judge", help="judge every statement with the judge model")
    judge.add_argument("--reasons", type=Path, default=None, help="transit-reasons --out JSONL")
    judge.add_argument(
        "--transcripts", type=Path, default=None, help="a folder of transit-ask JSON"
    )
    judge.add_argument("--out", type=Path, required=True)
    judge.add_argument(
        "--model", default=None, help="the judge model (default: ANTHROPIC_JUDGE_MODEL)"
    )
    judge.set_defaults(handler=command_judge)

    rescore = sub.add_parser("rescore", help="re-judge exactly the labelled statements")
    rescore.add_argument("--judged", type=Path, required=True)
    rescore.add_argument("--labels", type=Path, required=True)
    rescore.add_argument("--out", type=Path, required=True)
    rescore.add_argument(
        "--model", default=None, help="the judge model (default: ANTHROPIC_JUDGE_MODEL)"
    )
    rescore.set_defaults(handler=command_rescore)

    report = sub.add_parser("report", help="faithfulness and citation coverage per system")
    report.add_argument("--judged", type=Path, required=True)
    report.set_defaults(handler=command_report)

    sample = sub.add_parser("sample", help="a blind, stratified sample for the author to label")
    sample.add_argument("--judged", type=Path, required=True)
    sample.add_argument("--out", type=Path, required=True)
    sample.add_argument("--share", type=float, default=DEFAULT_SHARE)
    sample.add_argument("--seed", type=int, default=0)
    sample.add_argument(
        "--exclude",
        type=Path,
        action="append",
        default=[],
        help="a sheet from an earlier round, labelled or not: none of it is drawn again "
        "(repeatable)",
    )
    sample.set_defaults(handler=command_sample)

    kappa = sub.add_parser("kappa", help="Cohen's κ between the judge and the author's labels")
    kappa.add_argument("--judged", type=Path, required=True)
    kappa.add_argument("--labels", type=Path, required=True)
    kappa.set_defaults(handler=command_kappa)
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
