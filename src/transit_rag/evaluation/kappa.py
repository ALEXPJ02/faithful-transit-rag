"""Validating the judge against the author: Cohen's κ on a blind, stratified sample.

``docs/08`` §3.5: the author hand-labels a 20% stratified subsample, and Cohen's κ
against the judge is reported. Below 0.6, the judge's prompt is revised and the
subsample re-scored, **before** the full run and never after seeing results.

**Stratified by system and verdict.** A plain random 20% of mostly-supported
statements could hold no unsupported ones at all, and κ would then say nothing about
the case that matters. Each stratum contributes at least one statement.

**Blind.** The exported sheet holds the statement and its evidence, and leaves the
judge's verdict out. A label made beside the verdict is anchored by it, and κ would
measure agreement with the anchor.
"""

from __future__ import annotations

import csv
import math
import random
from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path

from transit_rag.evaluation.judge import JudgedStatement

#: ``docs/08`` §3.5's bar.
KAPPA_FLOOR = 0.6
DEFAULT_SHARE = 0.2


def cohens_kappa(first: Sequence[bool], second: Sequence[bool]) -> float:
    """Agreement beyond chance between two raters' yes/no labels.

    ``nan`` when chance agreement is total (both raters gave one label to
    everything): κ is then undefined, not perfect.
    """
    if len(first) != len(second):
        raise ValueError(f"{len(first)} labels against {len(second)}")
    if not first:
        return math.nan
    n = len(first)
    observed = sum(a == b for a, b in zip(first, second, strict=True)) / n
    p_first, p_second = sum(first) / n, sum(second) / n
    chance = p_first * p_second + (1 - p_first) * (1 - p_second)
    if chance == 1:
        return math.nan
    return (observed - chance) / (1 - chance)


def stratified_sample(
    statements: Sequence[JudgedStatement], share: float = DEFAULT_SHARE, seed: int = 0
) -> list[JudgedStatement]:
    """``share`` of each system x verdict stratum, at least one each, reproducibly."""
    if not 0 < share <= 1:
        raise ValueError(f"share must be in (0, 1], got {share}")
    # Exact, so 7% of 100 is 7: in floating point 0.07 x 100 is 7.000000000000001.
    exact = Fraction(share).limit_denominator(1000)
    strata: dict[tuple[str, bool | None], list[JudgedStatement]] = {}
    for statement in statements:
        if statement.supported is None:
            continue  # a failed verdict cannot be compared with anything
        strata.setdefault((statement.system, statement.supported), []).append(statement)
    rng = random.Random(seed)
    sample = []
    for key in sorted(strata, key=str):
        members = sorted(strata[key], key=lambda s: s.statement_id)
        size = math.ceil(exact * len(members))  # a positive share of one rounds up to it
        sample.extend(rng.sample(members, size))
    return sorted(sample, key=lambda s: s.statement_id)


def write_blind_sheet(
    sample: Sequence[JudgedStatement], evidence: dict[str, str], path: Path
) -> None:
    """The labelling sheet: statement and evidence, an empty ``human_supported`` column."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["statement_id", "system", "statement", "evidence", "human_supported"])
        for statement in sample:
            writer.writerow(
                [
                    statement.statement_id,
                    statement.system,
                    statement.statement,
                    evidence[statement.answer_id],
                    "",
                ]
            )


def read_labels(path: Path) -> dict[str, bool]:
    """The author's labels: ``yes`` or ``no`` per statement. Anything else is an error."""
    labels: dict[str, bool] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for number, row in enumerate(csv.DictReader(handle), start=2):
            value = (row.get("human_supported") or "").strip().lower()
            if value not in {"yes", "no"}:
                raise ValueError(
                    f"{path}:{number}: human_supported must be yes or no, got {value!r}"
                )
            labels[row["statement_id"]] = value == "yes"
    return labels
