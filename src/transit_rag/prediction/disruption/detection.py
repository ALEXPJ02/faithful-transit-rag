"""Scoring a disruption detector, and the persistence baseline it has to beat.

``docs/08`` §3.4. The headline is **average precision**: it summarises every
threshold, and many easy negative windows cannot inflate it the way they
inflate accuracy. At about 4.5% positives, "never disrupted" is 95.5% accurate
and useless. Precision, recall and F1 are reported at one threshold, chosen on
validation. **False alarms per day** says whether anyone could live with the
detector. Recall is also split by which rule made the target positive, because
the two rules measure different things (``docs/11`` §5). A delay-feature
detector can only be expected to find what rule (b) sees.

**Persistence is written first**, as ``baseline.py`` was for the delay model:
"the next window is disrupted if this one is" (``docs/08`` §3.3). It reads the
current window's own label, rule (a)'s hindsight included. That makes it a
*stronger* baseline than any deployable one, which is the conservative
direction for a claim that a model beats it.

Lead time needs the operator's alerts lined up against each flag, and it is
scored in the evaluation harness, not here.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DetectionScores:
    """One detector's performance on one set of rows."""

    n: int
    positives: int
    average_precision: float
    threshold: float
    precision: float
    recall: float
    f1: float
    false_alarms_per_day: float
    #: Recall over targets made positive by each rule. They overlap.
    recall_rule_a: float
    recall_rule_b: float
    days: int

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else math.nan


def _f1(hits: int, flagged: int, positives: int) -> float:
    """2TP / (2TP + FP + FN): zero when flags or positives exist but never meet."""
    return _ratio(2 * hits, flagged + positives)


def average_precision(target: np.ndarray, score: np.ndarray) -> float:
    """Area under the precision-recall step curve; ``nan`` when nothing is positive."""
    from sklearn.metrics import average_precision_score

    if not target.any():
        return math.nan
    return float(average_precision_score(target, score))


def score_detector(table: pd.DataFrame, score: pd.Series, threshold: float) -> DetectionScores:
    """Score ``score`` against ``table``'s target, on the rows where the target is known.

    ``table`` is a slice of :func:`~transit_rag.prediction.disruption.features.detection_table`.
    A row whose target is ``pd.NA`` is in no denominator. Nothing can be
    right or wrong about a window with no service in it.
    """
    known = table["target"].notna().to_numpy()
    target = table["target"].to_numpy(dtype=bool, na_value=False)[known]
    scores = score.to_numpy(dtype=float)[known]
    flagged = scores >= threshold
    rule_a = table["target_rule_a"].to_numpy(dtype=bool)[known]
    rule_b = table["target_rule_b"].to_numpy(dtype=bool)[known]

    true_positives = int((flagged & target).sum())
    precision = _ratio(true_positives, int(flagged.sum()))
    recall = _ratio(true_positives, int(target.sum()))
    f1 = _f1(true_positives, int(flagged.sum()), int(target.sum()))
    days = int(table.loc[known, "service_date"].nunique())
    false_alarms = int((flagged & ~target).sum())
    return DetectionScores(
        n=int(known.sum()),
        positives=int(target.sum()),
        average_precision=average_precision(target, scores),
        threshold=threshold,
        precision=precision,
        recall=recall,
        f1=f1,
        false_alarms_per_day=_ratio(false_alarms, days),
        recall_rule_a=_ratio(int((flagged & rule_a).sum()), int(rule_a.sum())),
        recall_rule_b=_ratio(int((flagged & rule_b).sum()), int(rule_b.sum())),
        days=days,
    )


def persistence(table: pd.DataFrame) -> pd.Series:
    """The baseline: the next 30 minutes are disrupted if this window is. 1.0 or 0.0."""
    return table["disrupted_now"].astype(float)


def best_f1_threshold(table: pd.DataFrame, score: pd.Series) -> float:
    """The threshold with the highest F1 on ``table``. Pass validation, never test.

    Ties go to the higher threshold: fewer alarms for the same F1.

    **The threshold sits between two scores, never on one.** It is the midpoint
    between the chosen score and the next lower distinct score, so it flags
    exactly the same validation rows. But a score that differs in its last bit,
    from another machine or another thread's summation order, cannot flip the
    row it would otherwise be sitting on. Measured 2026-10-05: one validation
    row lay within 1e-12 of the forest's threshold, and re-runs moved its
    precision and recall.
    """
    known = table["target"].notna().to_numpy()
    target = table["target"].to_numpy(dtype=bool, na_value=False)[known]
    scores = score.to_numpy(dtype=float)[known]
    candidates = np.unique(scores)[::-1]
    best_index, best_f1 = -1, -1.0
    for index, threshold in enumerate(candidates):
        flagged = scores >= threshold
        f1 = _f1(int((flagged & target).sum()), int(flagged.sum()), int(target.sum()))
        if f1 > best_f1:
            best_index, best_f1 = index, f1
    if best_index < 0:
        return 0.5
    chosen = float(candidates[best_index])
    if best_index + 1 < len(candidates):
        return (chosen + float(candidates[best_index + 1])) / 2
    return chosen / 2  # the lowest score: flag everything, with margin below it
