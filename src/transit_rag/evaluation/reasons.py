"""Scoring the reasons stage: cause macro-F1, accuracy and retrieval recall (``docs/08`` §3.4).

**One case per incident, never per alert.** TfNSW republishes an incident under
new alert ids (``docs/09`` §7). Scored per alert, the twins would inflate n and
the earlier twin would leak the later one's cause. A case is the situation at
the incident's first sighting, and retrieval for it excludes the whole incident.

**Macro-F1 over the groups that occur.** A group with no true case in the set
being scored has an undefined F1 (0/0). The average runs over the groups present
in the ground truth, and :class:`ReasonScores` names them, so a reader can see
which groups a number speaks for (``docs/08`` §3.2). A wrong prediction of an
absent group still costs: it is a missed case of the true group.

**The most-common-cause baseline is time-aware too.** It predicts the commonest
group among incidents first seen *before* the one being explained. A baseline
allowed to count the future would be leakier than the system it is compared
against.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import timedelta

import pandas as pd

from transit_rag.agent.reasons import CAUSE_GROUPS
from transit_rag.agent.situation import DEFAULT_LOOKBACK, Situation, situation_at
from transit_rag.ingestion.alerts import Incident, sydney_date


@dataclass(frozen=True)
class ReasonCase:
    """One incident to explain, and what the feed showed before it was posted."""

    incident: Incident
    situation: Situation

    @property
    def truth(self) -> str:
        return self.incident.cause_group


def build_cases(
    incidents: Sequence[Incident],
    events: pd.DataFrame,
    stations: dict[str, str],
    *,
    since: str | None = None,
    lookback: timedelta = DEFAULT_LOOKBACK,
) -> list[ReasonCase]:
    """A case per incident first seen on or after ``since`` (a Sydney date)."""
    cases = []
    for incident in incidents:
        if since is not None and sydney_date(incident.first_seen) < since:
            continue
        situation = situation_at(
            events, incident.lines, incident.first_seen, stations, lookback=lookback
        )
        cases.append(ReasonCase(incident=incident, situation=situation))
    return cases


def most_common_cause(case: ReasonCase, incidents: Sequence[Incident]) -> str:
    """The commonest group among incidents first seen before this case's.

    Ties go to the group listed first in ``CAUSE_GROUPS``, and an incident with
    no history behind it gets ``other_unknown``. Both rules are arbitrary but
    fixed, and stated so the baseline cannot be tuned.
    """
    earlier = Counter(
        incident.cause_group
        for incident in incidents
        if incident.first_seen < case.incident.first_seen
    )
    if not earlier:
        return "other_unknown"
    return max(CAUSE_GROUPS, key=lambda group: (earlier[group], -CAUSE_GROUPS.index(group)))


@dataclass(frozen=True)
class ReasonScores:
    """One system's cause predictions against the truth."""

    n: int
    accuracy: float
    macro_f1: float
    #: The groups macro-F1 averages over: those present in the truth.
    groups: tuple[str, ...]
    per_group_f1: dict[str, float] = field(default_factory=dict)


def score_reasons(truth: Sequence[str], predicted: Sequence[str]) -> ReasonScores:
    """Accuracy, and F1 per group averaged over the groups that occur."""
    if len(truth) != len(predicted):
        raise ValueError(f"{len(truth)} true causes but {len(predicted)} predictions")
    if not truth:
        return ReasonScores(n=0, accuracy=float("nan"), macro_f1=float("nan"), groups=())
    present = tuple(group for group in CAUSE_GROUPS if group in set(truth))
    per_group = {}
    for group in present:
        hits = sum(t == p == group for t, p in zip(truth, predicted, strict=True))
        flagged = sum(p == group for p in predicted)
        actual = sum(t == group for t in truth)
        per_group[group] = 2 * hits / (flagged + actual)
    return ReasonScores(
        n=len(truth),
        accuracy=sum(t == p for t, p in zip(truth, predicted, strict=True)) / len(truth),
        macro_f1=sum(per_group.values()) / len(per_group),
        groups=present,
        per_group_f1=per_group,
    )


def retrieval_recall(truth: Sequence[str], retrieved: Sequence[Sequence[str]]) -> float:
    """Share of cases where at least one retrieved incident shares the true group.

    ``retrieved`` holds, per case, the cause groups of the passages returned.
    """
    if len(truth) != len(retrieved):
        raise ValueError(f"{len(truth)} cases but {len(retrieved)} retrieval results")
    if not truth:
        return float("nan")
    return sum(t in set(groups) for t, groups in zip(truth, retrieved, strict=True)) / len(truth)
