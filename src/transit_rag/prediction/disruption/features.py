"""What a detector may know at the end of a window, and what it is asked.

``docs/08`` §3.3. At prediction time ``t``, the end of window ``w_t``, a
detector sees line-level aggregates over ``w_t`` and the windows before it. It
is asked whether the line is disrupted in the next 30 minutes, that is, in
``w_t+1`` or ``w_t+2``.

Three ways this goes wrong silently, each closed here:

- **share_late is rule (b)'s input.** Computed over the windows the target
  comes from, it *is* the target. Features come from ``w_t`` and earlier, and
  the target from the two windows after. They never share a window.
- **Rule (a) is hindsight.** The window label knows whether an alert outlived
  24 hours, applies overrides decided afterwards, and merges republications
  that had not happened yet. None of that is known at ``t``. So
  :func:`alert_in_feed` re-derives the alert feature from what the feed showed
  at the last alert poll before ``t``. It uses the same rule, with presence
  counted only up to that poll, and with no override.
- **A target that straddles service dates leaks across the split.** Its two
  windows would sit in different partitions of the chronological split, so the
  target is ``pd.NA``. That only happens around 03:00, when nothing runs.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import numpy as np
import pandas as pd

from transit_rag.ingestion.alerts import MAX_INCIDENT_PRESENCE, AlertRecord, classify
from transit_rag.prediction.disruption.labels import WINDOW, LabelRule, latest_by_service
from transit_rag.prediction.features.reconcile import AM_PEAK, PM_PEAK
from transit_rag.realtime.parsing import SYDNEY

#: "The next 30 minutes" (agreed with the supervisor), in 15-minute windows.
HORIZON_WINDOWS = 2

#: How many earlier windows a detector sees besides the current one: 45
#: minutes of history.
HISTORY_WINDOWS = 3

#: Aggregates of the current window that are also given for earlier windows.
LAGGED: tuple[str, ...] = ("share_late", "n_services", "max_delay_s")

#: What a detector may use. :data:`FORBIDDEN` says why the rest may not.
FEATURE_COLUMNS: tuple[str, ...] = (
    "line",
    "n_services",
    "n_late",
    "share_late",
    "mean_delay_s",
    "max_delay_s",
    *(f"{name}_lag{k}" for k in range(1, HISTORY_WINDOWS + 1) for name in LAGGED),
    "hour_local",
    "day_of_week",
    "is_weekend",
    "is_peak",
    "alert_in_feed",
)

#: Columns in the detection table that must never become features.
FORBIDDEN: dict[str, str] = {
    "target": "what is being predicted",
    "target_rule_a": "a component of the target",
    "target_rule_b": "a component of the target",
    "disrupted_now": "the window label, which knows rule (a)'s hindsight; the persistence "
    "baseline may read it, a detector may not",
    "rule_a": "hindsight (24-hour cap, overrides, republications not yet published)",
    "incident_ids": "names the incidents rule (a) was built from",
    "service_date": "the split key",
    "window_start_utc": "an identifier, not a condition",
    "window_start_local": "an identifier, not a condition",
}


def window_delays(events: pd.DataFrame, rule: LabelRule) -> pd.DataFrame:
    """Mean and maximum delay over the services observed in each line x window."""
    latest = latest_by_service(events, rule)
    delays = latest.groupby(["line", "window_start_utc"], as_index=False).agg(
        mean_delay_s=("delay_s", "mean"), max_delay_s=("delay_s", "max")
    )
    # Plain floats with NaN, which every detector reads natively. A nullable
    # Float64 would carry pd.NA, which no comparison can be made against.
    return delays.astype({"mean_delay_s": float, "max_delay_s": float})


def alert_in_feed(
    frame: pd.DataFrame, alerts: Sequence[AlertRecord], alert_polls: pd.Series
) -> pd.Series:
    """Whether, at each window's end, the feed carried an alert the rule would call a disruption.

    "Carried" means the alert was in the last alert poll at or before the
    window's end. "Would call" means :func:`~transit_rag.ingestion.alerts.classify`,
    with no override, on the cause and wording, and with presence counted
    only up to that poll. An alert that will later outlive 24 hours still
    counts until it has, because until then nobody could know.
    """
    polls = pd.to_datetime(alert_polls, utc=True).sort_values().reset_index(drop=True)
    ends = frame["window_start_utc"] + WINDOW
    index = (
        np.searchsorted(
            polls.dt.tz_convert(None).to_numpy(), ends.dt.tz_convert(None).to_numpy(), side="right"
        )
        - 1
    )
    last_poll = polls.iloc[np.clip(index, 0, None)].reset_index(drop=True)
    last_poll.index = frame.index
    last_poll = last_poll.where(index >= 0)

    flag = pd.Series(False, index=frame.index)
    for alert in alerts:
        # Cause and wording do not change with time; only presence does, and
        # the cap is applied against the poll below.
        at_first_sight = dataclasses.replace(alert, last_seen=alert.first_seen)
        if not classify(at_first_sight).is_incident:
            continue
        first, last = pd.Timestamp(alert.first_seen), pd.Timestamp(alert.last_seen)
        carried = (
            frame["line"].isin(alert.lines)
            & (last_poll >= first)
            & (last_poll <= last)
            & (last_poll - first <= MAX_INCIDENT_PRESENCE)
        )
        flag |= carried
    return flag


def _time_features(frame: pd.DataFrame) -> pd.DataFrame:
    local = frame["window_start_utc"].dt.tz_convert(SYDNEY)
    hour = local.dt.hour
    weekend = local.dt.dayofweek >= 5
    peak = (
        hour.between(AM_PEAK[0], AM_PEAK[1] - 1) | hour.between(PM_PEAK[0], PM_PEAK[1] - 1)
    ) & ~weekend
    return pd.DataFrame(
        {
            "hour_local": hour.astype(int),
            "day_of_week": local.dt.dayofweek.astype(int),
            "is_weekend": weekend.astype(bool),
            "is_peak": peak.astype(bool),
        },
        index=frame.index,
    )


def _targets(frame: pd.DataFrame) -> pd.DataFrame:
    """The next 30 minutes' label, and which rule made it positive.

    Kleene logic: disrupted if either window is, undisrupted only if both are
    known to be, otherwise ``pd.NA``. The per-rule columns let detection be
    reported against each rule separately (``docs/11`` §5).
    """
    by_line = frame.groupby("line", sort=False)
    known_false = pd.Series(True, index=frame.index)
    any_true = pd.Series(False, index=frame.index)
    rule_a = pd.Series(False, index=frame.index)
    rule_b = pd.Series(False, index=frame.index)
    same_day = pd.Series(True, index=frame.index)
    for step in range(1, HORIZON_WINDOWS + 1):
        later = by_line["disrupted"].shift(-step).astype("boolean")
        any_true |= later.fillna(False).astype(bool)
        known_false &= later.notna() & ~later.fillna(True).astype(bool)
        labelled = later.notna()
        rule_a |= labelled & by_line["rule_a"].shift(-step).fillna(False).astype(bool)
        rule_b |= labelled & by_line["rule_b"].shift(-step).fillna(False).astype(bool)
        same_day &= by_line["service_date"].shift(-step) == frame["service_date"]

    target = pd.Series(pd.NA, index=frame.index, dtype="boolean")
    target[any_true] = True
    target[known_false & ~any_true] = False
    target[~same_day] = pd.NA
    return pd.DataFrame(
        {
            "target": target,
            "target_rule_a": rule_a & target.notna(),
            "target_rule_b": rule_b & target.notna(),
        },
        index=frame.index,
    )


def detection_table(
    labels: pd.DataFrame,
    events: pd.DataFrame,
    alerts: Sequence[AlertRecord],
    alert_polls: pd.Series,
    rule: LabelRule | None = None,
) -> pd.DataFrame:
    """One row per line x window: what is known at its end, and the next 30 minutes' label.

    ``labels`` is :func:`~transit_rag.prediction.disruption.labels.label_windows`'
    output; ``events``, ``alerts`` and ``alert_polls`` come from the same
    snapshot.
    """
    rule = rule or LabelRule()
    frame = (
        labels.merge(window_delays(events, rule), on=["line", "window_start_utc"], how="left")
        .sort_values(["line", "window_start_utc"], kind="stable")
        .reset_index(drop=True)
    )
    by_line = frame.groupby("line", sort=False)
    for k in range(1, HISTORY_WINDOWS + 1):
        for name in LAGGED:
            frame[f"{name}_lag{k}"] = by_line[name].shift(k)

    frame = pd.concat([frame, _time_features(frame)], axis=1)
    frame["alert_in_feed"] = alert_in_feed(frame, alerts, alert_polls)
    frame = pd.concat([frame, _targets(frame)], axis=1)
    # The persistence baseline's input: the current window's own label, read
    # as "not disrupted" where nothing was observed to say otherwise.
    frame["disrupted_now"] = frame["disrupted"].fillna(False).astype(bool)

    leaked = sorted(set(FEATURE_COLUMNS) & set(FORBIDDEN))
    if leaked:  # pragma: no cover - the two constants are edited together
        raise RuntimeError(f"forbidden columns listed as features: {leaked}")
    return frame
