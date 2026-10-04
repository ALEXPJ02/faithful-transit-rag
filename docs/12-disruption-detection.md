# Disruption Detection

> RQ2's detection half: what a detector may see at the end of a window, what it is
> asked, how it is scored, and the baseline it has to beat. Built on 2026-10-05 on
> the label in [`11-disruption-labels.md`](./11-disruption-labels.md). The baseline
> comes first, as `baseline.py` did for the delay model, so no comparison can be
> retrofitted ([`08-evaluation-plan.md`](./08-evaluation-plan.md) §3.3).

## 1. The question

At prediction time *t*, which is the end of a 15-minute window, the question is
whether the line will be disrupted **in the next 30 minutes**, that is, in either of
the next two windows. The horizon and the window size are agreed with the supervisor.
The target uses the window labels of `docs/11` with Kleene logic:

- it is **disrupted** if either window is;
- it is **not disrupted** only if both windows are labelled and neither is disrupted;
- it is **unknown** (`pd.NA`) otherwise. In practice that means overnight windows
  with no service, a target that crosses the 03:00 service-day turnover, and the
  end of the data.

An unknown target is in no denominator. A target that crosses 03:00 would put its
windows in two partitions of the chronological split, so it is never built. Nothing
runs at that hour, so nothing is lost.

## 2. What a detector may see

`features.FEATURE_COLUMNS` lists what a detector may use. Everything comes from the
current window or earlier ones:

| Group | Features |
| --- | --- |
| The current window | services observed, services late, share late, mean and maximum delay |
| The 45 minutes before | share late, services observed and maximum delay, for each of the three previous windows |
| When | Sydney hour, day of week, weekend, weekday peak |
| Where | line |
| The operator | **`alert_in_feed`**: whether a disruption alert was in the feed (below) |

`features.FORBIDDEN` lists the rest, each with its reason. Share late is rule (b)'s
own input. If it were computed over the windows the target comes from, it would *be*
the target. That is why features look back, the target looks forward, and the two
never share a window. Every mutation of either boundary is caught by a test.

### The alert feature is re-derived, not copied

The window label's rule (a) is written in hindsight. It knows:

- whether an alert outlived 24 hours, which only becomes knowable 24 hours later;
- which overrides were decided afterwards;
- which republications were merged, before some of them were published.

None of that is known at *t*. So `alert_in_feed` asks what the feed showed at the
**last alert poll before *t***, and applies the same `classify()` rule with presence
counted only up to that poll and with no override. An alert that will later become
a standing notice counts until it has been in the feed 24 hours. Harris Park,
published as `MAINTENANCE`, never counts, because only the override let it into
the label.

On the 2026-10-05 snapshot, `alert_in_feed` is true at the end of 106 windows. The
hindsight rule (a) marks 111.

## 3. How a detector is scored

| Metric | Why |
| --- | --- |
| **Average precision** (headline) | Summarises every threshold. Negatives make up 94% of windows, and they cannot inflate it the way they inflate accuracy |
| Precision, recall, F1 | At one threshold, the best F1 **on validation**, with ties going to the higher threshold (fewer alarms) |
| False alarms per day | Whether anyone could live with the detector |
| Recall split by rule | The target's two rules measure different things (`11` §5). A detector built on delay features can only be expected to find what rule (b) sees |

Lead time needs each flag lined up against the operator's alert, so it is scored in
the evaluation harness.

## 4. The baseline: persistence

"The next 30 minutes are disrupted if this window is." This reads the current
window's own label, rule (a)'s hindsight included. That makes it **stronger** than any
deployable baseline, which is the conservative direction for a claim that a model
beats it. Where nothing ran, the current window counts as not disrupted.

`transit-detect --db data/delay_observations_20261005.db`. The split is chronological
by service date, 70/15/15:

| | Rows | Target known | Positive | Dates |
| --- | --- | --- | --- | --- |
| Train | 2,706 | 2,538 | 158 | 2026-09-14 to 09-28 (15) |
| Validation | 576 | 539 | 46 | 2026-09-29 to 10-01 (3) |
| Test | 554 | 537 | **6** | 2026-10-02 to 10-04 (3) |

| Persistence | AP | Precision | Recall | F1 | False alarms / day | Recall, rule (a) | Recall, rule (b) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Validation | 0.457 | 74% | 57% | 64% | 3.0 | 72% | 45% |
| Test | 0.011 | 0% | 0% | 0% | 0.7 | n/a | 0% |

Persistence is a strong baseline where disruptions last, and they usually do. On
validation it catches 72% of rule (a)'s targets, because an alert in the feed now is
usually still there in 30 minutes. It catches fewer of rule (b)'s, because lateness
builds and clears faster than the operator posts.

## 5. The detectors

`models.py` fits XGBoost and a random forest (`08` §3.3) on `FEATURE_COLUMNS` and
nothing else. Each is chosen the way the delay model is: a small explicit grid, the
candidate with the best **validation** average precision, and a threshold set on
validation. XGBoost also stops early on validation. Test is touched once. Neither
model is reweighted for the 6% base rate. Average precision does not need it, and
reweighting changes what a score means without improving the ranking.
`transit-detect` scores both, then both again **without `alert_in_feed`**, which is
the ablation `08` §3.5 names.

On the 2026-10-05 snapshot:

| Validation (46 positives) | AP | F1 | False alarms / day | Recall, rule (a) | Recall, rule (b) |
| --- | --- | --- | --- | --- | --- |
| Persistence | 0.457 | 64% | 3.0 | 72% | 45% |
| XGBoost | 0.631 | 62% | 6.7 | 67% | 62% |
| Random forest | 0.592 | 65% | 5.3 | 72% | 59% |
| XGBoost, no alert feature | 0.489 | 53% | 8.0 | 28% | 69% |
| Random forest, no alert feature | 0.477 | 52% | 3.7 | 22% | 55% |

**These validation figures are selection scores, not results.** The detectors'
settings, early stopping and thresholds were all chosen on these rows, so their AP is
optimistic in a way persistence's is not. The comparison that counts is on test.

**The ablation already says something.** Without the operator's alert, recall on
rule (a)'s targets collapses from about 70% to about 25%, while rule (b)'s holds.
Delay features find what rule (b) sees. Seeing what rule (a) sees needs the
operator, as `11` §5 predicted. In XGBoost the most important features are mean
delay (29%), services late (16%) and the alert flag (13%).

## 6. Today's test split is too thin to compare anything

Six positive targets, none from rule (a), on three quiet days. On that, a detector's
AP is noise: persistence scores 0.011 by missing the six. **No detection result on
today's test split should be quoted as a comparison.** The split that counts is drawn
once, over the data to the cut-off at the end of 2026-10-18. That puts about five dates
in test. Every scored result is reported with its count of positive targets beside
it, so a reader can see how much it rests on (`08` §3.1).

On today's test split, both detectors score AP 0.06–0.08 against persistence's
0.011, on six positives. That is noise and is not quoted as a result.

## 7. Next

- **The scored run**, on the split drawn at the cut-off, with every figure reported
  beside its count of positive targets and with confidence intervals from resampling
  whole service dates (`08` §3.5). Add an LSTM if time allows.
- **Lead time**, in the evaluation harness.
- **Timetabled services and cancellations as features.** They need a calendar-aware
  reading of the static timetable, the same one the label's cancellation check needs
  (`11` §7). Until that exists, a thinned service shows up only as fewer services
  observed.
