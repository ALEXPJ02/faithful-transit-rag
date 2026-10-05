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
| Precision, recall, F1 | At one threshold, the best F1 **on validation**, with ties going to the higher threshold (fewer alarms). The threshold is placed midway to the next lower score, never on one (§5) |
| Lead time | Minutes between the detector's first flag and the operator's alert, for each incident (`evaluation/lead_time.py`) |
| False alarms per day | Whether anyone could live with the detector |
| Recall split by rule | The target's two rules measure different things (`11` §5). A detector built on delay features can only be expected to find what rule (b) sees |

Every AP is reported with a **95% interval from resampling whole service dates**
(`evaluation/stats.py`, `08` §3.5). Windows within a day are not independent, and
resampling them would report an interval far narrower than the evidence allows.

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
| Test | 554 | 538 | **7** | 2026-10-02 to 10-04 (3) |

| Persistence | AP | Precision | Recall | F1 | False alarms / day | Recall, rule (a) | Recall, rule (b) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Validation | 0.457 | 74% | 57% | 64% | 3.0 | 72% | 45% |
| Test | 0.059 | 33% | 14% | 20% | 0.7 | n/a | 14% |

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
| Random forest, no alert feature | 0.477 | 54% | 3.7 | 22% | 59% |

**These validation figures are selection scores, not results.** The detectors'
settings, early stopping and thresholds were all chosen on these rows, so their AP is
optimistic in a way persistence's is not. The comparison that counts is on test.

**The ablation already says something.** Without the operator's alert, recall on
rule (a)'s targets collapses from about 70% to about 25%, while rule (b)'s holds.
Delay features find what rule (b) sees. Seeing what rule (a) sees needs the
operator, as `11` §5 predicted. In XGBoost the most important features are mean
delay (29%), services late (16%) and the alert flag (13%).

### Lead time, intervals, and a reproducibility fix

**Lead time** is scored for every incident on each line it names. It is the
operator's first alert minus the detector's first flag, searched from 60 minutes
before the alert to the incident's last window, so positive means earlier. The
60-minute lookback is fixed a priori: the detector's claim reaches 30 minutes ahead,
and the alert's own time is uncertain by one 30-minute poll. A lead time is
therefore good to about 30 minutes. On validation, which has 3 incidents:

| | Flagged | Before the alert | Median lead |
| --- | --- | --- | --- |
| Persistence, and both alert-fed detectors | 3 of 3 | 1 | −8 min |
| XGBoost, no alert feature | 2 of 3 | 2 | +8 min |
| Random forest, no alert feature | 1 of 3 | 1 | +9 min |

The detectors that read the operator's alert mostly flag **when it appears**. The
ones that cannot see it flag **before** it on what they catch, but they catch less.
That is the trade-off rule (a) and rule (b) already suggested (`11` §5). With three
incidents, it is a direction, not a finding. Today's test split holds no incident.

**The intervals are wide, as they should be at three dates.** For example, the
validation AP is 0.631 [0.473, 0.789] for XGBoost and 0.457 [0.204, 0.607] for
persistence.

**Re-runs now reproduce byte for byte.** Comparing two runs showed the no-alert
forest's precision and recall moving at an unchanged threshold. The forest predicted
in parallel, summing tree probabilities in thread order, so scores differed in their
last bit (1e-16). One validation row sat within 1e-12 of a threshold that was
*equal* to a score, so it flipped. The forest now predicts on one thread, and every
threshold sits midway between two scores. Validation flags are unchanged by
construction, and two runs of `transit-detect` now give identical output. The table
above is from the fixed code.

## 6. Today's test split is too thin to compare anything

Seven positive targets, none from rule (a), on three quiet days. On that, a detector's
AP is noise: persistence scores 0.059 by catching one of the seven. **No detection result on
today's test split should be quoted as a comparison.** The split that counts is drawn
once, over the data to the cut-off at the end of 2026-10-18. That puts about five dates
in test. Every scored result is reported with its count of positive targets beside
it, so a reader can see how much it rests on (`08` §3.1).

On today's test split the detectors score AP 0.08–0.12 against persistence's
0.059, on seven positives. That is noise and is not quoted as a result.

## 7. Next

- **The scored run**, on the split drawn at the cut-off, with every figure reported
  beside its count of positive targets and its date-resampled interval. Add an LSTM
  if time allows.
- **Timetabled services and cancellations as features.** Cancelled calls can now be
  placed in windows, which is how the label check does it (`11` §7). They exist only from
  2026-09-29, so the feature is `pd.NA` before then. A count of *timetabled* services
  needs a calendar-aware reading of the whole timetable, which is not built. Until it
  is, a thinned service shows up only as fewer services observed.
