# Research Questions and the Evaluation Plan

> Restructured 2026-09-22 on Dr Ramezani's direction: two research questions, trains
> only, service disruptions on T1 and T4. The old RQ1–RQ3 became RQ1's objectives and
> the old RQ4 became RQ2's. **Opal fare policy is out**; RAG stays, and now retrieves
> past service alerts to explain a disruption instead of retrieving fare documents.
>
> **Updated 2026-09-28** with the student's decisions on the open items: the disruption
> definition (§3.2), a data cut-off (§3.1), the weather class, "other transport events"
> and reinforcement learning. Each is marked *pending the supervisor's confirmation*
> until she agrees it. The 30-minute horizon and the 15-minute window are agreed.
>
> The academic source of truth is `RQ_List_and_Evaluation_Methods_v3.docx`, one level
> up in `Capstone/`. This file is the engineering half: what has to be built, what has
> been measured, and what is still assumed.

Two numbering schemes meet here, so to be explicit: the **RQ1–RQ4** the supervisor
refers to are those in the progress report of 2026-09-16, and her restructure folded
report RQ1–RQ3 into this file's RQ1 objectives and report RQ4 into RQ2. The
**SQ1–SQ5** below are this repo's own earlier decomposition, which is a different
thing entirely and is now gone.

This supersedes the five-sub-question decomposition (SQ1–SQ5) that stood here until
2026-09-22. That decomposition answered a different question — faithfulness of an
agentic RAG system over three sources, one of them Opal policy — and is gone rather
than renumbered, because renumbering would leave its assumptions in place.

## 1. The two questions

**RQ1 — Feasibility.** Investigate the feasibility of running machine learning-based
prediction models on real-time GTFS data to predict or determine service disruptions,
delays, and other transport events on Sydney Trains' T1 and T4 lines.

**RQ2 — Evaluation.** Develop a comprehensive evaluation plan — datasets, performance
metrics, baseline methods, experimental design — to determine which model detects the
disruptions and their reasons better.

Both are the supervisor's wording with the T1/T4 scope added.

**"Other transport events" become future work** *(decided 2026-09-28, pending the
supervisor's confirmation)*. The phrase is hers and was never defined. On this data it
could mean planned trackwork (collected, but scheduled rather than predicted),
cancellations and skipped stops (collected **from 2026-09-29 only**, in
`trip_statuses`, and used as a check on the disruption label — §3.2), station and
accessibility notices (collected as alerts), or crowding and special-event services
(not collected). RQ1 covers service disruptions and delays; the write-up names these
and says which the collected data could support.

## 2. RQ1 — the workflow, stage by stage

The objectives are the stages, in the order the workflow runs. An orchestrator agent
(the project's own Claude loop) calls each as an MCP tool and writes the final answer.

| | Task and tool | In → out | How it is checked | State |
| --- | --- | --- | --- | --- |
| **O1** Retrieval | Live T1/T4 trip updates and service alerts, joined to the static timetable. **Deterministic Python, not an LLM** | Trip updates (120 s), alerts (30 min), timetable → per-stop delay observations; active alerts with cause, effect, scope, text | Tool-faithfulness: share of the agent's statements about live conditions matching the logged tool response. Feed currency: lag between tool timestamp and snapshot | **Working** |
| **O2** Prediction | Is T1/T4 disrupted, or about to be, and by how much. **XGBoost inside the tool handler** | *Delay half, per stop event:* the nine features in `dataset.FEATURE_COLUMNS` — scheduled arrival, stop sequence, previous-stop delay, hour, day, weekend, peak, line, stop — plus an active-alert flag *(once overlap allows)* → expected delay with a 90% interval. *Disruption half, per line × 15-minute window:* line-level window features (§3.3) → probability the line is disrupted in the next 30 minutes, and a flag | The answer states the interval the tool returned, unnarrowed. Calibration: a 90% interval contains the truth ~90% of the time on held-out days | **Delay half working; disruption label built, classifier not** |
| **O3** Reasons | Likely cause of a predicted or detected disruption. **Retrieval agent over past alerts + Claude writing the reason** | O2's prediction with context, plus the five most similar past alerts → a cause category and a one/two-sentence explanation citing alert ids | Faithfulness: every statement supported by a retrieved alert or a tool output (Papageorgiou et al., 2025) | **Built; the scored run waits for the cut-off** |

### Where each objective actually stands

**O1 — working.** Trip updates collected since 2026-09-03, alerts since 2026-09-15
(Sydney time; the first poll is 2026-09-14T14:40Z). As of the 2026-09-24 snapshot: **321,634 stop events over 22 service
dates**, T1 192,114 · T4 129,520; 134 alerts and 4,272 scopes, of which **40 alerts touch T1 or T4** — that 40 is the number RQ2 actually has to work with; 15,038 of 15,039 polls
successful (one TfNSW 502) and 467 of 467 alert polls.

**O2 — the delay half is measured; the disruption half does not exist.** On the 17
schedule-covered dates, test split 2026-09-22..24 scored once:

| | MAE | RMSE | MAPE* | |
| --- | --- | --- | --- | --- |
| Naive persistence | 18.67 s | 49.2 s | 95% | |
| XGBoost | **15.45 s** | 41.5 s | 68% | |
| | | | | **MASE 0.828** |

\* MAPE covers only the 47% of rows with a non-zero delay; it is undefined on the rest.

The model beats persistence by **17.2%**. Five schedule-blind dates (2026-09-11..15,
whose timetable era was never archived) are excluded before the split — that exclusion
does *not* produce the margin: it cannot reach the test split at all, both settings
score the identical 41,022 rows, and MASE moves 0.834 → 0.828. See
`07-training-table.md`.

**The disruption label is built** (`transit-label`,
[`11-disruption-labels.md`](./11-disruption-labels.md)) from the definition in §3.2,
which is pending the supervisor's confirmation. Its thresholds are parameters, so it
can be re-run unchanged if she moves one. To 2026-10-04 it marks 4.7% of T1 windows
and 4.4% of T4 windows disrupted. The two rules agree less than one might expect.
Rule (b) fires on most windows of four incidents: Edgecliff, North Sydney, Harris Park
and the Bondi Junction repairs of 22 Sep. It fires on one window of Chatswood and on
none of the other seven. It also fires on 56 windows that no alert covers. That is the case for keeping both rules
(`11` §5).

**O3 — built, 2026-10-05.**
`transit-alerts audit` turns collected alerts into incidents — which alerts are
disruptions, and which are republications of one — with every decision printed and
reasoned (`09-service-alerts.md` §7). `transit-index build --source alerts` indexes one
passage per incident, cited by its alert ids since an alert has no page
(`10-retrieval.md` §5): **8 incidents to 2026-09-28**. Retrieval takes the two leakage
guards of §3.5 as arguments. `transit-reasons` (`13-reasons.md`) explains each
incident from the delay feed before its first alert, never from the alert itself. It
is scored against a time-aware most-common-cause baseline and the same model without
retrieval. A development run on all 12 incidents is recorded in `13` §5. It is not a
result: over three runs, retrieval doubles the model's accuracy (0.14 to 0.28), but
the majority baseline (0.42) still wins at n = 12.

## 3. RQ2 — the evaluation plan

### 3.1 Dataset

The project's own collection from the TfNSW Sydney Trains feeds, filtered to T1 and
T4. TfNSW publishes no historical realtime archive for Sydney Trains, so there is no
other source and a day not collected is gone.

- **Unit of observation:** one line (T1 or T4) in one 15-minute window. *Agreed with
  the supervisor.* The prediction horizon is 30 minutes, also agreed.
- **Data cut-off: the end of service date 2026-10-18** *(decided 2026-09-28, pending
  the supervisor's confirmation)*. Everything collected to then is the evaluation
  dataset, and the split is drawn over it once. Collection keeps running afterwards,
  but nothing after the cut-off enters a scored result. The date leaves two weeks to
  score and write before the final report is due on 2026-11-02.
- **A shortfall is reported, not hidden.** If the reasons evaluation comes out
  under-powered for lack of alerts, the write-up states how many incidents and dates
  the test split held, which metrics could not be computed and why, and what that does
  to the conclusions — cause and effect, in a form that goes straight into the report.
- **Split:** chronological by whole service date, 70/15/15, never random. A shuffle
  puts the same afternoon on both sides and invalidates the baseline comparison.
- Thresholds and settings are chosen on validation; the test dates are scored once.

### 3.2 What counts as a disruption, and as its reason

**Decided 2026-09-28, pending the supervisor's confirmation.** The thresholds are
parameters of the labeller, so a change she asks for is a re-run, not a rebuild. They
are fixed before the test dates are scored.

**Disruption.** A window is disrupted if either

(a) an alert that the **incident rule** classifies as a disruption is in the feed for
the line during the window, timed by **feed presence**
(`first_seen_utc`..`last_seen_utc`), not by `active_period`. The rule
(`09-service-alerts.md` §7) requires a cause other than `MAINTENANCE`, feed presence of
no more than **24 hours** (longer is a standing notice, not an incident), and a
description that states an effect on train running rather than planned work. Or,
(b) at least a quarter of the line's observed services are **more than 5 minutes
late**.

The 5-minute threshold is Transport for NSW's own on-time-running definition: a
suburban train is on time if it "arrived at its destination no later than its arrival
time as listed in the timetable plus an on-time tolerance" of five minutes (Audit Office
of New South Wales, 2017, *Passenger rail punctuality*). "Late" is therefore **more
than** 300 s, and the threshold is used alone. A stricter rider's standard (3 minutes)
was considered and not adopted: it is a judgement where 5 minutes is a citation, and it
moves the label from "the line is disrupted" towards ordinary lateness. The
one-quarter share is a modelling choice. Rule (b) catches disruptions the operator
never posted an alert for.

**The threshold is adapted, not copied.** TfNSW judges a train once, at its
destination. The feed reports every stop, and a window needs an answer before most
trains reach their destination, so rule (b) applies the threshold per window: a service
counts as late in a window if its **latest observed delay in that window** is above
300 s. That is the labeller's default, and the write-up states it as an adaptation.

**Why the description test** *(added 2026-09-29, pending the supervisor's
confirmation)*. Cause and duration alone admitted three planned-trackwork notices and
a police operation that closed streets, all published as `UNKNOWN_CAUSE` and all under
24 hours — and cause alone would also delete the Edgecliff closure, which is
`UNKNOWN_CAUSE` too. The description separates them. The rule was written by reading
the 22 alerts to 2026-09-28, so alerts from 2026-09-29 are held out to check it, and a
case it gets wrong is corrected in a tracked overrides file with a written reason.

**Three limits of the rules, stated rather than discovered later.**

- **Rule (b) cannot see a full closure.** With no trains running there are no services
  to be late, so a closure is caught by rule (a) or not at all.
- **The 24-hour cap would misfile a closure longer than a day** — storm or flood
  damage, say — as a notice. The cap sits in a wide empty gap on the history so far
  (incidents run 0–242 minutes; the Tangara notice below runs 7,752), so no current
  alert lands near it. If a genuine incident ever exceeds it, that is reported as a
  known misclassification, not quietly relabelled.
- **Rule (b) undercounts against TfNSW's own measure.** TfNSW counts cancelled and
  skipped-stop trains as late. TfNSW publishes a cancellation as a trip-level
  `CANCELED` with no stop updates, which the delay collector cannot see, so
  cancellations were not collected at all until **2026-09-29**; since then they go
  to `trip_statuses`. The main label still leaves them out, so that it means the same
  thing on every date. A **check** re-runs rule (b) counting cancelled and
  skipped-stop services as late, on the dates that have them, and reports how many
  windows change label *(decided 2026-09-29, pending the supervisor's confirmation)*.

> **Rule (a) as written is degenerate, and this is measured.** Applied over 15-minute
> windows from 2026-09-15 to 09-24, "an unplanned alert is active" labels **100% of T1
> and T4 windows disrupted**. A label that is always positive makes average precision
> equal to the base rate and a constant "always disrupted" predictor near-optimal.
>
> The cause is structural, not a bad threshold: **no unplanned alert carries an end
> time.** All 30 of them, across all 2,681 scopes, have `active_period_end = 0`
> (unbounded), against 361 of 1,591 for `MAINTENANCE`.
> [`09-service-alerts.md`](./09-service-alerts.md) §1 records the opposite — "always
> explicit `start` **and** `end`" — which was measured on a first-day sample that was
> entirely trackwork. Unbounded periods then compound: one standing notice, *"No access
> to Last Carriage on Tangara trains"* (`OTHER_CAUSE`, scoped to both T1 and T4 from
> 09-15 onward), is alone enough to mark every later window disrupted.
>
> **The fix is to bound the incident by feed presence, not by `active_period`:**
> `first_seen_utc` to `last_seen_utc`, at the 30-minute alert-poll resolution. On the
> real incidents that gives 0–242 minutes (median ~60), while the Tangara notice runs
> **7,752 minutes** — so a duration cap separates standing notices from incidents
> without a hand-written exclusion list.
>
> **Do not filter by cause instead.** `UNKNOWN_CAUSE` looks like a safe thing to drop —
> all 7 such alerts are headed "Station Update — *stations*" — but their
> `description_text` shows five of the seven are about services, and one is *"Trains
> are not running between Bondi Junction and Central due to an incident requiring
> emergency services at Edgecliff"*: a genuine unplanned T4 disruption, and exactly the
> event RQ2 exists to detect. **Header text does not carry the cause; read the
> description.**

**Reason (ground truth).** The cause of the *incident*: the first specific cause any
of its alerts names, and unknown only if none ever does *(decided 2026-09-29, pending
the supervisor's confirmation)*. Edgecliff was first posted as unknown and then as
police activity; the correction is the answer. Planned trackwork is excluded because it
is scheduled, not predicted. Windows flagged only by rule (b) have no recorded cause,
so they count for detection but not for reasons.

**Cause categories.** Grouped so each has enough examples. Counts below are
**incidents**, not alerts, to 2026-09-28 (`transit-alerts audit`):

| Group | Feed causes | Incidents |
| --- | --- | --- |
| Technical / infrastructure | `TECHNICAL_PROBLEM`; `MAINTENANCE` only by override, as an urgent repair (from 2026-10-05) | 4 |
| Incident on the network | `ACCIDENT` 1, `POLICE_ACTIVITY` 1, `MEDICAL_EMERGENCY` 1 | 3 |
| Weather or external | `WEATHER`, `STRIKE`, `DEMONSTRATION`, `CONSTRUCTION`, `HOLIDAY` | **0** |
| Other or unknown | `OTHER_CAUSE` 0, `UNKNOWN_CAUSE` 1 | 1 |
| **Total incidents** | from 14 alerts | **8** |

The 46 alerts naming T1 or T4 in that snapshot also include 24 `MAINTENANCE` alerts, 4
standing notices, and 4 planned-work or non-train notices, all excluded with a stated
reason.

`UNKNOWN_CAUSE` sits in "other or unknown" — the supervisor's grouping names that
class explicitly — not outside the table. The rule (a) blockquote above is why it
cannot simply be dropped.

**The four groups stay, weather included** *(decided 2026-09-28, pending the
supervisor's confirmation)*. Weather is a real cause of rail disruption even though
none has occurred yet. Note that `WEATHER` is the cause TfNSW's staff choose when they
publish an alert, not a severity threshold the feed applies, so it appears if and when
they label an incident that way.

While a group has no true cases in the test split its F1 is undefined (0/0). Macro-F1
is therefore averaged over the groups that **occur in the test split's ground truth**,
and the report names any group left out. A wrong "weather" prediction still costs the
model: it is a missed case of the true group, which lowers that group's recall.

**Eight incidents in 14 days of alert history to 2026-09-28 — not one a day, and not
one per alert.**

| | |
| --- | --- |
| Sydney dates carrying any | **5** — 09-15 (3), 09-21 (2), 09-22, 09-25, 09-26. Nine of fourteen dates have none |
| Lines | T1 6, T4 3 — Edgecliff is scoped to both |
| Alerts behind them | **14.** TfNSW republishes an incident under a new `entity.id` when its scope or text changes, because the id is content-derived: Edgecliff and Chatswood are three alerts each, North Sydney and Martin Place two each |

**Updated 2026-10-05: 12 incidents on 8 dates to 2026-10-04.** Three are new: 29 Sep
(T4, technical) and two on 1 Oct (T1, both police activity). The rule called all eight
held-out alerts correctly. The fourth addition, Harris Park signal repairs on 27 Sep
(T1), predates the freeze. TfNSW published it as `MAINTENANCE`, which the rule drops
unread, so it is in by override. The 12 come from 18 alerts. By line it is T1 9 and
T4 4. By group: technical 6, network incident 5, other/unknown 1, weather/external 0
([`09-service-alerts.md`](./09-service-alerts.md) §7). A 70/15/15 split over the 32
dates collected so far would put test at 09-30..10-04, which holds 2 incidents of one
group. That is still too narrow for macro-F1. The split that counts is drawn once, at
the cut-off.

**Two consequences the rest of this plan has to absorb.**

*The date-level bootstrap in §3.5 is degenerate.* Resampling whole service dates with
incidents on three of them means most resamples contain none, so a 95% interval on
cause macro-F1 is not meaningful at this sample size.

*The reasons test set may be empty.* On the data to 2026-09-24, a chronological
70/15/15 over the 22 collected dates put test at 09-22..24, which holds **2** unplanned alerts, both
`TECHNICAL_PROBLEM` — one cause group. **Macro-F1 over four groups is not computable
on it.** Splitting over the 10 alert-covered dates is worse: test lands on 09-23..24
with zero.

This is the binding constraint on RQ2 and it is a scheduling problem, not a metrics
problem — the reasons evaluation needs materially more alert history than exists now.
Collection continues to the fixed cut-off in §3.1, and whatever it yields is what gets
scored; a shortfall is reported there, with its consequences.

### 3.3 Models and baselines

| | Detecting disruptions | Identifying the reason |
| --- | --- | --- |
| **Baselines** | Persistence: the next window is disrupted if this one is | Most-common-cause; and the same LLM **without retrieval** (Chen et al., 2024) |
| **Compared** | XGBoost classifier on line-level window features; random forest on the same (Cottreau et al., 2025); LSTM if time allows (Boudabbous et al., 2026) | The LLM given the five most similar past alerts — the proposed RAG approach |

Baselines are written **before** the models they are compared against, as
`baseline.py` was, so no comparison can be retrofitted.

**The detectors cannot reuse the O2 features as they stand.** The delay regressor sees
one train at one stop; a detector sees one line over a 15-minute window. Its features
are aggregates over the windows up to the prediction time — for example the share of
services more than 5 minutes late, mean and maximum delay, services observed against
services timetabled, cancelled services, hour and peak, and the active-alert flag.
The cancelled-services count and the alert flag are `pd.NA` before their collection
began (2026-09-29 and 2026-09-15), never `0` or `False`. The label comes from the next 30 minutes, never from
the windows the features were built from. That matters because share-late is also
rule (b)'s input: computed over the same window as the label, it would *be* the label.

**Reinforcement learning — assessed, not used** *(the student's assessment of
2026-09-28, pending the supervisor's confirmation)*. The supervisor asked whether an
LLM or reinforcement learning works for the model. LLMs are used: the reasons agent,
and the no-retrieval baseline it is compared against. Reinforcement learning learns a
*policy* — which action to take — from rewards earned by acting in an environment. It
fits disruption **response**, where there are actions to choose (holding, short-running
or rescheduling trains). Detecting a disruption and naming its cause are prediction
problems with a known right answer for every past window, which is what supervised
learning is for. With ~6 distinct incidents there would also be far too few reward
signals to learn a policy from. RL for disruption response is recorded as future work.

### 3.4 Metrics

| Stage | Metric | What it tells us |
| --- | --- | --- |
| **Detection** | **Average precision** *(primary)* | Performance across every threshold; suits rare events, because many normal windows cannot inflate it (Cottreau et al., 2025) |
| | Precision, recall, F1 | At the chosen threshold |
| | **Lead time** | Minutes between the model's first flag and the operator's alert appearing. Positive means earlier. Alerts are polled every 30 minutes, so an alert's first appearance is known only to within 30 minutes, and lead time is reported at that resolution |
| | False alarms per day | Whether the detector is usable in practice (Tiong et al., 2025) |
| **Delay** *(supporting)* | MAE, RMSE, MASE | Error against naive persistence |
| | Interval coverage | Share of true delays inside the stated 90% interval |
| **Reasons** | **Macro-F1** *(primary)* | Every cause counts equally, so rare causes are not hidden by common ones (Chen et al., 2024) |
| | Cause accuracy | Share given the correct category |
| | Retrieval recall@5 | Share where at least one retrieved alert shares the true cause (Huang et al., 2026) |
| **Explanation** | Faithfulness rate | Share of statements supported by a retrieved alert or tool output (Papageorgiou et al., 2025) |
| | Citation coverage | Share of statements citing at least one retrieved alert (Huang et al., 2026) |

### 3.5 Experimental design

- **Same test dates for every model.** Nothing is tuned on them.
- **"Better" means** higher average precision for detection and higher macro-F1 for
  reasons, each with a 95% confidence interval from resampling **whole service dates**
  — not rows, which are correlated within a day.
- **Ablations:** reasons with and without retrieval; detection with and without the
  active-alert feature and the cancelled-services feature. **Label check:** rule (b)
  with and without cancelled and skipped-stop services counted as late, on the dates
  collected from 2026-09-29.
- **Leakage guards.** The incident's own alert is never shown to the reasons model —
  its `cause` *is* the answer — so retrieval must be **time-aware**: explaining an
  incident at time *T* may only see alerts first seen before *T*. Dates before alert
  collection began are marked unknown for the alert feature, never `False`.
- **Exclude by event, not by alert id.** The time filter alone still leaks. TfNSW
  republishes an incident with a widened scope under a **new** `entity.id` (§3.2 counts
  three such pairs among ten alerts), so when the later twin is explained, the earlier
  one passes the time filter carrying the same `cause`. Alerts are grouped into events
  before indexing, the whole event is excluded, and reasons are scored per event —
  otherwise twins also inflate the incident count. **Built:** `transit-alerts` groups
  them (`09-service-alerts.md` §7), and the index takes `seen_before` and
  `exclude_incident` as search arguments (`10-retrieval.md` §5).
- **Judge validation.** The author hand-labels a **20% stratified subsample** and
  reports **Cohen's κ** against the LLM judge. An unvalidated judge is an unvalidated
  instrument, and every explanation number depends on one. If κ < 0.6 the prompt is
  revised and the subsample re-scored — before the full run, not after seeing results.
- **Repeats.** 3 runs per system at fixed temperature; report mean ± sd. One run of a
  stochastic system is an anecdote.
- **Chunk size and k are frozen before scoring.** Both are swept once on a
  development subset and fixed before the test set is touched, exactly as the
  model's test split is scored once. A retrieval number is meaningless without the
  configuration that produced it, which is why every index carries an
  `IndexFingerprint` (`10-retrieval.md`) and `transit-index status` exits non-zero
  when the stored configuration and the current one disagree.
- **Determinism.** Live-condition questions are scored against *frozen* feed
  snapshots, never the live API — re-running a week later must reproduce the numbers.
  The realtime client already separates fetching from parsing, so this needs no new
  abstraction.

## 4. The prediction interval

RQ1 Objective 2 requires the prediction to state an error margin, and that margin has
to hold up.

**The margin cannot be the global MAE.** This was re-derived on 2026-10-05 with the
current pipeline, on the 2026-10-05 table. Validation is 09-27..30 and test is
10-01..04. This is the validation |residual|, by how late the train already was at
its previous stop:

| Already | Rows | MAE | 90th percentile |
| --- | --- | --- | --- |
| on time (≤ 60 s) | 42,689 | 9.7 s | 26.5 s |
| 1–5 min late | 10,937 | 29.5 s | 61.4 s |
| more than 5 min late | 2,121 | 89.9 s | 235.9 s |
| first stop of the trip | 3,675 | 61.4 s | 167.6 s |

This replaces the 7.8–82.4 s measured on the 7-day fit of 2026-09-10, which could not
be re-derived. It makes the same point more sharply: a nine-fold spread at the 90th
percentile.

The failure is worse than uneven: it is **anti-correlated with the question**. Nobody
asks whether their on-time train is on time. The tool is invoked when a rider suspects
a delay, which is exactly where a global margin covers worst. A metric that scored such
an answer "faithful" would certify the system's most misleading behaviour.

**Built: split conformal intervals, by line × lateness**
(`prediction/model/conformal.py`, run by `transit-train`). They are fitted on validation
and never on test. The score is the absolute residual. Within each bin, the margin is
the ⌈(n+1)(1−α)⌉-th smallest score, at 90%. The tool emits `prediction ± q̂`. The
half-widths are saved in the model artefact, so the agent tool states the margin that
was calibrated.

**The bins are not the ones this plan first named, and the reason is measured.** The
plan named hour band × peak × route. Hour band moves the 90th-percentile error only
1.7-fold (32–54 s), while lateness moves it nine-fold. The candidates were compared on
the validation split alone. Each was calibrated on two validation dates and checked on
the other two, both ways round:

| Bins | Coverage of trains already more than 5 min late | Median half-width |
| --- | --- | --- |
| One global margin | 64% / 48% | 40–51 s |
| Hour band × peak × line (the plan) | 61% / 49% | 42–46 s |
| **Line × lateness** | **89% / 75%** | **28 s** |
| The plan's bins + lateness | 86% / 67% | 31 s |

Line × lateness was chosen on that evidence, before the test split was looked at.
*Proposed 2026-10-05, pending confirmation (§5).* The plan's bins remain available as a
key.

**On the test dates**, in a development run on the 2026-10-05 table (test 10-01..04).
This is not the scored run, which comes after the cut-off:

| Already | Rows | Conformal | ± validation MAE |
| --- | --- | --- | --- |
| on time | 40,930 | 91.4% | 88.3% |
| 1–5 min late | 9,505 | 90.0% | 50.6% |
| more than 5 min late | 1,136 | 96.2% | **32.3%** |
| first stop | 3,430 | 91.4% | 63.5% |
| all | 55,001 | 91.3% | 79.1% |

A global margin covers a third of the trains riders ask about. The conformal interval
holds every band at 90% or above. Its half-widths run from ±26.5 s for a train on time
to ±292 s for a T1 train already more than five minutes late.

**Exchangeability is the residual risk, and it shows.** Rail delays are autocorrelated
within a day and drift between days. In the cross-fit, the same bins covered late
trains 89% of the time one way round and 75% the other. The 75% came from calibrating
on 27–28 Sep and checking on 29–30 Sep, which was the start of T4's late week
(`11-disruption-labels.md` §5). Binning mitigates drift; it does not remove it. State it.

## 5. To confirm with Dr Ramezani

**Agreed with her:** the 30-minute prediction horizon and the 15-minute window (§3.1).

**Decided by the student on 2026-09-28 and 2026-09-29, each needing her confirmation:**

1. **The disruption definition** (§3.2) — an unplanned alert timed by feed presence,
   with alerts present longer than 24 hours treated as standing notices; or a quarter
   of the line's services more than 5 minutes late. Rule (a) as first written labelled
   **100%** of windows disrupted, because no unplanned alert carries an end time.
2. **A fixed data cut-off** at the end of 2026-10-18 (§3.1), with any shortfall in the
   reasons evaluation reported with its consequences, rather than waiting for a target
   number of incidents.
3. **Weather stays** as the fourth cause group, with macro-F1 over the groups that
   occur (§3.2).
4. **"Other transport events" become future work** (§1).
5. **Reinforcement learning assessed and not used** (§3.3), in answer to her question
   whether an LLM or reinforcement learning works for the model.
6. **Which alerts are disruptions** is decided by their description as well as their
   cause (§3.2): planned-trackwork notices published as unknown cause are excluded, and
   unknown-cause closures kept. Republished alerts are grouped into one incident, whose
   cause is the first specific cause any of its alerts names.
7. **Cancellations are collected from 2026-09-29** and used as a check on rule (b) and
   as a detector feature, not in the main label (§3.2).

**Proposed on 2026-10-05, needing the student's and then her confirmation:**

9. **Rule (b) fires only on five or more observed services** (`11` §3). Five is the
   smallest count at which one late train cannot reach a quarter on its own. With no
   minimum, 26 of the 28 extra windows it marks are between 01:00 and 04:30, with one
   to four trains running.
10. **The prediction interval is binned by line × lateness, not hour band × peak ×
    line** (§4). On validation alone, the planned bins covered trains already more than
    5 minutes late 61% and 49% of the time. These bins covered them 89% and 75%, with
    a narrower interval.

**Still open:**

8. **The retrain window.** The 5 schedule-blind dates split the collection into
   2026-09-03..10 and 09-16..24. Current models exclude them, which puts the gap
   inside training and leaves validation and test clean and contiguous.

## 6. Threats to validity

| Threat | Mitigation |
| --- | --- |
| ~10 unplanned incidents is a very small reasons set | Group causes; report confidence intervals from resampling whole dates; fixed cut-off, with any shortfall reported with its consequences; state it as the headline limitation |
| Republished alerts look like separate incidents | Group alerts into events; exclude and score by event (§3.5) |
| Rule (b) cannot see a full closure | Stated; closures rest on rule (a) |
| Rule (b) omits cancelled and skipped-stop trains, which TfNSW counts as late | Collected from 2026-09-29; a check re-runs rule (b) counting them, on the dates covered. Before that the share is a lower bound on TfNSW's own lateness |
| TfNSW's threshold is judged at the destination; rule (b) judges each window | Stated as an adaptation (§3.2) |
| The 24-hour cap would misfile a closure longer than a day | Cap sits in a wide empty gap on current data; any incident exceeding it is reported, not relabelled |
| Lead time is only known to within the 30-minute alert poll | Reported at that resolution |
| Author wrote both system and evaluation | Ground truth fixed at authoring time; judge validated against a human subsample; adversarial cases written to fail |
| 22 service dates is a short window | Stated; no seasonal or incident-period claims made |
| 5 dates have no timetable join | Excluded before the split, disclosed, and shown not to move the headline (`07-training-table.md`) |
| Validation is only 2 dates, one a disruption day | Disclosed in `07-training-table.md`; widens as collection continues |
| `RTTA_*` movements are uncollected | ~40% of overnight trips are out-of-service and correctly excluded, but a service altered beyond timetable expression may also land there. **Measure it and state the number** |
| No alert-flag feature yet | Alerts began after delays, so the flag would be `False` across the back-catalogue. Rows predating alert collection get `pd.NA`, never `False` |
| Conformal assumes exchangeability | Class-conditional binning mitigates; state the residual |

## 7. What this commits to building

In dependency order.

1. ✅ **Alert corpus ingestion** — alerts to incidents, T1/T4 only
   (`transit-alerts`, 2026-09-29)
2. ✅ **Alert retrieval index** — one passage per incident in `tfnsw_alerts`
3. ✅ **Time-aware retrieval** — the §3.5 leakage guards, time *and* incident
4. **Realtime tools.** ✅ They exist over frozen snapshots, as of a moment
   (`14-agent.md`, 2026-10-05), including the delay tool, which serves each
   prediction with its 90% interval. The live feed behind them is next.
5. ✅ **The agent loop**: `transit-ask` (`14-agent.md`, 2026-10-05). The rules the
   evaluation checks are in the system prompt from the first version, never bolted on
6. ✅ **Conformal calibration** on the validation split. `transit-train` fits it
   beside the model, by line × lateness (§4, 2026-10-05)
7. **The disruption labeller and classifier.** ✅ The labeller: `transit-label`
   (`11-disruption-labels.md`, 2026-10-05), with thresholds as parameters. ✅ The
   line-level window features, the 30-minute target and the persistence baseline:
   `transit-detect` (`12-disruption-detection.md`). ✅ The detectors, XGBoost and
   random forest, chosen on validation, with the no-alert ablation (`12` §5). Their
   scored comparison waits for the split drawn at the cut-off
8. **The evaluation harness** — detection metrics, cause macro-F1, the judge, then
   judge validation, then the scored run on data to the §3.1 cut-off

Items 6 and 7 do not depend on 1–5 and can proceed in parallel.
