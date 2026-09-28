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
cancellations and skipped stops (**not collected**: the collector keeps only
`SCHEDULED` stop calls, `realtime/parsing.py`), station and accessibility notices
(collected as alerts), or crowding and special-event services (not collected). RQ1
covers service disruptions and delays; the write-up names these and says which the
collected data could support.

## 2. RQ1 — the workflow, stage by stage

The objectives are the stages, in the order the workflow runs. An orchestrator agent
(the project's own Claude loop) calls each as an MCP tool and writes the final answer.

| | Task and tool | In → out | How it is checked | State |
| --- | --- | --- | --- | --- |
| **O1** Retrieval | Live T1/T4 trip updates and service alerts, joined to the static timetable. **Deterministic Python, not an LLM** | Trip updates (120 s), alerts (30 min), timetable → per-stop delay observations; active alerts with cause, effect, scope, text | Tool-faithfulness: share of the agent's statements about live conditions matching the logged tool response. Feed currency: lag between tool timestamp and snapshot | **Working** |
| **O2** Prediction | Is T1/T4 disrupted, or about to be, and by how much. **XGBoost inside the tool handler** | *Delay half, per stop event:* the nine features in `dataset.FEATURE_COLUMNS` — scheduled arrival, stop sequence, previous-stop delay, hour, day, weekend, peak, line, stop — plus an active-alert flag *(once overlap allows)* → expected delay with a 90% interval. *Disruption half, per line × 15-minute window:* line-level window features (§3.3) → probability the line is disrupted in the next 30 minutes, and a flag | The answer states the interval the tool returned, unnarrowed. Calibration: a 90% interval contains the truth ~90% of the time on held-out days | **Delay half working; disruption half not built** |
| **O3** Reasons | Likely cause of a predicted or detected disruption. **Retrieval agent over past alerts + Claude writing the reason** | O2's prediction with context, plus the five most similar past alerts → a cause category and a one/two-sentence explanation citing alert ids | Faithfulness: every statement supported by a retrieved alert or a tool output (Papageorgiou et al., 2025) | **Not built** |

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

**The disruption output's definition is decided (§3.2), pending the supervisor's
confirmation.** Its thresholds are parameters, so the labeller can be built now and
re-run unchanged if she moves one.

**O3 — not built, and the blocker is narrow.** The retrieval stack is proven end to
end against real Voyage embeddings (`10-retrieval.md`), but it indexes the Opal PDFs.
What changes is the corpus, not the machinery: nothing yet reads `service_alerts` back
out, and an alert has no page number, so the citation invariant needs a locator rather
than a page.

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

(a) an **unplanned** service alert — any cause other than `MAINTENANCE` — is in the
feed for the line during the window, timed by **feed presence**
(`first_seen_utc`..`last_seen_utc`), not by `active_period`; **and** that alert's feed
presence is no longer than **24 hours**. Anything longer is a standing notice, not an
incident. Or,
(b) at least a quarter of the line's observed services are **more than 5 minutes
late**.

The 5-minute threshold is Transport for NSW's own on-time-running definition, and it
is used alone. A stricter rider's standard (3 minutes) was considered and not
adopted: it is a judgement where 5 minutes is a citation, and it moves the label from
"the line is disrupted" towards ordinary lateness. The one-quarter share is a
modelling choice. Rule (b) catches disruptions the operator never posted an alert for.

**Two limits of the rules, stated rather than discovered later.**

- **Rule (b) cannot see a full closure.** With no trains running there are no services
  to be late, so a closure is caught by rule (a) or not at all.
- **The 24-hour cap would misfile a closure longer than a day** — storm or flood
  damage, say — as a notice. The cap sits in a wide empty gap on the history so far
  (incidents run 0–242 minutes; the Tangara notice below runs 7,752), so no current
  alert lands near it. If a genuine incident ever exceeds it, that is reported as a
  known misclassification, not quietly relabelled.

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

**Reason (ground truth).** The `cause` field of the alert. Planned trackwork is
excluded because it is scheduled, not predicted. Windows flagged only by rule (b) have
no recorded cause, so they count for detection but not for reasons.

**Cause categories.** Grouped so each has enough examples. Counts below are distinct
alerts touching T1 or T4 over 2026-09-14..24:

| Group | Feed causes | n |
| --- | --- | --- |
| Technical / infrastructure | `TECHNICAL_PROBLEM` | 6 |
| Incident on the network | `ACCIDENT` 1, `POLICE_ACTIVITY` 2, `MEDICAL_EMERGENCY` 0 | 3 |
| Weather or external | `WEATHER`, etc. | **0** |
| Other or unknown | `OTHER_CAUSE` 1, `UNKNOWN_CAUSE` 7 | 8 |
| *(excluded — planned)* | `MAINTENANCE` | *23* |
| **Total alerts touching T1/T4** | | **40** |

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

**Ten unplanned alerts touch T1/T4 in the 10 days of history — but they are not one
a day, and they are not ten incidents.**

| | |
| --- | --- |
| Sydney dates carrying any | **3** — 09-15 (4), 09-21 (4), 09-22 (2). Seven of ten dates have none |
| Lines | T1 6, T4 5 — these overlap; one alert is scoped to both |
| Distinct operational events | **~6.** TfNSW re-publishes an alert with a widened scope under a new `entity.id`, because the id is a content-derived UUIDv5. Three pairs here are the same event twice, one pair byte-identical in `description_text` |

**Two consequences the rest of this plan has to absorb.**

*The date-level bootstrap in §3.5 is degenerate.* Resampling whole service dates with
incidents on three of them means most resamples contain none, so a 95% interval on
cause macro-F1 is not meaningful at this sample size.

*The reasons test set may be empty.* A chronological 70/15/15 over the 22 collected
dates puts test at 09-22..24, which holds **2** unplanned alerts, both
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
services timetabled, hour and peak, and the active-alert flag (`pd.NA` before alert
collection began, never `False`). The label comes from the next 30 minutes, never from
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
  active-alert feature.
- **Leakage guards.** The incident's own alert is never shown to the reasons model —
  its `cause` *is* the answer — so retrieval must be **time-aware**: explaining an
  incident at time *T* may only see alerts first seen before *T*. Dates before alert
  collection began are marked unknown for the alert feature, never `False`.
- **Exclude by event, not by alert id.** The time filter alone still leaks. TfNSW
  republishes an incident with a widened scope under a **new** `entity.id` (§3.2 counts
  three such pairs among ten alerts), so when the later twin is explained, the earlier
  one passes the time filter carrying the same `cause`. Alerts are grouped into events
  before indexing, the whole event is excluded, and reasons are scored per event —
  otherwise twins also inflate the incident count.
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

## 4. The prediction interval — carried forward from the old plan

RQ1 Objective 2 requires the prediction to state an error margin and for that margin
to hold up. The work behind this is real and survives the restructure.

**The margin cannot be the global MAE.** Measured on the **7-day fit of 2026-09-10**
— 14,669 rows, not the current model — conditional error spanned roughly **7.8 s to
82.4 s**, a factor of ten, between trains running to time and trains already late. A single global margin is over-conservative on the first and
badly overconfident on the second.

The failure is worse than uneven: it is **anti-correlated with the question**. Nobody
asks whether their on-time train is on time. The tool is invoked when a rider suspects
a delay — exactly the rows a global margin covers worst. A metric scoring such an
answer "faithful" would certify the system's most misleading behaviour.

> **Neither figure has been re-derived against the corrected 17-date model**, and the
> segment/coverage table they came from was removed with the old decomposition, so its
> n's and percentages are not recoverable from this repo. Re-deriving is the only path,
> and it has to happen before any of this is quoted. The *argument* does not depend on
> the exact numbers — it depends on conditional error varying by an order of magnitude,
> which is a property of the data, not of that fit.

**The fix: split conformal intervals**, fitted on validation and never on test —
absolute residuals, the ⌈(n+1)(1−α)⌉/n quantile, emit `prediction ± q̂`. It is
distribution-free and assumes only exchangeability. Because conditional coverage is
the whole problem, use **Mondrian (class-conditional) conformal**: residual quantiles
within bins of hour-band × peak × route, so a 22:00 prediction carries a wider bound
than an 06:00 one by construction. Report coverage marginally *and* per bin — the
per-bin table is the evidence the fix worked.

Rail delays are autocorrelated within a day, so exchangeability is imperfect;
class-conditional binning mitigates but does not remove this. State it.

## 5. To confirm with Dr Ramezani

**Agreed with her:** the 30-minute prediction horizon and the 15-minute window (§3.1).

**Decided by the student on 2026-09-28, each needing her confirmation:**

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

**Still open:**

6. **The retrain window.** The 5 schedule-blind dates split the collection into
   2026-09-03..10 and 09-16..24. Current models exclude them, which puts the gap
   inside training and leaves validation and test clean and contiguous.

## 6. Threats to validity

| Threat | Mitigation |
| --- | --- |
| ~10 unplanned incidents is a very small reasons set | Group causes; report confidence intervals from resampling whole dates; fixed cut-off, with any shortfall reported with its consequences; state it as the headline limitation |
| Republished alerts look like separate incidents | Group alerts into events; exclude and score by event (§3.5) |
| Rule (b) cannot see a full closure | Stated; closures rest on rule (a) |
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

1. **Alert corpus ingestion** — `service_alerts` → cited chunks, filtered to T1/T4
   *(blocks O3)*
2. **Alert retrieval index** — reuse the existing Voyage/Chroma stack *(blocks O3)*
3. **Time-aware retrieval** — the §3.5 leakage guards, time *and* event *(needs the
   alerts grouped into events first)*
4. **Realtime tools** over the existing client *(blocks O1 as an agent tool)*
5. **The agent loop** — the margin requirement is in the system prompt from the first
   version, never bolted on, or O2's check measures a retrofit
6. **Conformal calibration** on the validation split *(blocks O2's interval)*
7. **The disruption labeller and classifier** — definition decided (§3.2), thresholds
   as parameters; line-level window features (§3.3) *(blocks RQ2 detection)*
8. **The evaluation harness** — detection metrics, cause macro-F1, the judge, then
   judge validation, then the scored run on data to the §3.1 cut-off

Items 6 and 7 do not depend on 1–5 and can proceed in parallel.
