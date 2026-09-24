# Research Questions and the Evaluation Plan

> Restructured 2026-09-22 on Dr Ramezani's direction: two research questions, trains
> only, service disruptions on T1 and T4. The old RQ1–RQ3 became RQ1's objectives and
> the old RQ4 became RQ2's. **Opal fare policy is out**; RAG stays, and now retrieves
> past service alerts to explain a disruption instead of retrieving fare documents.
>
> The academic source of truth is `RQ_List_and_Evaluation_Methods_v2.docx`, one level
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

## 2. RQ1 — the workflow, stage by stage

The objectives are the stages, in the order the workflow runs. An orchestrator agent
(the project's own Claude loop) calls each as an MCP tool and writes the final answer.

| | Task and tool | In → out | How it is checked | State |
| --- | --- | --- | --- | --- |
| **O1** Retrieval | Live T1/T4 trip updates and service alerts, joined to the static timetable. **Deterministic Python, not an LLM** | Trip updates (120 s), alerts (30 min), timetable → per-stop delay observations; active alerts with cause, effect, scope, text | Tool-faithfulness: share of the agent's statements about live conditions matching the logged tool response. Feed currency: lag between tool timestamp and snapshot | **Working** |
| **O2** Prediction | Is T1/T4 disrupted, or about to be, and by how much. **XGBoost inside the tool handler** | The nine features in `dataset.FEATURE_COLUMNS` — scheduled arrival, stop sequence, previous-stop delay, hour, day, weekend, peak, line, stop — plus an active-alert flag *(once overlap allows)* → disruption probability over a 30-minute horizon, a flag, and expected delay with a 90% interval | The answer states the interval the tool returned, unnarrowed. Calibration: a 90% interval contains the truth ~90% of the time on held-out days | **Delay half working; disruption half not built** |
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

**The disruption output needs §3.2 agreed before anything can be built.**

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

- **Unit of observation:** one line (T1 or T4) in one 15-minute window. *Proposed.*
- **Split:** chronological by whole service date, 70/15/15, never random. A shuffle
  puts the same afternoon on both sides and invalidates the baseline comparison.
- Thresholds and settings are chosen on validation; the test dates are scored once.

### 3.2 What counts as a disruption, and as its reason

**This needs the supervisor's agreement before any model is scored.**

**Disruption (proposal).** A window is disrupted if either

(a) an **unplanned** service alert is active for the line, or
(b) at least a quarter of the line's observed services are more than 5 minutes late.

The 5-minute threshold is settled — it is Transport for NSW's own on-time-running
definition. The one-quarter share is a modelling choice, fixed before the test dates
are scored. Rule (b) catches disruptions the operator never posted an alert for.

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
class explicitly — not outside the table. The blockquote below is why it cannot simply
be dropped.

> **The weather group is empty, and macro-F1 over an empty class is undefined.**
> Either collapse to three groups or state explicitly that weather may stay empty and
> report macro-F1 over the classes that occur. Decide before scoring, not after.

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
Collection continues; the decision of when there is enough belongs in §5.

### 3.3 Models and baselines

| | Detecting disruptions | Identifying the reason |
| --- | --- | --- |
| **Baselines** | Persistence: the next window is disrupted if this one is | Most-common-cause; and the same LLM **without retrieval** (Chen et al., 2024) |
| **Compared** | XGBoost classifier on the O2 features; random forest on the same (Cottreau et al., 2025); LSTM if time allows (Boudabbous et al., 2026) | The LLM given the five most similar past alerts — the proposed RAG approach |

Baselines are written **before** the models they are compared against, as
`baseline.py` was, so no comparison can be retrofitted.

### 3.4 Metrics

| Stage | Metric | What it tells us |
| --- | --- | --- |
| **Detection** | **Average precision** *(primary)* | Performance across every threshold; suits rare events, because many normal windows cannot inflate it (Cottreau et al., 2025) |
| | Precision, recall, F1 | At the chosen threshold |
| | **Lead time** | Minutes between the model's first flag and the operator's alert appearing. Positive means earlier |
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

1. **The disruption definition** (§3.2) — rule (a) as written labels **100%** of
   windows disrupted, because no unplanned alert carries an end time. The proposed fix
   is to bound an incident by feed presence (`first_seen_utc`..`last_seen_utc`) with a
   duration cap, rather than by `active_period` or by excluding causes.
2. **When there is enough alert history to evaluate reasons at all.** On the current
   test dates the reasons set holds 2 alerts of one cause group, so cause macro-F1 is
   not computable and the date-level bootstrap is degenerate. This is the decision that
   most affects the timetable for RQ2.
3. **The empty weather class** (§3.2) — collapse to three groups, or report macro-F1
   over occurring classes only.
4. **"Other transport events"** in RQ1 — do these stay, or become future work, given
   the two-line scope?
5. **The retrain window.** The 5 schedule-blind dates split the collection into
   2026-09-03..10 and 09-16..24. Current models exclude them, which puts the gap
   inside training and leaves validation and test clean and contiguous.
6. **The 30-minute prediction horizon** named in the RQ doc as "proposed". §3.2's
   window label cannot be specified without it.

## 6. Threats to validity

| Threat | Mitigation |
| --- | --- |
| ~10 unplanned incidents is a very small reasons set | Group causes; report confidence intervals from resampling whole dates; state it as the headline limitation |
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
3. **Time-aware retrieval** — the §3.5 leakage guard *(gated on §3.2)*
4. **Realtime tools** over the existing client *(blocks O1 as an agent tool)*
5. **The agent loop** — the margin requirement is in the system prompt from the first
   version, never bolted on, or O2's check measures a retrofit
6. **Conformal calibration** on the validation split *(blocks O2's interval)*
7. **The disruption labeller and classifier** *(gated on §3.2; blocks RQ2 detection)*
8. **The evaluation harness** — detection metrics, cause macro-F1, the judge, then
   judge validation, then the scored run

Items 6 and the labelling design do not depend on 1–5 and can proceed in parallel.
