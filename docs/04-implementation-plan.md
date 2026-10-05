# Implementation Plan

> One semester, one engineer, 41029 + 41030 concurrently. The plan is written around
> what can be cut, because something will be.

## Status — 29 September 2026

| Workstream | State |
| --- | --- |
| Literature review | **Done** — Task 1 complete |
| Problem definition + methodology | **Superseded** — the 24 August notes predate the 2026-09-22 restructure (Opal fares, a rider Q&A assistant, and a collection start of 2026-08-23 where the real one is 2026-09-03). The problem statement is rewritten in the Task 3 proposal rather than kept as a separate document |
| Research questions | **Done** — RQ1 feasibility, RQ2 evaluation, set by the supervisor 2026-09-22. Scope narrowed to T1/T4 service disruptions; Opal fare policy dropped. Academic record: `RQ_List_and_Evaluation_Methods_v3.docx` (2026-09-28) |
| Data source selection | **Done** — see [`03-data-sources.md`](./03-data-sources.md); its §4 recommendation is superseded by the scope change |
| Tech stack | **Done** — see [`02-tech-stack.md`](./02-tech-stack.md) |
| Repo + CI + module layout | **Done** — this scaffold |
| Scheduled collection workflow | **Done** — `.github/workflows/collect.yml`, now the backup |
| System architecture diagram | **Done** — [`01-architecture.md`](./01-architecture.md) §1 |
| **Delay collection running** | **Done** — live since 3 September 2026 on the always-on collector, 120 s cadence, T1 + T4 ([`06-always-on-collector.md`](./06-always-on-collector.md)) |
| Service Alerts collection | **Done** — `transit-poller` polls alerts every 30 min into `service_alerts`/`alert_scopes` ([`09-service-alerts.md`](./09-service-alerts.md)). The training-table flag waits for an overlap window |
| Reconciliation → training table | **Done** — `transit-reconcile`, schema in [`07-training-table.md`](./07-training-table.md) |
| Prediction: delay regression | **Done** — `transit-train`; naive persistence written first. Test MAE **15.45 s** vs baseline **18.67 s**, **MASE 0.828**, over the 17 schedule-covered dates to 2026-09-24. **Margin:** a 90% split conformal interval by line × how late the train already is, calibrated on validation ([`08`](./08-evaluation-plan.md) §4). In a development run on the 2026-10-05 table it holds every lateness band at 90% or above, where ± MAE covers 32% of trains already more than 5 minutes late |
| Prediction: disruption classification | **Label, baseline and detectors built; scored run waits for the cut-off.** Added to scope on 2026-09-22 for RQ2. `transit-label` builds the ground truth ([`11-disruption-labels.md`](./11-disruption-labels.md)). It marks 4.7% of T1 windows and 4.4% of T4 windows disrupted to 2026-10-04. `transit-detect` builds what a detector may see and scores persistence, which was written first ([`12-disruption-detection.md`](./12-disruption-detection.md)). XGBoost and a random forest are chosen on validation, with an ablation without the alert feature (`12` §5). Without the operator's alert, recall on alert-defined targets falls from about 70% to about 25%. Today's test split holds 7 positive targets, too few to compare anything |
| Timetable bundle archive | **Done** — daily on the VM; `bundles.discover()` finds every era, so reconcile needs no flag. 2026-09-11..15 predate it and are unrecoverable |
| Corpus ingestion | **Done for PDFs, retired from the RQs** — the three Opal PDFs pinned and chunked into 144 cited passages. Retained as code; not the corpus any more |
| **Alert corpus ingestion** | **Done** — `transit-alerts audit` decides which alerts are disruptions and groups republications into incidents ([`09-service-alerts.md`](./09-service-alerts.md) §7): 8 incidents on 5 dates to 2026-09-28, and 12 on 8 to 2026-10-04. The rule is checked on alerts from 2026-09-29, which it was not written from. To 2026-10-04 it agrees with 8 of 8, but none was `UNKNOWN_CAUSE`, so the description markers are not yet tested out of sample. One urgent repair published as `MAINTENANCE` is in by override |
| Retrieval index | **Done** — `transit-index`, Voyage embeddings into fingerprinted Chroma collections ([`10-retrieval.md`](./10-retrieval.md)). `tfnsw_alerts`: 8 incident passages, `voyage-4-lite`, built 2026-09-28, with time-aware retrieval that excludes the incident being explained. `opal_policy` (144 chunks) is retired |
| Reasons stage (RQ1 O3) | **Built** — `transit-reasons` ([`13-reasons.md`](./13-reasons.md)). It reads a situation from the delay feed (never the incident's own alert), retrieves past incidents under both leakage guards, and returns a structured answer whose citations are checked. It is compared with a time-aware most-common-cause baseline and with the same model without retrieval. The development run is not a result, and the scored run waits for the cut-off |
| Agent loop + MCP tools | **Loop built; MCP server not started.** `transit-ask` ([`14-agent.md`](./14-agent.md)) is a hand-rolled tool-use loop over three tools: line status, disruption risk and past incidents. A fourth tool gives each running train's expected delay with its 90% interval, stated unnarrowed as the hard requirement asks. They answer as of a moment in a frozen snapshot, and every call is recorded verbatim for scoring. Still to come: the MCP server, and the live feed behind the tools |
| Evaluation plan | **Updated 2026-09-28** — [`08-evaluation-plan.md`](./08-evaluation-plan.md) follows RQ1's objectives and RQ2's metrics, and now records the student's decisions on the open items, which await the supervisor's confirmation (§5) |
| Evaluation harness | **In progress.** Built: detection metrics with lead time, cause macro-F1 and retrieval recall, and 95% intervals from resampling whole service dates ([`12`](./12-disruption-detection.md), [`13`](./13-reasons.md)). Re-runs reproduce byte for byte. The faithfulness judge scores every statement against the evidence its system was shown ([`15`](./15-faithfulness-judge.md)). Its rubric is frozen, and its Cohen's κ validation awaits the author's labels on a blind sample. The k sweep compares retrieval with chance ([`13`](./13-reasons.md) §7). k is frozen at the cut-off. Still to come: the scored runs at the cut-off |

## 1. The one thing that cannot be caught up

Every other task on this list can absorb a slip by being done faster or smaller
later. Collection cannot. There is no historical bus/train GTFS-Realtime archive to
fall back on ([`02-tech-stack.md`](./02-tech-stack.md) §3), so the training set is
exactly what gets collected between the day the poller starts and the day the model
is needed — and not a row more.

Collection has been running since 3 September 2026 and is no longer the bottleneck.
What matters now is that it keeps running: the risk has moved from *starting* to
*silently stopping*, which is what the weekly `--status` check exists to catch.

**Consequence for the fallback.** The Week-6 checkpoint below exists to decide
between a live-trained model and a methodological feasibility study benchmarked
against published results. On the volume collected so far that gate looks passed —
but it should be *stated* from a row count at the meeting, not assumed.

## 2. Phases

### Weeks 1–3 — Research preparation ✅
Literature review, dataset selection, supervisor onboarding, stack decisions.

### Weeks 4–7 — Core build (**current**)
Ordered by what unblocks what:

1. **Collection.** ✅ Done — key, endpoint confirmation, route lookup, live polling and
   always-on scheduling. Keep checking it; do not let it stop.
2. **Ingestion + retrieval.** ✅ Machinery done, ⚠️ corpus wrong. The three Opal PDFs
   became 144 chunks carrying document and page, embedded into a persisted Chroma
   collection against the real Voyage model. Citation metadata is an ingestion contract
   rather than an afterthought, and the retriever re-checks it on the way out because
   an untyped database sits in between. Remaining, for RQ1 Objective 3: read
   `service_alerts` into cited chunks, generalise the page-based citation to a locator
   an alert can satisfy, rebuild, then sweep chunk size and k
   ([`10-retrieval.md`](./10-retrieval.md)).
3. **Realtime tools.** Trip Update and Alerts wrapped as MCP tools over the existing
   `realtime/` client, joined against the static bundle for human-readable stop and
   route names.
4. **Agent loop.** The hand-rolled tool-use loop, with the error-margin requirement
   built into the system prompt from the first version so it is never bolted on.

### Weeks 8–10 — Evaluation harness (**protect this**)
This answers RQ2. If Weeks 4–7 overrun, scope comes out of the *system*, not here.

- **Detection** — average precision, precision/recall/F1, lead time, false alarms per
  day, against a persistence baseline.
- **Reasons** — cause macro-F1 and retrieval recall@5, against most-common-cause and
  against the same LLM with no retrieval.
- **Explanations** — faithfulness rate and citation coverage, judged per statement,
  with the judge itself validated by Cohen's κ on a hand-labelled 20% subsample.
- **Delay** — MAE/RMSE/MASE and interval coverage, supporting rather than headline.

**Week 6 checkpoint — passed for delay, not for reasons.** 321,634 stop events over
22 service dates as of 2026-09-24 comfortably support a chronological split, so no
feasibility-study fallback is needed for the delay model. The reasons side has not
passed: 10 unplanned alerts touch T1/T4, they fall on **3 Sydney dates**, and the
current test dates hold 2 of one cause group — so cause macro-F1 is not yet
computable. Collection continues to a fixed cut-off at the end of 2026-10-18, and any
shortfall is reported rather than waited out; see [`08`](./08-evaluation-plan.md) §3.1.

### Weeks 11–13 — Polish and write-up
MCP integration finished, final report, architecture diagram, evaluation numbers.
Phase 2 deployment happens **after** grading.

| Date | What |
| --- | --- |
| 2026-10-18 | **Data cut-off** — the last service date that enters a scored result |
| 2026-10-25 | 41029 Task 2 (Low-risk Student Project Application), Task 3 (Research Project Proposal), Reflection |
| 2026-11-02 | 41030 Final Report and Project Video |

## 3. Risks

| Risk | Impact | Mitigation |
| --- | --- | --- |
| **Collection silently failing** | Fatal to the prediction layer | `--status` after every scheduling change; check `poll_log` weekly. Now the live risk, since collection has started |
| Static bundle and feed from different versions | Collects nothing, looks fine | `transit-poller --probe` names this case explicitly; run it before any long collection |
| Laptop uptime | Gaps in the training window | Mitigated — collection runs off the laptop on an always-on box, with the scheduled Action as backup ([`06-always-on-collector.md`](./06-always-on-collector.md)) |
| Weeks 4–7 overrun into the harness | Loses the highest-value phase | Cut system scope, not harness scope. Trip Planner API is the escape hatch for GTFS joins |
| Anthropic spend overrun | Budget | Console spend limit; Haiku for judging; cache eval-set embeddings |
| The 2026-09-28 decisions not confirmed | Relabelling late | The thresholds are labeller parameters, so a change is a re-run; raise them at the next supervisor meeting |
| Supervisor's expertise is a step from RAG specifics | Unreviewed technical choices | Every non-obvious decision is written down with its rejected alternative ([`02-tech-stack.md`](./02-tech-stack.md) §6) |

## 4. Outstanding supervisor items

Agreed with her: the 30-minute horizon and the 15-minute window.

1. **Confirm the student's decisions of 2026-09-28** ([`08`](./08-evaluation-plan.md)
   §5): the disruption definition (feed-presence timing, a 24-hour cap for standing
   notices, a quarter of services more than 5 minutes late); the data cut-off at the
   end of 2026-10-18; weather kept as a cause group; "other transport events" as
   future work; and reinforcement learning assessed and not used. Bring the measured
   finding behind the first: rule (a) as first written labels **100% of windows
   disrupted**, because no unplanned alert carries an `active_period_end`.
2. **The retrain window** — 2026-09-11..15 have no timetable era and are excluded,
   splitting collection into 09-03..10 and 09-16..24. Current models exclude them,
   which leaves validation and test clean and contiguous.
3. **The prediction interval.** It is built: a 90% conformal interval, re-derived on
   the 2026-10-05 table (`08` §4). Validation error varies nine-fold with how late the
   train already is, and ± MAE covers 32% of trains already more than 5 minutes late.
   The interval's bins are line × lateness, not the planned hour band × peak × line.
   On validation alone the planned bins covered those trains about 50% of the time,
   and these bins covered them 75–89%. That change needs her confirmation.
