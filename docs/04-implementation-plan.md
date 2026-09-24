# Implementation Plan

> One semester, one engineer, 41029 + 41030 concurrently. The plan is written around
> what can be cut, because something will be.

## Status — 24 September 2026

| Workstream | State |
| --- | --- |
| Literature review | **Done** — 11 references, 3 from 2026, three thematic clusters |
| Problem definition + methodology | **Done** — written up, pending supervisor sign-off |
| Research questions | **Done** — RQ1 feasibility, RQ2 evaluation, set by the supervisor 2026-09-22. Scope narrowed to T1/T4 service disruptions; Opal fare policy dropped |
| Data source selection | **Done** — see [`03-data-sources.md`](./03-data-sources.md); its §4 recommendation is superseded by the scope change |
| Tech stack | **Done** — see [`02-tech-stack.md`](./02-tech-stack.md) |
| Repo + CI + module layout | **Done** — this scaffold |
| Scheduled collection workflow | **Done** — `.github/workflows/collect.yml`, now the backup |
| System architecture diagram | **Done** — [`01-architecture.md`](./01-architecture.md) §1 |
| **Delay collection running** | **Done** — live since 3 September 2026 on the always-on collector, 120 s cadence, T1 + T4 ([`06-always-on-collector.md`](./06-always-on-collector.md)) |
| Service Alerts collection | **Done** — `transit-poller` polls alerts every 30 min into `service_alerts`/`alert_scopes` ([`09-service-alerts.md`](./09-service-alerts.md)). The training-table flag waits for an overlap window |
| Reconciliation → training table | **Done** — `transit-reconcile`, schema in [`07-training-table.md`](./07-training-table.md) |
| Prediction: delay regression | **Done** — `transit-train`; naive persistence written first. Test MAE **15.45 s** vs baseline **18.67 s**, **MASE 0.828**, over the 17 schedule-covered dates to 2026-09-24 |
| Prediction: disruption classification | **Not started** — added to scope 2026-09-22 for RQ2. Gated on the disruption definition in [`08-evaluation-plan.md`](./08-evaluation-plan.md) §3.2 being agreed |
| Timetable bundle archive | **Done** — daily on the VM; `bundles.discover()` finds every era, so reconcile needs no flag. 2026-09-11..15 predate it and are unrecoverable |
| Corpus ingestion | **Done for PDFs, retired from the RQs** — the three Opal PDFs pinned and chunked into 144 cited passages. Retained as code; not the corpus any more |
| **Alert corpus ingestion** | **Not started** — nothing reads `service_alerts` back out. Blocks RQ1 Objective 3 |
| Retrieval index | **Done, wrong corpus** — `transit-index`, Voyage embeddings into a fingerprinted Chroma collection ([`10-retrieval.md`](./10-retrieval.md)). Built and real: `opal_policy`, 144 chunks, `voyage-4-lite`, 2026-09-17. Repointing it at alerts is the next build step |
| Agent loop + MCP tools | Not started |
| Evaluation plan | **Rewritten 2026-09-24** — [`08-evaluation-plan.md`](./08-evaluation-plan.md) now follows RQ1's objectives and RQ2's metrics. Two items need the supervisor: the disruption definition (§3.2) and the empty weather cause class |
| Evaluation harness | Not started |

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
computable. Collection continues; see [`08`](./08-evaluation-plan.md) §3.2.

### Weeks 11–13 — Polish and write-up
MCP integration finished, final report, architecture diagram, evaluation numbers.
Phase 2 deployment happens **after** grading.

## 3. Risks

| Risk | Impact | Mitigation |
| --- | --- | --- |
| **Collection silently failing** | Fatal to the prediction layer | `--status` after every scheduling change; check `poll_log` weekly. Now the live risk, since collection has started |
| Static bundle and feed from different versions | Collects nothing, looks fine | `transit-poller --probe` names this case explicitly; run it before any long collection |
| Laptop uptime | Gaps in the training window | Mitigated — collection runs off the laptop on an always-on box, with the scheduled Action as backup ([`06-always-on-collector.md`](./06-always-on-collector.md)) |
| Weeks 4–7 overrun into the harness | Loses the highest-value phase | Cut system scope, not harness scope. Trip Planner API is the escape hatch for GTFS joins |
| Anthropic spend overrun | Budget | Console spend limit; Haiku for judging; cache eval-set embeddings |
| RQ or methodology not signed off | Rework late | Both are flagged pending; raise at the next supervisor meeting |
| Supervisor's expertise is a step from RAG specifics | Unreviewed technical choices | Every non-obvious decision is written down with its rejected alternative ([`02-tech-stack.md`](./02-tech-stack.md) §6) |

## 4. Outstanding supervisor items

1. **The disruption definition** ([`08`](./08-evaluation-plan.md) §3.2) — the alert
   rule and the one-quarter late-services share. Nothing can be scored until this is
   agreed. Bring the measured finding: rule (a) as written labels **100% of windows
   disrupted**, because no unplanned alert carries an `active_period_end`. Bound the
   incident by feed presence instead.
2. **The empty weather cause class** (§3.2) — collapse the four groups to three, or
   report macro-F1 over occurring classes only.
3. **"Other transport events"** in RQ1 — in scope, or future work?
4. **The retrain window** — 2026-09-11..15 have no timetable era and are excluded,
   splitting collection into 09-03..10 and 09-16..24. Current models exclude them,
   which leaves validation and test clean and contiguous.
5. **The prediction interval** — the margin moves from "consistent with global MAE" to
   a conditional conformal interval, because conditional error spans roughly
   7.8–82.4 s and a global margin is worst exactly where the tool is asked. Needs
   re-deriving against the corrected model before it is quoted.
6. **Sign off the problem definition and methodology** — still outstanding, and its
   stated collection start date (2026-08-23) is wrong; the real start is 2026-09-03.
7. **Task 1 word count** — the expanded review runs past the original 800–1000 word
   cap. Confirm the limit before final submission.
