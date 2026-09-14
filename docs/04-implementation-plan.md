# Implementation Plan

> One semester, one engineer, 41029 + 41030 concurrently. The plan is written around
> what can be cut, because something will be.

## Status — 14 September 2026

| Workstream | State |
| --- | --- |
| Literature review | **Done** — 11 references, 3 from 2026, three thematic clusters |
| Problem definition + methodology | **Done** — written up, pending supervisor sign-off |
| Research question | **Merged into one**, pending supervisor sign-off |
| Data source selection | **Done** — see [`03-data-sources.md`](./03-data-sources.md) |
| Tech stack | **Done** — see [`02-tech-stack.md`](./02-tech-stack.md) |
| Repo + CI + module layout | **Done** — this scaffold |
| Scheduled collection workflow | **Done** — `.github/workflows/collect.yml`, now the backup |
| System architecture diagram | **Done** — [`01-architecture.md`](./01-architecture.md) §1 |
| **Delay collection running** | **Done** — live since 3 September 2026 on the always-on collector, 120 s cadence, T1 + T4 ([`06-always-on-collector.md`](./06-always-on-collector.md)) |
| Service Alerts collection | **Done** — `transit-poller` polls alerts every 30 min into `service_alerts`/`alert_scopes` ([`09-service-alerts.md`](./09-service-alerts.md)). The training-table flag waits for an overlap window |
| Reconciliation → training table | **Done** — `transit-reconcile`, schema in [`07-training-table.md`](./07-training-table.md) |
| Prediction: baseline + model training | **Done** — `transit-train`; naive persistence written first. Test MAE 17.92 s vs baseline 19.09 s, **MASE 0.938** ([`08-evaluation-plan.md`](./08-evaluation-plan.md) §3) |
| Corpus ingestion + retrieval | Not started |
| Agent loop + MCP tools | Not started |
| Evaluation plan (supervisor items 4–6) | **Done** — [`08-evaluation-plan.md`](./08-evaluation-plan.md): five sub-RQs, each with criterion, metric and method. Pending sign-off |
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
2. **Ingestion + retrieval.** The three Opal PDFs into chunks that carry document and
   page, embedded into a persisted Chroma collection. Chunks without citations are
   useless to the judge, so citation metadata is part of the ingestion contract, not
   an afterthought.
3. **Realtime tools.** Trip Update and Alerts wrapped as MCP tools over the existing
   `realtime/` client, joined against the static bundle for human-readable stop and
   route names.
4. **Agent loop.** The hand-rolled tool-use loop, with the error-margin requirement
   built into the system prompt from the first version so it is never bolted on.

### Weeks 8–10 — Evaluation harness (**protect this**)
The portfolio differentiator and the answer to the research question. If Weeks 4–7
overrun, scope comes out of the *system*, not out of here.

- 50–100 QA pairs across the three question kinds, with gold evidence.
- Retrieval precision/recall; faithfulness and hallucination rate via Ragas plus the
  custom judge.
- The **prediction-faithfulness metric**: does a prediction-grounded answer state an
  error margin, and is that margin consistent with the model's measured MAE? This is
  the novel measurement and it does not exist off the shelf.
- The static-retrieval-only baseline, run over the same QA set.

**Week 6 checkpoint (hard gate):** count the rows actually collected. If the volume
can't support a defensible train/validation/test split, reframe the prediction layer
as a feasibility study benchmarked against Boudabbous et al. (2026) and Elliker et
al. (2026) — and tell the supervisor at that meeting, not at submission.

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

1. **Confirm the merged research question** (supersedes the earlier RQ1/RQ2 split).
2. **Sign off the problem definition and methodology.**
3. **Evaluation plan** — written: [`08-evaluation-plan.md`](./08-evaluation-plan.md).
   Needs sign-off on one change: the prediction-faithfulness margin moves from
   "consistent with global MAE" to a conditional conformal interval, because
   conditional error spans 7.8-82.4 s and a global margin covers only 31% of the
   delayed trains the tool is actually asked about (§6.2).
4. **Task 1 word count** — the expanded review runs past the original 800–1000 word
   cap. Confirm the limit before final submission.
5. **Live-only data collection** — answered empirically. 166,076 stop events over 12
   service dates; the Week-6 gate is passed and no feasibility-study fallback is
   needed. The open question is now presentational: the model beats its baseline by
   only 6%, and §3 argues that is a reportable finding rather than a problem.
