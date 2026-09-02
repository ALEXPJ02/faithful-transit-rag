# Implementation Plan

> One semester, one engineer, 41029 + 41030 concurrently. The plan is written around
> what can be cut, because something will be.

## Status — 2 September 2026

| Workstream | State |
| --- | --- |
| Literature review | **Done** — 11 references, 3 from 2026, three thematic clusters |
| Problem definition + methodology | **Done** — written up, pending supervisor sign-off |
| Research question | **Merged into one**, pending supervisor sign-off |
| Data source selection | **Done** — see [`03-data-sources.md`](./03-data-sources.md) |
| Tech stack | **Done** — see [`02-tech-stack.md`](./02-tech-stack.md) |
| Repo + CI + module layout | **Done** — this scaffold |
| Scheduled collection workflow | **Done** — `.github/workflows/collect.yml` |
| System architecture diagram | **Done** — [`01-architecture.md`](./01-architecture.md) §1 |
| **Delay collection running** | **Collector built and tested; not yet collecting.** Blocked only on the API key and a first live poll — critical path, see §1 |
| Corpus ingestion + retrieval | Not started |
| Agent loop + MCP tools | Not started |
| Prediction: reconciliation, features, training | Not started (blocked on collection) |
| Evaluation plan (supervisor item 4) | Not started |
| Evaluation harness | Not started |

## 1. The one thing that cannot be caught up

Every other task on this list can absorb a slip by being done faster or smaller
later. Collection cannot. There is no historical bus/train GTFS-Realtime archive to
fall back on ([`02-tech-stack.md`](./02-tech-stack.md) §3), so the training set is
exactly what gets collected between the day the poller starts and the day the model
is needed — and not a row more.

The methodology write-up records collection as having started 2026-08-23. It has
not: there is no observation database and no route lookup. That gap is real, it is
counted in days, and closing it is the first task in this repo's queue.

**Consequence for the fallback.** The Week-6 checkpoint below exists to decide
between a live-trained model and a methodological feasibility study benchmarked
against published results. Every week collection is delayed moves that decision
closer to being made for us rather than by us.

## 2. Phases

### Weeks 1–3 — Research preparation ✅
Literature review, dataset selection, supervisor onboarding, stack decisions.

### Weeks 4–7 — Core build (**current**)
Ordered by what unblocks what:

1. **Start collection.** API key, endpoint confirmation, route lookup, a real poll,
   then always-on scheduling. Nothing else on this list is time-critical.
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
| **Collection not running / silently failing** | Fatal to the prediction layer | Start immediately; `--status` after every scheduling change; check `poll_log` weekly |
| Wrong realtime endpoint path | Collects nothing, looks fine | The client raises rather than storing garbage when the response isn't protobuf; verify with `--once` before any long run |
| Laptop uptime | Gaps in the training window | Schedule it off the laptop — GitHub Actions cron or an always-on box ([`05-setup-checklist.md`](./05-setup-checklist.md) §5) |
| Weeks 4–7 overrun into the harness | Loses the highest-value phase | Cut system scope, not harness scope. Trip Planner API is the escape hatch for GTFS joins |
| Anthropic spend overrun | Budget | Console spend limit; Haiku for judging; cache eval-set embeddings |
| RQ or methodology not signed off | Rework late | Both are flagged pending; raise at the next supervisor meeting |
| Supervisor's expertise is a step from RAG specifics | Unreviewed technical choices | Every non-obvious decision is written down with its rejected alternative ([`02-tech-stack.md`](./02-tech-stack.md) §6) |

## 4. Outstanding supervisor items

1. **Confirm the merged research question** (supersedes the earlier RQ1/RQ2 split).
2. **Sign off the problem definition and methodology.**
3. **Evaluation plan** — datasets, metrics, baselines, experimental design. Not yet
   started; due before the harness is built, not after.
4. **Task 1 word count** — the expanded review runs past the original 800–1000 word
   cap. Confirm the limit before final submission.
5. **Live-only data collection** — is a model trained on weeks of self-collected data
   acceptable, or should the feasibility-study framing be committed to upfront?
