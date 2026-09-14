# CLAUDE.md

Guidance for Claude Code working in this repository. Read this before writing code.

## What this project is

A UTS Capstone (41029 + 41030 taken concurrently in one semester, 13 weeks from
August 2026, one engineer). It is an **agentic RAG system for Sydney public
transport** that answers questions from three sources with deliberately different
trust properties — static Opal policy PDFs, live TfNSW GTFS-Realtime conditions,
and a self-trained delay-prediction model — plus an **evaluation harness** that
measures whether answers stay honest about which source they came from.

**Research question.** How can automated faithfulness and hallucination evaluation
be adapted to an agentic RAG system that answers transport queries using static
policy retrieval, real-time GTFS-Realtime conditions, and a bounded-accuracy
delay-prediction model — and how does this combined system perform against a
static-retrieval-only baseline?

The evaluation harness is the point of the project, not a final step. The novel
measurement is the **prediction-faithfulness metric**: does an answer grounded in
the delay model state an error margin, and is that margin consistent with the
model's measured MAE? If build work overruns, scope comes out of the *system*, not
out of evaluation.

## State as of 2026-09-09 (verified against the repo, not recalled)

**Built, tested, running:**

- Delay collection — `transit-poller` (`src/transit_rag/prediction/collection/`).
  Live on a GCP e2-micro since 2026-09-03T06:46Z, polling every 120 s, T1 and T4
  only, SQLite sink. A GitHub Actions job (`.github/workflows/collect.yml`) is the
  backup, writing immutable CSV snapshots to the `collected-data` branch.
- Reconciliation → training table — `transit-reconcile`
  (`src/transit_rag/prediction/features/`). Schema and rationale in
  `docs/07-training-table.md`.
- Realtime client and parsers (`src/transit_rag/realtime/`).
- Service Alerts collection — the same poller, on a 30-minute clock, into
  `service_alerts`/`alert_scopes` (`docs/09-service-alerts.md`). Alerts are
  stored **unfiltered**: 32% of the route ids they name are absent from the
  realtime bundle, so filtering to T1/T4 at collection would discard 28 of 41
  alerts permanently. Filter in reconcile, which can be re-run.
- Repo scaffolding: 151 tests passing, CI green (ruff + mypy + pytest) plus a
  SHA-pinned Trivy workflow. Tests mirror the source tree, so
  `prediction/features/quality.py` is covered by
  `tests/prediction/features/test_quality.py`.

**Not started — these packages contain only an empty `__init__.py`:**

- `ingestion/` — Opal PDFs into cited chunks
- `retrieval/` — Voyage embeddings into a persisted Chroma collection
- `agent/` — hand-rolled Anthropic tool-use loop
- `mcp_server/` — MCP tool interface + FastAPI
- `evaluation/` — Ragas + custom LLM-as-judge harness
- The XGBoost training script and the naive-persistence baseline (nothing under
  `prediction/` trains a model yet; `models/` is empty)

**`data/delay_observations.db` is a frozen snapshot from 2026-09-04T07:30Z**
(18,100 stop events, 734 polls, 0 failures). It is not live. Re-pull from the VM
before quoting any number from it.

## Hard constraints — breaking these costs real work

- **Python floor is 3.12**, set in `pyproject.toml` (`requires-python = ">=3.12"`).
  It is load-bearing: mypy targeting 3.11 cannot parse numpy's PEP 695 stubs and
  silently skips checking the entire project. `pyproject.toml` is authoritative if
  anything ever disagrees with it.
- **Never `rm -rf .venv`.** The developer's virtualenv lives there.
- **Collected data cannot be re-collected.** TfNSW publishes no historical bus or
  train GTFS-Realtime archive, so a day not collected is training data gone
  permanently. Anything that risks the collector or its database gets a higher bar
  than ordinary code changes.
- **Academic drafts stay out of this repo.** `.docx`/`.pptx` are gitignored as a
  second line of defence; they live one level up, in the `Capstone/` folder.
- **`data/` and `models/` are gitignored** except the tracked
  `data/routes_lookup.csv`.
- Dependency split matters: base is only `requests` + `python-dotenv`. The RAG
  stack sits behind the `rag` extra, with `realtime`/`prediction`/`serve`/
  `evaluation`/`dev` alongside. The collector rebuilds its environment on every
  scheduled run, so nothing heavy may land in base.

## Verify before claiming anything is done

```bash
ruff check . && ruff format --check . && mypy && pytest
```

To reproduce CI exactly (a venv missing the extras will pass locally and fail in
CI):

```bash
uv sync --frozen --extra dev --extra realtime --extra prediction
uv run ruff check . && uv run mypy && uv run pytest
```

## Conventions

Modelled on `github.com/joseph-abdallah04/2026SIS_Group1` — substantial root
README, numbered `docs/NN-topic.md`, `.editorconfig` / `.gitignore` /
`.env.example`, `.github/workflows/`. Use the same shape for anything new.

- ruff for lint and format, line length 100, target py312.
- mypy with `disallow_untyped_defs` — every function is typed.
- pytest; `pyproject.toml` sets `pythonpath = ["src","tests"]`, so no `PYTHONPATH`.
- `uv.lock` is committed; CI and the collector install with `uv sync --frozen`.
- Tests mirror the source layout.
- Comments explain **why**, not what. Commit subjects are written the same way.

## Working rules that have measurably prevented bugs here

1. **Check a claim against live data before designing on top of it.** This is what
   caught the 89% trip-id join rate and a static GTFS bundle that did not pair with
   the realtime feed — both would have silently produced a broken dataset.
2. **No placeholders in copy-paste command blocks.** The developer uses zsh, which
   accepts `YOUR_PROJECT_ID` without complaint, reads `<bundle>.zip` as a
   redirection, and passes a trailing `#` comment through as arguments. Derive
   values with `$(...)`, or put them on their own line with a check that they were
   found.
3. **Ask for an adversarial review before pushing a milestone.** The first such
   review found three blockers, including one that silently fabricated training
   data, plus a CI that would have been red on every push.

## Design decisions already settled — do not relitigate

- **Delay regression, not disruption classification** (disruptions are too rare in
  a short collection window). **XGBoost**, baseline naive persistence, scope T1 and
  T4 only. Metrics MAE / RMSE / MAPE.
- **Chronological split by whole service date, 70/15/15.** Never random — a shuffle
  puts the same afternoon on both sides and invalidates the baseline comparison.
- **The last observation naming a stop event is the outcome proxy**, because
  GTFS-Realtime stops reporting a stop once the train reaches it.
  `stops_ahead_final` records how close that final prediction was.
- **Unmatched trips are kept, not dropped.** Their delays are real; dropping them
  biases toward whichever timetable was current.
- `stop_sequence` is absent from the feed (always the `-1` sentinel) — stop order
  comes from `stop_times.txt`.
- Hand-rolled agent loop rather than a framework; Claude API rather than a local
  model.
- **The active-alert training feature waits for an overlap window.** Alerts began
  collecting after delays did, so adding the flag now would make it `False`
  across the back-catalogue and put a structural break inside the chronological
  split. Rows predating alert collection get `pd.NA`, never `False`.

## Keeping the documents honest

`docs/04-implementation-plan.md`'s status table and the README's status section
both make dated claims, so they go stale silently. When you change what the system
does, update them in the same pass rather than leaving a later reader to be misled.
Both were last reconciled against the repo on 2026-09-09.
