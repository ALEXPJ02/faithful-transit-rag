# CLAUDE.md

Guidance for Claude Code working in this repository. Read this before writing code.

## What this project is

A UTS Capstone (41029 + 41030 taken concurrently in one semester, 13 weeks from
August 2026, one engineer). It is an **agentic AI workflow that predicts and
explains train service disruptions on Sydney Trains' T1 and T4 lines** from
real-time GTFS data, plus the **evaluation** that says which models do it better.

The workflow is three stages, and each names its tool or agent:

    retrieve  ->  predict  ->  explain
    GTFS-RT       ML model      reasons grounded in
    feed tool     + margin      retrieved past alerts

**RQ1 — Feasibility.** Investigate the feasibility of running machine
learning-based prediction models on real-time GTFS data to predict or determine
service disruptions, delays, and other transport events on Sydney Trains' T1 and
T4 lines. Its objectives are the three stages above: (1) data retrieval, (2)
prediction stating its error margin, (3) reasons grounded in retrieved evidence.

**RQ2 — Evaluation.** Develop a comprehensive evaluation plan — datasets,
performance metrics, baseline methods, experimental design — to determine which
model detects the disruptions and their reasons better.

Both are Dr Ramezani's wording, with the T1/T4 scope added. The decomposition
lives in `docs/08-evaluation-plan.md`; the authoritative source is
`RQ_List_and_Evaluation_Methods_v2.docx`, one level up in `Capstone/`.

**Scope, set 2026-09-22.** Trains only, service disruptions only, T1 and T4 only.
**Opal fare policy is out** — see "Design decisions" below for what that means for
the code, which is less than it sounds: RAG stays, and the retrieval stack is
reused as-is. Only the corpus changes, from fare PDFs to past service alerts.

If build work overruns, scope comes out of the *system*, not out of evaluation.

## State as of 2026-09-24 (verified against the repo and the live VM, not recalled)

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
  stored **unfiltered**: on the 2026-09-24 snapshot 1,626 of 4,272 scopes (38%)
  name a route absent from the realtime bundle, and only **40 of 134** alerts
  touch T1 or T4 — so filtering at collection would have discarded 94 of them
  permanently. Filter at ingestion, which can be re-run.
- Timetable bundle archive — `transit-bundle-archive.timer` on the VM keeps one
  copy of each distinct static GTFS era, daily. `bundles.discover()` finds them in
  `data/` and `data/bundles/` and dedupes by content, so `transit-reconcile` needs
  no `--bundle` flag.
- Delay model — `transit-train` (`src/transit_rag/prediction/model/`). Naive
  persistence written first. On the 17 schedule-covered dates to 2026-09-24:
  **test MAE 15.45 s vs baseline 18.67 s, MASE 0.828**
  (`models/delay_model_20260924_metrics.json`). Two filters run before the split —
  a 2 h plausibility bound (`quality.MAX_PLAUSIBLE_DELAY_S`) and a whole-date
  schedule-coverage filter (`quality.MIN_SCHEDULE_COVERAGE`) that drops
  2026-09-11..09-15, whose timetable era was never archived. Neither flatters the
  result; see `docs/07-training-table.md`.
- Corpus ingestion — the three Opal PDFs pinned to content hashes and chunked
  page-by-page into **144 cited passages** (`src/transit_rag/ingestion/`).
  **Retired from the research questions, retained as code** (see "Design
  decisions"); the PDF path still works and still passes its tests.
- Retrieval index — `transit-index` (`src/transit_rag/retrieval/`): Voyage
  embeddings into a persisted Chroma collection, carrying an `IndexFingerprint`
  so a retrieval number can be traced to the configuration that produced it
  (`docs/10-retrieval.md`). `VOYAGE_API_KEY` **is** set, and the collection
  `opal_policy` is built and real — 144 chunks, `voyage-4-lite`, 1024-dim,
  cosine, built 2026-09-17T14:01Z. The stack is proven end to end; what has to
  change for RQ1 Objective 3 is the corpus, not the machinery.
- Repo scaffolding: 371 tests passing, CI green (ruff + mypy + pytest) plus
  CodeQL and a SHA-pinned Trivy workflow. Tests mirror the source tree, so
  `prediction/features/quality.py` is covered by
  `tests/prediction/features/test_quality.py` — but basenames must be globally
  unique, because the suite has no `__init__.py` files and pytest imports each
  module by bare basename.

**Not started — these packages contain only an empty `__init__.py`:**

- `agent/` — hand-rolled Anthropic tool-use loop, the RQ1 orchestrator
- `mcp_server/` — MCP tool interface + FastAPI
- `evaluation/` — the RQ2 harness
- **The alert corpus** — nothing reads `service_alerts` back out yet.
  `reconcile.py` does not mention alerts, so the T1/T4 filter `docs/09` promised
  does not exist. This is RQ1 Objective 3's blocker.
- **The disruption classifier** — RQ2 compares detectors, and none exists. Gated
  on the §3.2 disruption definition being agreed.

**Snapshots in `data/` are frozen, never live.** The newest is
`delay_observations_20260924.db` — 321,634 stop events over 22 service dates
(2026-09-03..24), T1 192,114 · T4 129,520, 134 alerts, 4,272 scopes, integrity
`ok`. Re-pull from the VM before quoting any number, **and pull
`data/bundles/` in the same pass** — the instance archives a timetable era daily
and nothing else does.

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
uv sync --frozen --extra dev --extra realtime --extra prediction --extra rag
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

- **Delay regression *and* disruption classification.** Changed 2026-09-22 by the
  supervisor; the old entry here read "delay regression, **not** classification
  (disruptions are too rare in a short collection window)" and a later session
  reading only that rationale would revert this. Regression survives as RQ1
  Objective 2's supporting output (XGBoost, baseline naive persistence, MAE /
  RMSE / MAPE). Classification is added for RQ2 and is **not yet built**: it is
  gated on the disruption definition in `docs/08` §3.2 being agreed.

  The rarity concern was correct and has not gone away. Ten unplanned alerts touch
  T1/T4 in the whole alert history to 2026-09-24, and they fall on **three Sydney
  dates**, not one a day. On the current test dates the reasons set holds 2 alerts
  of a single cause group, so cause macro-F1 is not yet computable. That is a
  scheduling constraint on RQ2, not a reason to avoid the question — but do not
  quote "about one a day", which is wrong.
- **Opal fare policy is out of scope**, set 2026-09-22. RAG stays; the corpus
  becomes past T1/T4 service alerts. The Opal PDFs, `ingestion/corpus.py` and the
  built `opal_policy` collection are **retained, not deleted** — they are working,
  tested, reviewed code and the PDF path is the second source that proves the
  ingestion contract is not alert-specific. Do not index them for the RQs.
- **Chronological split by whole service date, 70/15/15.** Never random — a shuffle
  puts the same afternoon on both sides and invalidates the baseline comparison.
- **The last observation naming a stop event is the outcome proxy**, because
  GTFS-Realtime stops reporting a stop once the train reaches it.
  `stops_ahead_final` records how close that final prediction was.
- **Unmatched *trips* are kept; schedule-blind *dates* are dropped.** A trip the
  current bundle does not describe still has a real delay, so reconciliation keeps
  it. A whole service date whose timetable era was never archived is different:
  every row lacks `scheduled_arrival_s` and `stop_sequence`, so a partition built
  from it measures a nine-feature model on seven. `transit-train` drops those
  dates before the split (`quality.MIN_SCHEDULE_COVERAGE`), and
  `--keep-schedule-blind` audits what that removes.
- `stop_sequence` is absent from the feed (always the `-1` sentinel) — stop order
  comes from `stop_times.txt`.
- Hand-rolled agent loop rather than a framework; Claude API rather than a local
  model.
- **Retrieval is cosine, and the index is fingerprinted.** Chroma's default is
  L2, which would rank partly by vector magnitude; `hnsw:space` cannot be
  changed after creation. Every collection stores the embedding model, chunk
  size, overlap and corpus content hash, because `docs/08` §3.5 freezes chunk size
  and k before scoring and a number that cannot be traced to its configuration
  is not reproducible. `transit-index status` exits non-zero when they disagree.
- **Documents and queries use different Voyage `input_type` values**, and
  `search` returns cosine *similarity*, never distance. Both are silent
  failures: the first costs retrieval quality invisibly, the second returns the
  worst passages first while still looking correct.
- **A rebuild of the index is destructive, not incremental.** The corpus is 144
  chunks and ~26.6k tokens, so a full re-embed is ~0.013% of the Voyage free
  tier — cheap enough that guaranteeing no passages survive from a previous
  chunk size is worth more than any saving.
- **The active-alert training feature waits for an overlap window.** Alerts began
  collecting after delays did, so adding the flag now would make it `False`
  across the back-catalogue and put a structural break inside the chronological
  split. Rows predating alert collection get `pd.NA`, never `False`.

## Keeping the documents honest

`docs/04-implementation-plan.md`'s status table and the README's status section
both make dated claims, so they go stale silently. When you change what the system
does, update them in the same pass rather than leaving a later reader to be misled.
Both were last reconciled against the repo on 2026-09-17.
