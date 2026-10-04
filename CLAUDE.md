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
`RQ_List_and_Evaluation_Methods_v3.docx`, one level up in `Capstone/`. v2 is kept
as the version the supervisor saw, but its Objective 2 input list and its "roughly
one a day" are wrong — do not copy from it.

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
- Trip-status collection — cancelled, added, replacement and stop-skipping trips on
  T1/T4 into `trip_statuses`, **from 2026-09-29 only** (`docs/01` §5). TfNSW
  publishes a cancellation with no stop updates, so `stop_observations` can never
  hold one. Used as a check on rule (b) and as a detector feature (`pd.NA` before
  2026-09-29), **not** in the main label, which must mean the same on every date.
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
  cosine, built 2026-09-17T14:01Z — retired from the RQs.
- **The alert corpus** (RQ1 Objective 3) — `transit-alerts audit --db <snapshot>`
  (`ingestion/alerts.py`) decides which T1/T4 alerts are disruptions and groups
  republications into incidents, printing every decision with its reason
  (`docs/09` §7). `transit-index build --source alerts --db <snapshot>` indexes one
  passage per incident into `tfnsw_alerts`, cited by alert ids; `search` takes the
  leakage guards `seen_before` and `exclude_incident` (`docs/10` §5). Built
  2026-09-28: **8 incidents on 5 dates**, `voyage-4-lite`, 1024-d. Snapshots are
  opened read-only.
- Repo scaffolding: tests passing, CI green (ruff + mypy + pytest) plus
  CodeQL and a SHA-pinned Trivy workflow. Tests mirror the source tree, so
  `prediction/features/quality.py` is covered by
  `tests/prediction/features/test_quality.py` — but basenames must be globally
  unique, because the suite has no `__init__.py` files and pytest imports each
  module by bare basename.

- **The disruption label** (RQ2 ground truth) — `transit-label --db <snapshot>`
  (`prediction/disruption/`, `docs/11`) labels every T1/T4 15-minute window from the
  §3.2 definition, with every threshold as a parameter. Rule (b) fires only on five or
  more observed services. That minimum was proposed on 2026-10-05 and is pending
  confirmation. Windows with no observed service, or from before alert collection,
  are `pd.NA`, never `False`. To 2026-10-04 it marks 4.7% of T1 windows and 4.4% of
  T4 windows disrupted.

**Not started — these packages contain only an empty `__init__.py`:**

- `agent/` — hand-rolled Anthropic tool-use loop, the RQ1 orchestrator
- `mcp_server/` — MCP tool interface + FastAPI
- `evaluation/` — the RQ2 harness
- **The reasons agent** — Claude writing a cause from the retrieved incidents. The
  corpus and guarded retrieval it needs exist.
- **The disruption classifier** — RQ2 compares detectors, and none exists. The label
  it is scored against is built (above). It needs line-level window features, not the
  delay model's per-stop ones, and its target is the next 30 minutes' label, never
  the label of the windows its features come from (`docs/08` §3.3).

**Snapshots in `data/` are frozen, never live.** The newest is
`delay_observations_20261005.db`: 481,084 stop events over 32 service dates
(2026-09-03..10-04), T1 286,765 · T4 194,319. It also has 184 alerts, 5,624 scopes,
and 628 trip statuses over 6 dates. Integrity is `ok`, and its sha256 matches the
VM's copy. 19 bundle eras (to 20261004) are in `data/bundles/`. Re-pull from the VM
before quoting any number, **and pull `data/bundles/` in the same pass** — the
instance archives a timetable era daily and nothing else does.

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
3. **Every change goes through a reviewed PR.** Pull the latest `main`, commit on
   a new branch, open a PR, let Cursor Bot review it, and merge only once every
   check passes and every issue it flags is resolved. Set 2026-09-28; it replaces a
   separate adversarial-agent review before each push, which earned its place —
   the first found three blockers, including one that silently fabricated
   training data — so read what the bot flags rather than dismissing it.

## Design decisions already settled — do not relitigate

- **Delay regression *and* disruption classification.** Changed 2026-09-22 by the
  supervisor; the old entry here read "delay regression, **not** classification
  (disruptions are too rare in a short collection window)" and a later session
  reading only that rationale would revert this. Regression survives as RQ1
  Objective 2's supporting output (XGBoost, baseline naive persistence, MAE /
  RMSE / MAPE). Classification is added for RQ2 and is **not yet built**; its
  definition is the next entry.

  The rarity concern was correct and has not gone away. Ten unplanned alerts touch
  T1/T4 in the whole alert history to 2026-09-24, and they fall on **three Sydney
  dates**, not one a day. On the current test dates the reasons set holds 2 alerts
  of a single cause group, so cause macro-F1 is not yet computable. That is a
  scheduling constraint on RQ2, not a reason to avoid the question — but do not
  quote "about one a day", which is wrong.
- **The disruption definition, decided 2026-09-28 by the student, pending the
  supervisor's confirmation** (`docs/08` §3.2). A line × 15-minute window is
  disrupted if (a) an unplanned (non-`MAINTENANCE`) alert is in the feed for the
  line, timed by feed presence (`first_seen_utc`..`last_seen_utc`), and present
  no longer than 24 hours — longer is a standing notice; or (b) at least a quarter
  of the line's services are more than 5 minutes late. **5 minutes, not 3**: a
  3-minute rider's standard was considered and rejected, because 5 is TfNSW's own
  definition and citable. The 30-minute horizon and 15-minute window are agreed
  with the supervisor.
- **Evaluation data has a fixed cut-off: the end of service date 2026-10-18.**
  Nothing after it enters a scored result. If the reasons evaluation is
  under-powered, report the shortfall and its consequences; do not move the date.
- **Four cause groups, weather kept although empty.** Macro-F1 averages over the
  groups present in the test split's ground truth, and the report names any left
  out.
- **Which alerts are disruptions is decided by the description, not the cause
  alone** (`docs/09` §7, decided 2026-09-29 pending the supervisor). Planned
  trackwork notices are published as `UNKNOWN_CAUSE` and so is the Edgecliff
  closure; only the wording separates them. The markers were fitted on the 22
  alerts to 2026-09-28 — alerts from 2026-09-29 are the held-out check, and a wrong
  call goes in `data/alert_overrides.csv` with a reason, **never** into edited
  markers. An incident's cause is the first specific cause any of its alerts names.
- **"Other transport events" are future work; reinforcement learning was assessed
  and not used** (`docs/08` §1, §3.3). Both pending the supervisor's confirmation.
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
Both were last reconciled against the repo on 2026-09-24, and their decisions
updated on 2026-09-28.
