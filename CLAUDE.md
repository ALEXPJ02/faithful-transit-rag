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

## State as of 2026-10-20 (verified against the repo and the live VM, not recalled)

**The scored run is done** ([`docs/17-results.md`](docs/17-results.md), 2026-10-20). One
frozen file, `data/evaluation_20261018.db` (sha256 `44e90956…32ac`), was scored by the
rehearsed chain. It was scored again after the alert rule's held-out check was completed,
which made 18 urgent repairs published as `MAINTENANCE` into incidents by override
(`docs/09` §7). The second run is the result, and `docs/17` §9 sets the first beside it.
- **Detection**, 5 test dates: AP 0.80 (forest) and 0.75 (XGBoost) against persistence's
  0.72, intervals overlapping. Persistence is the better alarm at validation's thresholds.
- **Delay:** MASE 0.963, and the 90% interval covers 88.9%.
- **Reasons:** k = 3. The test split held 8 incidents, all technical, so cause macro-F1
  across groups could not be computed.
- **Faithfulness**, by the judge, which gives upper bounds: agent 94%, reasons with
  retrieval 15%, and without it 46%.

**Do not re-run the scored chain.** Its outputs are the results. The model answers are
sampled, so a re-run changes them, and any change needs a new run disclosed beside these.

**One post-hoc comparison is disclosed beside them** (`docs/18`), at the supervisor's
request. `transit-detect --compare` and `transit-train --compare` add LightGBM, CatBoost
and logistic regression on the same frozen file. Both reproduce the scored rows byte for
byte and write nothing, so the run changes no result. The agent keeps the scored models.

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
  hold one. Used as a check on rule (b) (`transit-label --cancellation-check`,
  `docs/11` §7), **not** in the main label, which must mean the same on every date. On
  the six dates to 2026-10-04 the check changes no label. They are not yet a detector
  feature. GTFS stop times count from noon less 12 h of the service date
  (`schedule.service_day_origin`), which differs from midnight on clock-change days
  such as 2026-10-04.
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
  result; see `docs/07-training-table.md`. **Its margin is a 90% split conformal
  interval** (`model/conformal.py`), calibrated on validation and saved in the
  artefact. It is binned by line × how late the train already is, because validation
  error varies nine-fold that way and ± MAE covers 32% of already-late trains
  (`docs/08` §4). The supervisor has confirmed the bins.
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
  leakage guards `seen_before` and `exclude_incident` (`docs/10` §5). On the frozen
  evaluation file the rule gives **34 incidents on 17 Sydney dates**: technical 23,
  network 10 and other/unknown 1. Nine of them come from overrides added when the
  held-out check was completed at the cut-off. The index is `voyage-4-lite`, 1024-d.
  Snapshots are opened read-only.
- Repo scaffolding: tests passing, CI green (ruff + mypy + pytest) plus
  CodeQL and a SHA-pinned Trivy workflow. Tests mirror the source tree, so
  `prediction/features/quality.py` is covered by
  `tests/prediction/features/test_quality.py` — but basenames must be globally
  unique, because the suite has no `__init__.py` files and pytest imports each
  module by bare basename.

- **The disruption label** (RQ2 ground truth) — `transit-label --db <snapshot>`
  (`prediction/disruption/`, `docs/11`) labels every T1/T4 15-minute window from the
  §3.2 definition, with every threshold as a parameter. Rule (b) fires only on five or
  more observed services. That minimum, proposed on 2026-10-05, is confirmed
  by the supervisor. Windows with no observed service, or from before alert collection,
  are `pd.NA`, never `False`. To the cut-off it marks 7.5% of T1 windows and 7.3% of
  T4 windows disrupted.

- **The reasons stage** (RQ1 O3) — `transit-reasons` (`agent/situation.py`,
  `agent/reasons.py`, `evaluation/reasons.py`, `docs/13`). It explains each incident
  from a *situation* built from the delay feed alone, never the incident's own alert,
  plus five past incidents retrieved under both leakage guards. The answer is
  structured output with four cause groups, and its citations are checked against
  the passages shown. It is scored against a time-aware most-common-cause baseline
  and the same model without retrieval. The run refuses an alert index that does not
  match the snapshot. A development run on 2026-10-05 is recorded in `docs/13` §5. It
  is not a result.

**The rest of the system, and what is not built:**

- The **live feed** behind the agent's tools is **future work** (decided 2026-10-11).
  The **MCP server** is built
  (`transit-mcp`, `docs/16`). It serves the four tools through `ToolBox.call`, with the
  agent's system prompt as its instructions, and needs the `serve` extra (`mcp` 2.x,
  where `FastMCP` became `MCPServer`). CI installs that extra. The **agent loop** itself is built (`transit-ask`, `docs/14`). It is a
  hand-rolled tool-use loop over line status, disruption risk and past incidents,
  answering as of a moment in a frozen snapshot, with every tool call recorded
  verbatim. A fourth tool gives each running train's expected delay with its 90%
  interval, which the prompt requires stated unnarrowed.
- The rest of the **RQ2 harness** in `evaluation/`. k was frozen at 3 by the confirmed
  rule, and the scored runs are done (`docs/17`). The k sweep is built (`transit-reasons --sweep-k`, `docs/13` §7). It
  compares recall@k with chance, a random draw from the same guarded pool. On the 12
  development incidents, the ranking is no better than chance, and recall@5 (0.75)
  equals chance (0.74), so never quote it alone. Lead time and date-resampled 95% intervals are built
  (`evaluation/lead_time.py`, `evaluation/stats.py`). Detection re-runs reproduce byte
  for byte, because the forest predicts on one thread and thresholds sit between
  scores. **The faithfulness judge is built** (`transit-judge`, `docs/15`). It judges
  each statement against exactly what its system was shown, and every verdict carries
  the fingerprint of the rubric it applied. **Validated, narrowly** (`docs/15` §4):
  - **Round 1**, 21 statements: Haiku 4.5 failed at κ 0.52, and **Claude Opus 4.6, the
    judge now**, passed at 0.71.
  - **Round 2**, 30 held-out statements: it **failed at κ 0.48** under rubric v3
    `e926f55c0cf7`. It kept passing comparisons and extent words its own rules exclude.
  - **Rubric v4 `da0a503ea811`** (2026-10-14, confirmed by the supervisor) makes those
    rules concrete. Fitted to the 51 labels it gives κ 0.71, which is not a validation.
  - **Round 3**, all 41 remaining validation statements: **κ 0.61 [0.38, 0.82]**, which
    passes the rule fixed before its labels. The scored run uses v4.

  **The judge is lenient.** On round 3 it passed 21 of 41 where the author passed 13,
  and every miss ran that way, so its rates are upper bounds. Every validation statement
  now has an author label. On those, retrieval is less faithful than no retrieval, by
  the author (14% against 35%) and by the judge (25% against 42%). These are development
  figures. The labels are assisted: an assistant suggested each, and the author kept or
  changed it.
- **The scored detection run** — the detectors (XGBoost and random forest,
  `prediction/disruption/models.py`) and the persistence baseline are built and run
  by `transit-detect` (`docs/12`). Features look back and the target looks forward.
  The alert feature is re-derived causally, because the label's rule (a) is
  hindsight. Validation figures are selection scores, not results. The scored split
  holds 114 positive targets on 5 test dates (`docs/17` §3).

**The orchestrator is scored on a fixed question set** (`transit-questions`,
`docs/14` §7): both peaks per test date and line, plus each test-date incident 15
minutes after its first sighting, on the detection split's test dates. Write it from
the frozen file at the cut-off, never after seeing answers.

**The scored runs read one frozen file.** No scoring command takes an end date, so
`transit-freeze --db <final snapshot> --cutoff 2026-10-18 --out
data/evaluation_20261018.db` prunes a copy to the cut-off (`docs/08` §3.1). It was
frozen on 2026-10-20 from `data/delay_observations_20261020.db`. It holds 674,833 stop
events over 46 service dates, and its sha256 is `44e90956…32ac`.

**Snapshots in `data/` are frozen, never live.** The newest is
`delay_observations_20261020.db`: 696,618 stop events over 48 service dates
(2026-09-03..10-20, the last partial), with 321 alerts and 9,611 scopes. Integrity is
`ok`, and its sha256 (`f2558722…0a43`) matches the VM's copy. 34 bundle eras (to
20261019) are in `data/bundles/`. The collector keeps running after the cut-off
(`docs/08` §3.1), but nothing after it enters a result. Re-pull from the VM before
quoting any live number, **and pull `data/bundles/` in the same pass** — the instance
archives a timetable era daily and nothing else does.

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
- **The disruption definition, decided 2026-09-28 by the student and confirmed by
  the supervisor** (`docs/08` §3.2). A line × 15-minute window is
  disrupted if (a) an unplanned (non-`MAINTENANCE`) alert is in the feed for the
  line, timed by feed presence (`first_seen_utc`..`last_seen_utc`), and present
  no longer than 24 hours — longer is a standing notice; or (b) at least a quarter
  of the line's services are more than 5 minutes late. **5 minutes, not 3**: a
  3-minute rider's standard was considered and rejected, because 5 is TfNSW's own
  definition and citable. The 30-minute horizon and 15-minute window are agreed
  with the supervisor.
- **Evaluation data has a fixed cut-off: the end of service date 2026-10-18**,
  confirmed by the supervisor.
  Nothing after it enters a scored result. If the reasons evaluation is
  under-powered, report the shortfall and its consequences; do not move the date.
- **Four cause groups, weather kept although empty.** Macro-F1 averages over the
  groups present in the test split's ground truth, and the report names any left
  out.
- **Which alerts are disruptions is decided by the description, not the cause
  alone** (`docs/09` §7, decided 2026-09-29 and confirmed by the supervisor). Planned
  trackwork notices are published as `UNKNOWN_CAUSE` and so is the Edgecliff
  closure; only the wording separates them. The markers were fitted on the 22
  alerts to 2026-09-28 — alerts from 2026-09-29 are the held-out check, and a wrong
  call goes in `data/alert_overrides.csv` with a reason, **never** into edited
  markers. An incident's cause is the first specific cause any of its alerts names.
  The markers' first out-of-sample decision, Croydon on 2026-10-07, was wrong:
  "unable to stop" is an effect they miss. It is overridden, and it changed no label.
- **"Other transport events" are future work; reinforcement learning was assessed
  and not used** (`docs/08` §1, §3.3). Both pending the supervisor's confirmation.
  **A random forest for the unknown alerts**, which the supervisor suggested, was
  answered as not useful (`docs/08` §3.3). One incident has no published cause, so there
  is no ground truth to learn or check, and 17 known-cause incidents are too few.
- **More models are a post-hoc comparison, not the result** (`docs/18`, built
  2026-10-23 at the supervisor's request, after the scored run).
  - **What it adds.** Detection adds LightGBM, CatBoost and logistic regression, and
    delay adds LightGBM and CatBoost.
  - **The grids mirror XGBoost's.** Depths and learning rates are the same, and
    `num_leaves` is 2^depth. `min_child_weight` is the same hessian floor in both
    libraries, which is not a row count under log loss.
  - **The grids are frozen.** They were merged before any added model touched test.
    Never tune them after seeing test, and never promote a model to the agent on its
    test score: that would be choosing on test.
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
Both were last reconciled against the repo and the scored run on 2026-10-20, and
against the post-hoc comparison (`docs/18`) on 2026-10-23.
