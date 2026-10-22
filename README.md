# Sydney Transit RAG

An agentic AI workflow that **predicts and explains train service disruptions on
Sydney Trains' T1 and T4 lines** from real-time GTFS data, and the evaluation that
says which models do it better.

Three stages, each with its own tool or agent:

    retrieve  ->  predict  ->  explain
    GTFS-RT       ML model      reasons grounded in
    feed tool     + margin      retrieved past alerts

UTS Capstone (41029 + 41030), 2026.

**RQ1 — Feasibility.** Investigate the feasibility of running machine learning-based
prediction models on real-time GTFS data to predict or determine service disruptions,
delays, and other transport events on Sydney Trains' T1 and T4 lines.

**RQ2 — Evaluation.** Develop a comprehensive evaluation plan — datasets, performance
metrics, baseline methods, experimental design — to determine which model detects the
disruptions and their reasons better.

Scope was narrowed to T1/T4 service disruptions on 2026-09-22, and Opal fare policy
dropped. Retrieval stays: the corpus becomes past service alerts rather than fare
documents. "Other transport events" are future work, pending the supervisor's
confirmation. See [`docs/08-evaluation-plan.md`](./docs/08-evaluation-plan.md).

## Prerequisites

- Python 3.12+ (see `.python-version`; `pyproject.toml` sets the floor)
- A free [TfNSW Open Data Hub](https://opendata.transport.nsw.gov.au) API key
- Anthropic and Voyage AI API keys

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,realtime]"
cp .env.example .env
```

Add `,rag` to the extras once you are building retrieval, and `,prediction` for
the model. Then open `.env` and fill in the keys.

Dependencies are split so the collector installs almost nothing: base is
`requests` + `python-dotenv`, and the RAG stack lives behind the `rag` extra.
`uv.lock` pins every version — CI installs from it with `uv sync --frozen`.

Full walkthrough, including the endpoint confirmation step you should not skip:
[`docs/05-setup-checklist.md`](./docs/05-setup-checklist.md).

## Delay collection

The prediction layer trains on data this project collects itself — TfNSW publishes no
historical bus or train GTFS-Realtime archive, so **a day not collected is a day of
training data gone permanently**. Getting this running comes before everything else.

Build the route lookup once. `--fetch` downloads the "For Realtime" static bundle
with the same API key the feeds use:

```bash
python -m transit_rag.prediction.collection.routes --fetch
```

Then, in order — check the feed, collect, and check on it:

```bash
transit-poller --probe
transit-poller
transit-poller --status
```

`--probe` before anything else: it fetches once, stores nothing, and separates the
three failures that otherwise look identical — a wrong endpoint, a static bundle
that does not pair with the feed, and the tracked lines simply not running yet.

`--status` is the one to keep coming back to. The dangerous failure is not a crash
but a collector that runs for a week while every request fails; it prints the last
ten poll outcomes, so that shows up immediately.

In production the collector runs on an always-on GCP e2-micro rather than a laptop,
polling every 120 s into SQLite — see
[`docs/06-always-on-collector.md`](./docs/06-always-on-collector.md). A scheduled
GitHub Action (`.github/workflows/collect.yml`) backs it up, writing immutable
per-poll CSV snapshots to a dedicated `collected-data` branch.

## Retrieval

The evidence half of the workflow. **Every passage carries its source and a locator**
— a passage that cannot be cited is unusable to the faithfulness judge, so ingestion
rejects it rather than storing it.

> **The corpus is changing.** Retrieval was built against three Opal fare-policy PDFs,
> pinned to content hashes because TfNSW revises them without notice. Opal went out of
> scope on 2026-09-22; RQ1 Objective 3 retrieves **past T1/T4 service alerts** instead,
> to explain a disruption's cause. The PDF path is retained and still passes its tests
> — the machinery is unchanged, only the corpus moves.

```bash
python -m transit_rag.ingestion.corpus --fetch   # download and verify the three PDFs
transit-index build --dry-run                    # 144 chunks; no API calls, nothing written
transit-index build                              # embed with Voyage, persist to Chroma
transit-index status                             # what is on disk, and whether it is stale
transit-index query "how does a daily cap work"
```

The reasons corpus is built from a collection snapshot instead:

```bash
transit-alerts audit --db data/delay_observations_20260929.db    # every alert, verdict and reason
transit-index build --source alerts --db data/delay_observations_20260929.db
transit-index query --source alerts "T1 delays near Chatswood" --seen-before 2026-09-25T17:19:00+10:00
```

`status` exits non-zero when the stored index no longer matches the pinned corpus or
the configured chunking, so it works as a pre-eval check. That matters because a
stale index does not fail — it answers confidently, with citations, from a document
revision the write-up no longer names.
[`docs/10-retrieval.md`](./docs/10-retrieval.md) has the rest.

## Development

```bash
ruff check .
ruff format .
mypy
pytest
```

Lint, format, typecheck, tests. `pytest` needs no `PYTHONPATH` — `pyproject.toml`
sets it. CI runs all four on every push and PR.

## Layout

```
src/transit_rag/
  config.py        environment-driven settings (stdlib only)
  ingestion/       documents -> cited chunks (Opal PDFs; alerts next)
  retrieval/       Voyage embeddings -> Chroma
  realtime/        TfNSW GTFS-Realtime client + parsers
  prediction/      delay collection, reconciliation, XGBoost model
  agent/           hand-rolled Anthropic tool-use loop
  mcp_server/      the agent's tools as an MCP server (transit-mcp)
  evaluation/      scoring, intervals, and the faithfulness judge + its κ
```

Rationale in [`docs/01-architecture.md`](./docs/01-architecture.md) §3.

## Documentation

- [`docs/00-overview.md`](./docs/00-overview.md) — problem, research questions, glossary, scope
- [`docs/01-architecture.md`](./docs/01-architecture.md) — system diagram, components, repo layout, conventions
- [`docs/02-tech-stack.md`](./docs/02-tech-stack.md) — technology choices, rationale, cost
- [`docs/03-data-sources.md`](./docs/03-data-sources.md) — TfNSW dataset selection and access
- [`docs/04-implementation-plan.md`](./docs/04-implementation-plan.md) — 13-week plan, current status, risks
- [`docs/05-setup-checklist.md`](./docs/05-setup-checklist.md) — keys, endpoints, keeping collection running
- [`docs/06-always-on-collector.md`](./docs/06-always-on-collector.md) — the always-on GCP collector
- [`docs/07-training-table.md`](./docs/07-training-table.md) — reconciliation and the training-table schema
- [`docs/08-evaluation-plan.md`](./docs/08-evaluation-plan.md) — the research questions and how each one gets measured
- [`docs/09-service-alerts.md`](./docs/09-service-alerts.md) — the Service Alerts feed and the alert tables
- [`docs/10-retrieval.md`](./docs/10-retrieval.md) — chunking, the index fingerprint, and the retrieval decisions
- [`docs/11-disruption-labels.md`](./docs/11-disruption-labels.md) — the disruption label RQ2's detectors are scored against
- [`docs/12-disruption-detection.md`](./docs/12-disruption-detection.md) — what a detector may see, how it is scored, and the baseline
- [`docs/13-reasons.md`](./docs/13-reasons.md) — explaining a disruption from past alerts, and scoring it
- [`docs/14-agent.md`](./docs/14-agent.md) — the orchestrator: the loop, its tools, and answering as of a moment
- [`docs/15-faithfulness-judge.md`](./docs/15-faithfulness-judge.md) — the faithfulness judge: per-statement verdicts, its frozen rubric, and κ against the author
- [`docs/16-mcp-server.md`](./docs/16-mcp-server.md) — the agent's tools served to any MCP client, as of a moment
- [`docs/17-results.md`](./docs/17-results.md) — the scored run at the 2026-10-18 cut-off: every result, with its counts and intervals
- [`docs/18-model-comparison.md`](./docs/18-model-comparison.md) — LightGBM, CatBoost and logistic regression, compared after the scored run on the same file

## Status

Every figure below is from the scored run on the frozen evaluation file at the
2026-10-18 cut-off ([`docs/17`](./docs/17-results.md)), made on 2026-10-20.

- **Collection (RQ1 O1)** — live since 3 September 2026, plus the scheduled-Action
  backup. To the cut-off: **674,833 stop events over 46 service dates**, T1 403,185 ·
  T4 271,648, with 32,348 of 32,349 trip polls and all 1,624 alert polls succeeding.
  115 service alerts name T1 or T4. Cancelled and altered trips are collected from
  2026-09-29 (`trip_statuses`). The frozen file's sha256 is `44e90956…32ac`.
- **Prediction (RQ1 O2)** — the delay model beats naive persistence narrowly on the 6
  test dates: **MAE 22.0 s vs 22.8 s, MASE 0.963**, and it is worse on RMSE. A paired
  interval added afterwards cannot tell that gain from zero
  ([`docs/18`](./docs/18-model-comparison.md) §6). Its
  **margin**, a 90% conformal interval by line × how late the train already is
  ([`docs/08`](./docs/08-evaluation-plan.md) §4), covers **88.9%** of test rows, where
  ± MAE covers 30% of trains already more than 5 minutes late. The **disruption label**
  ([`docs/11`](./docs/11-disruption-labels.md)) marks 7.5% of T1 windows and 7.3% of T4
  windows disrupted. On 5 test dates with 114 positive targets, the detectors
  ([`docs/12`](./docs/12-disruption-detection.md)) rank windows slightly better than
  persistence, at **AP 0.80 (random forest) and 0.75 (XGBoost) against 0.72**. The
  intervals overlap almost entirely, and at validation's thresholds persistence is the
  better alarm.
- **Reasons (RQ1 O3)** — collected alerts become incidents, which are indexed one passage
  per incident with retrieval that cannot see the future or the incident being
  explained ([`docs/09`](./docs/09-service-alerts.md) §7,
  [`docs/10`](./docs/10-retrieval.md) §5). To the cut-off there are **34 incidents on 17
  dates**. Nine are urgent repairs published as `MAINTENANCE`, which the rule excluded
  until its held-out check was completed at the cut-off. k was frozen at 3. The test split held **8
  incidents, all technical**, so cause macro-F1 across groups could not be computed. The
  most-common-cause baseline named all 8. The model named 3 of 24 without retrieval and
  2 of 24 with it ([`docs/13`](./docs/13-reasons.md), `docs/17` §5).
- **Orchestrator (RQ1)** — `transit-ask` ([`docs/14`](./docs/14-agent.md)) is a
  hand-rolled Claude tool-use loop over line status, disruption risk, each train's
  expected delay with its 90% interval, and past incidents. It answers as of a moment in
  a frozen snapshot and records every tool call for scoring. It answered 28 questions on
  the test dates. Every delay prediction carried its interval. The judge found 94% of
  its statements supported, against 15% for the reasons model with retrieval and 46%
  without (`docs/17` §6–7).
- **Evaluation harness (RQ2)** — detection is scored with average precision, lead time
  against the operator's alerts, and 95% intervals from resampling whole service dates.
  Reasons are scored with macro-F1 and its interval. Explanations are judged per
  statement against exactly what each system was shown (`transit-judge`,
  [`docs/15`](./docs/15-faithfulness-judge.md)). The judge, Claude Opus 4.6 with rubric
  version 4, passed a third held-out round against the author at κ 0.61. It leans
  lenient, so its faithfulness rates are upper bounds.
- **More models, after the scored run** — at the supervisor's request, `--compare`
  adds LightGBM, CatBoost and logistic regression to detection, and LightGBM and
  CatBoost to delay prediction, on the same frozen file
  ([`docs/18`](./docs/18-model-comparison.md)). Their grids mirror XGBoost's and were
  frozen before any of them saw test. Run once on 2026-10-23:
  - **Delay:** LightGBM beats persistence clearly, **MASE 0.813**, with a paired 95%
    interval of [−5.2, −3.3] s. It beats persistence on RMSE as well, where XGBoost
    loses, and validation would also have chosen it.
  - **Detection:** no model beats persistence significantly, even paired by date.
    Logistic regression (AP 0.78) ranks as well as the boosted trees (0.75–0.76).

  It is disclosed beside the result, never part of it, and the agent keeps the scored
  models.
- **Next** — the write-up. A live feed behind the agent's tools is future work. The MCP
  server is built (`transit-mcp`, [`docs/16`](./docs/16-mcp-server.md)).

See [`docs/04-implementation-plan.md`](./docs/04-implementation-plan.md) for the
phase plan and risks.

## Licence

MIT — see [`LICENSE`](./LICENSE). TfNSW data is used under its own licence terms
(mostly CC BY 4.0; verify per dataset).
