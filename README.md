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
documents. See [`docs/08-evaluation-plan.md`](./docs/08-evaluation-plan.md).

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
  mcp_server/      MCP tool interface + FastAPI
  evaluation/      Ragas + custom LLM-as-judge harness
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

## Status

Reconciled against the repo and the live collector on 2026-09-24.

- **Collection (RQ1 O1)** — live since 3 September 2026, plus the scheduled-Action
  backup. **321,634 stop events over 22 service dates**, T1 192,114 · T4 129,520;
  134 service alerts, 40 of which touch T1 or T4. 15,038 of 15,039 polls successful.
- **Prediction (RQ1 O2)** — `transit-train` fits the delay model against a
  naive-persistence baseline written first. On the 17 schedule-covered dates:
  **test MAE 15.45 s vs 18.67 s, MASE 0.828**. The disruption classifier that RQ2
  compares does not exist yet — it is gated on the disruption definition in
  [`docs/08`](./docs/08-evaluation-plan.md) §3.2.
- **Retrieval (RQ1 O3)** — the stack is proven end to end against real Voyage
  embeddings: collection `opal_policy`, 144 chunks, `voyage-4-lite`, cosine
  ([`docs/10-retrieval.md`](./docs/10-retrieval.md)). It indexes the **wrong corpus**
  for the current scope; repointing it at past T1/T4 alerts is the next build step.
- **Next** — alert ingestion and its index, then the agent loop and MCP tools, then
  the evaluation harness. All still empty packages.

See [`docs/04-implementation-plan.md`](./docs/04-implementation-plan.md) for the
phase plan and risks.

## Licence

MIT — see [`LICENSE`](./LICENSE). TfNSW data is used under its own licence terms
(mostly CC BY 4.0; verify per dataset).
