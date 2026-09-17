# Sydney Transit RAG

An agentic RAG system that answers Sydney public transport questions from three
sources with very different trust properties — static Opal policy documents, live
TfNSW GTFS-Realtime conditions, and a bounded-accuracy delay-prediction model — and
an evaluation harness that measures whether its answers stay honest about which is
which.

UTS Capstone (41029 + 41030), 2026.

**Research question.** How can automated faithfulness and hallucination evaluation be
adapted to an agentic RAG system that answers transport queries using static policy
retrieval, real-time GTFS-Realtime conditions, and a bounded-accuracy delay-prediction
model — and how does this combined system perform against a static-retrieval-only
baseline?

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

## Policy retrieval

The static half of the system: three Opal policy PDFs, pinned to content hashes
because TfNSW revises them without notice, chunked so that **every passage carries
its document title and page** — a passage that cannot be cited is unusable to the
faithfulness judge, so ingestion rejects it rather than storing it.

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
  ingestion/       Opal PDFs -> cited chunks
  retrieval/       Voyage embeddings -> Chroma
  realtime/        TfNSW GTFS-Realtime client + parsers
  prediction/      delay collection, reconciliation, XGBoost model
  agent/           hand-rolled Anthropic tool-use loop
  mcp_server/      MCP tool interface + FastAPI
  evaluation/      Ragas + custom LLM-as-judge harness
```

Rationale in [`docs/01-architecture.md`](./docs/01-architecture.md) §3.

## Documentation

- [`docs/00-overview.md`](./docs/00-overview.md) — problem, research question, glossary, scope
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

Research preparation is complete and the build is in the Weeks 4–7 band.

- **Collection** — live since 3 September 2026 on the always-on collector, plus the
  scheduled-Action backup. Reconciliation into the training table is built and tested.
- **Prediction** — `transit-train` fits the delay model against a naive-persistence
  baseline written first. Test MAE 16.02 s vs 18.17 s, MASE 0.881, over 14 service dates.
- **Retrieval** — the three Opal PDFs are pinned, chunked into 144 cited passages and
  indexed by `transit-index` into a persisted Chroma collection
  ([`docs/10-retrieval.md`](./docs/10-retrieval.md)). Not yet built against the real
  embedding model — `VOYAGE_API_KEY` is unset.
- **Next** — the agent loop and MCP tools, then the evaluation harness, which are
  still empty packages.

See [`docs/04-implementation-plan.md`](./docs/04-implementation-plan.md) for the
phase plan and risks.

## Licence

MIT — see [`LICENSE`](./LICENSE). TfNSW data is used under its own licence terms
(mostly CC BY 4.0; verify per dataset).
