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

- Python 3.11+ (3.12 recommended — see `.python-version`)
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

Build the route lookup once, from a static GTFS bundle:

```bash
python -m transit_rag.prediction.collection.routes path/to/gtfs.zip
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

In production the collector runs as a scheduled GitHub Action rather than on a
laptop, writing immutable per-poll snapshots to a dedicated `collected-data` branch.

See [`docs/05-setup-checklist.md`](./docs/05-setup-checklist.md) §5 for running it
somewhere that stays awake.

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
  prediction/      delay collection, features, XGBoost model
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

## Status

Research preparation is complete; the build is at the start of Weeks 4–7. Collection
is the critical path and is the next thing to start — see
[`docs/04-implementation-plan.md`](./docs/04-implementation-plan.md) §1.

## Licence

MIT — see [`LICENSE`](./LICENSE). TfNSW data is used under its own licence terms
(mostly CC BY 4.0; verify per dataset).
