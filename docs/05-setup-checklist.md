# Setup Checklist

> Everything needed to go from a fresh clone to a running delay collector. Work
> through §1–§5 in order; §6 onward is for the parts not built yet.

## 1. Environment

Python 3.11+ (3.12 recommended — see `.python-version`).

```bash
git clone <repo> && cd <repo>
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,realtime]"        # add ,prediction when training starts
cp .env.example .env
```

Verify the toolchain before writing anything:

```bash
ruff check . && ruff format --check . && mypy && pytest
```

## 2. API keys

| Key | Where from | Needed for |
| --- | --- | --- |
| `TFNSW_API_KEY` | [opendata.transport.nsw.gov.au](https://opendata.transport.nsw.gov.au) → your account → **Applications** | Collection + live tools |
| `ANTHROPIC_API_KEY` | [Anthropic Console](https://console.anthropic.com) | Generation + eval judge |
| `VOYAGE_API_KEY` | [Voyage AI dashboard](https://dash.voyageai.com) | Embeddings |

Register a TfNSW application scoped to the **Realtime Trip Update** and **Service
Alerts** products. Registration is free and the key is issued immediately — no
approval wait. The default **Bronze plan** allows 60,000 calls/day at 5 req/sec;
polling every 2 minutes is 720 calls/day, so there is room for several feeds at once.

Put keys in `.env` only. It is gitignored; the repo is public.

**Set a spend limit in the Anthropic Console now**, before the first eval run.

## 3. Confirm the realtime endpoint ← *do not skip*

The exact resource path sits behind login on the Open Data Hub, so `config.py` ships
a documented default that may not match your account's. A wrong URL is the worst
failure mode available here: it usually returns HTML with a `200`, which would look
like a working collector storing nothing.

The client guards against this — it raises when a response isn't a GTFS-Realtime
protobuf rather than writing garbage — but confirm it explicitly:

```bash
transit-poller --once
```

A healthy result prints a poll with a non-zero entity count. If it fails, copy the
exact URL from the API's resource page in your account and override it:

```bash
# in .env
TFNSW_TRIP_UPDATE_URL=https://api.transport.nsw.gov.au/v1/gtfs/realtime/<actual-path>
```

## 4. Build the route lookup

The realtime feed identifies trips by `route_id`; `T1` and `T4` only exist in the
static bundle. Download **Timetables Complete GTFS** (or the "For Realtime" bundle)
from the Open Data Hub, then:

```bash
python -m transit_rag.prediction.collection.routes path/to/gtfs.zip
```

This writes `data/routes_lookup.csv`. **Open it and confirm `T1` and `T4` appear
against plausible `route_id`s** before a long run. Without the file the collector
still runs, but logs every line unfiltered — a bigger database, not a wrong one.
That is deliberate: collecting too much is recoverable, collecting nothing is not.

## 5. Keep it running

The collector's uptime is the project's critical path
([`04-implementation-plan.md`](./04-implementation-plan.md) §1). Options, in
increasing order of reliability:

**a. tmux on your laptop** — fine for a first day, not for a collection window.
Collects only while the machine is awake, which for a laptop means "not overnight".

```bash
tmux new -s poller
transit-poller          # Ctrl-b d to detach
```

**b. An always-on box** (Raspberry Pi, old laptop, free-tier VM) — the reliable
choice. Run it under systemd so it restarts after a reboot:

```ini
# /etc/systemd/system/transit-poller.service
[Unit]
Description=TfNSW delay collector
After=network-online.target

[Service]
Type=simple
WorkingDirectory=/opt/transit-rag
EnvironmentFile=/opt/transit-rag/.env
ExecStart=/opt/transit-rag/.venv/bin/transit-poller
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
```

**c. GitHub Actions cron** — no machine to own, at the cost of coarser granularity
(GitHub's scheduler is best-effort and can drift by several minutes) and the
awkwardness of committing the database back to the repo. Use `--once` per run.
Reasonable as a *supplement* to (b) or when local uptime is genuinely unavailable.

Whichever you choose, verify it after every change:

```bash
transit-poller --status
```

It prints the total collected and the last ten polls. Consecutive `error:` rows mean
collection has silently stopped — the failure this command exists to catch.

## 6. Weekly check (5 minutes)

```bash
transit-poller --status
```

- Is the total still climbing at roughly the expected rate?
- Are recent polls `ok`?
- Are rows non-zero during peak hours? (Zero at 3am is normal; zero at 8am is not.)

Log the running total against the Week 6 checkpoint in
[`04-implementation-plan.md`](./04-implementation-plan.md) §2.

## 7. Not built yet

Ingestion, retrieval, the agent loop, the MCP server, reconciliation, model training
and the evaluation harness. Their setup steps land here as each is built.
