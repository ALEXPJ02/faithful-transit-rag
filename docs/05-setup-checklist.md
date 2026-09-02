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

## 2. What to get from the Open Data Hub

Three separate things, used at different times. Rationale for each choice is in
[`03-data-sources.md`](./03-data-sources.md); this is just the shopping list.

| # | What | Type | Used for | Needed |
| --- | --- | --- | --- | --- |
| 1 | **Realtime Trip Update** (Sydney Trains) | API (protobuf) | Delay collection + the agent's live tools | **Now** |
| 2 | **Timetables — For Realtime** (or *Timetables Complete GTFS*) | Static zip | `routes.txt` → the T1/T4 route lookup; later, human-readable stop and route names | **Now** |
| 3 | Opal **Fares Business Rules**, Opal **Terms of Use**, **Fares & Ticketing brochure** | 3 PDFs | The retrieval corpus | Weeks 4–7 |

Also worth adding to the same API key while you are there, since it costs nothing
and the agent needs it later: **Realtime Service Alerts** (Sydney Trains).

**On the static bundle:** either works — the lookup script only reads `routes.txt`
out of the zip. *For Realtime* is smaller and is scoped to operators that actually
have live feeds, which is what the realtime joins will want later. *Complete GTFS*
is the full network including regional and trackwork routes.

**Not using:** the Trip Planner API (it would outsource the trip logic the agent is
supposed to reason through), the structured Opal Fares CSVs (tabular, not the prose
a faithfulness judge can cite), and anything roads or parking related. See
[`03-data-sources.md`](./03-data-sources.md) §4.

## 3. API keys

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

## 4. The realtime endpoint

Both confirmed from the products' own OpenAPI consoles, and both are the repo
defaults — there is nothing to set unless TfNSW changes them:

| Feed | URL |
| --- | --- |
| Trip updates | `https://api.transport.nsw.gov.au/v2/gtfs/realtime/sydneytrains` |
| Service alerts | `https://api.transport.nsw.gov.au/v2/gtfs/alerts/sydneytrains` |

**The version is a path prefix, not a suffix** — `/v2/gtfs/realtime`, never
`/v1/gtfs/realtime/.../v2`. The two products differ only in the segment after the
prefix. Trip updates also serve `/metro` and `/lightrail/innerwest`; alerts
additionally serve `/all` across every operator, plus `/buses`, `/ferries` and
`/nswtrains`.

Two things the console gives you for free, worth using before touching the
terminal:

- **Authorize** (top right) accepts your API key and lets you fire the request from
  the browser. If it returns binary, the key works and the path is right.
- Adding `?debug=true` returns the feed as readable text instead of protobuf —
  useful for eyeballing what a trip update actually contains.

Nothing reads the alerts feed yet — it feeds the prediction layer's active-alert
flag and the agent's live tools, both still to be built. Adding it to the key now
just saves a second trip to the portal.

## 4b. Probe the feed before trusting it

TfNSW publishes the static bundle and the realtime feed in **versioned pairs**, and
`route_id` values differ between versions. Mixing them is the quietest failure in
the pipeline: the fetch succeeds, the parse succeeds, the filter matches nothing,
and you collect zero rows for a week without a single error.

```bash
transit-poller --probe
```

It fetches once, stores nothing, and tells you which lines the feed is actually
carrying and whether the lookup resolves them:

```
Entities: 312  (trip updates: 298)
Distinct route_ids: 14

route_id                 line     trips
  APS_1a                 T1       41
  APS_4a                 T4       23
  ...

T1: 41 active trips
T4: 23 active trips

Feed and lookup agree. Safe to start collecting.
```

If instead every `route_id` shows `—`, the bundle and the feed are from different
versions — re-download the static bundle that pairs with the realtime product on
your key. If the ids resolve but T1/T4 are absent, you are probably probing
overnight; try again during service hours before changing anything.

## 5. Build the route lookup

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

## 6. Keep it running — GitHub Actions

The collector's uptime is the project's critical path
([`04-implementation-plan.md`](./04-implementation-plan.md) §1), and a laptop is not
an uptime strategy. `.github/workflows/collect.yml` runs it on GitHub's
infrastructure instead.

**The repository must be public.** Actions minutes are unmetered on public repos; on
a private one this schedule would exhaust the 2,000-minute monthly free tier in
about a week. The repo is meant to be public anyway — the academic drafts are
gitignored precisely so it can be.

### One-time setup

1. **Push to GitHub** as a **public** repo (`faithful-transit-rag`):

   ```bash
   gh repo create faithful-transit-rag --public --source=. --remote=origin --push
   ```

2. **Commit the route lookup.** Unlike the rest of `data/`, `data/routes_lookup.csv`
   is tracked — the scheduled run needs it in the repo to scope itself to T1/T4.
   Without it the collector logs every Sydney Trains line, which is not wrong but
   is many times the volume.

   ```bash
   git add data/routes_lookup.csv && git commit -m "Add T1/T4 route lookup" && git push
   ```

3. **Add the API key as a secret.** Repo → Settings → Secrets and variables →
   Actions → New repository secret: `TFNSW_API_KEY`.

   Until this exists the workflow runs on schedule and **exits quietly** without
   collecting. That is deliberate: the schedule activates the moment the workflow
   file reaches the default branch, which is long before the key does, and 288
   failure notifications a day is not a useful signal.

   If your account's endpoint path differs from the default, also add a repository
   **variable** (not a secret) named `TFNSW_TRIP_UPDATE_URL`.

4. **Run it once by hand.** Actions tab → *Collect delay observations* → *Run
   workflow*. The `collected-data` branch is created automatically on that first
   real run; check it for a new file under `observations/<today>/`. If the run
   succeeds but writes nothing, the endpoint or the route filter is wrong — not
   the schedule.

### Cadence

Every 5 minutes, GitHub's floor for scheduled workflows. The scheduler is
best-effort and drifts under load, sometimes by 10+ minutes. That is why the
collector keeps the next **three** upcoming stops per trip rather than only the
next one — a trip can pass two stops between polls, and the extra rows cover the
gap.

### Alternatives

**An always-on box** (Raspberry Pi, old laptop, free-tier VM) gives finer
granularity and no dependence on GitHub's scheduler. Use the SQLite sink and run it
under systemd:

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

**tmux on your laptop** is fine for a first afternoon and nothing longer — it
collects only while the machine is awake, which for a laptop means "not overnight".

Running both is not wasteful: they write to different sinks, and overlapping
coverage is insurance against one silently stopping.

## 7. Weekly check (5 minutes)

```bash
transit-poller --status              # local SQLite collector
transit-poller --status --sink csv   # snapshots (run on the collected-data branch)
```

It prints the total collected, the service dates covered, and the last ten polls.

- Is the total still climbing at roughly the expected rate (~25k rows/day)?
- Are recent polls `ok`? Consecutive `error:` rows mean collection has stopped.
- Are rows non-zero during peak hours? Zero at 3am is normal; zero at 8am is not.

For the Actions runner, the equivalent check is the workflow's run history — a red
run, or no runs for a few hours, both mean the same thing. Log the running total
against the Week 6 checkpoint in
[`04-implementation-plan.md`](./04-implementation-plan.md) §2.

## 8. Not built yet

Ingestion, retrieval, the agent loop, the MCP server, reconciliation, model training
and the evaluation harness. Their setup steps land here as each is built.
