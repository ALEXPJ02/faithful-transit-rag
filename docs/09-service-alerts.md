# Service Alerts

What the TfNSW Service Alerts feed actually sends, and why the collector stores
it the way it does. Everything in §1 was measured against the live feed on
2026-09-14, not read off the spec — several of the values differ from what the
spec would let you assume, and each difference changed a design decision.

## 1. What one poll contains

`https://api.transport.nsw.gov.au/v2/gtfs/alerts/sydneytrains` — 89,513 bytes:

| | |
| --- | --- |
| Alert entities | **41** |
| `informed_entity` selectors | **908** |
| `active_period` ranges | **55** (27 alerts × 1, 14 × 2) |
| Scopes after flattening | **1,051** (selectors × periods) |

| Field | What TfNSW actually sends |
| --- | --- |
| `entity.id` | UUIDv5, 41 distinct. **A stable key already exists** — do not synthesise one |
| `informed_entity` | `agency_id` + `route_id` + `direction_id` on all 908; `stop_id` on 590. `trip` never set |
| `active_period` | Explicit `start` **and** `end` *on this sample, which was entirely trackwork*. **It does not generalise:** measured on 134 alerts to 2026-09-24, every one of the 30 unplanned alerts has `active_period_end = 0` (unbounded) across all 2,681 of their scopes, against 361 of 1,591 scopes for `MAINTENANCE`. A disruption labeller therefore cannot bound an incident from `active_period` and must derive an end from feed presence — see [`08-evaluation-plan.md`](./08-evaluation-plan.md) §3.2 |
| `cause` / `effect` | 38 MAINTENANCE / MODIFIED_SERVICE, 3 UNKNOWN |
| `severity_level` | **`UNKNOWN_SEVERITY` on all 41.** TfNSW never populates it |
| `header_text` | One `en` translation, 53–76 characters |
| `description_text` | **Two translations: `en` (~200–400 chars) and `en/html` (~1.1–1.5 KB)** |
| `url` | Set, e.g. `https://transportnsw.info/alerts/details#/ems-77752` |

## 2. Three findings that changed the design

### The route namespace is a superset of the realtime bundle

**292 of 908 selectors (32%) name a `route_id` that `data/routes_lookup.csv`
does not contain.** This is *not* a stale-bundle problem — `NSN_*` and `WST_*`
match exactly. The absentees are intercity and regional services the For
Realtime bundle simply does not describe: ~100 `4T.C.*` and `4T.T.*` ids, plus
`APS`, `BNK`, `IWL` and `SMNW`.

Consequently **alerts are collected unfiltered**. Filtering to T1/T4 at
collection time would keep 104 of 908 selectors and **13 of 41 alerts** —
discarding 68% of the feed permanently. Collection is irreversible;
reconciliation is not, so the filtering happens in `reconcile.py` where it can
be re-run. `extract_alerts` takes no `tracked_routes` parameter at all, which
makes that decision unforgeable rather than a convention.

### `description_text` arrives twice, and the order is not guaranteed

Every alert carries the same description as both `en` prose and an `en/html`
fragment. `translation[0]` happens to be the prose today. Selecting by index
would, on whatever day TfNSW reorders them, put ~1.3 KB of markup into every
row of the column the RAG layer reads as text — silently, and in a table that
cannot be recollected. `_translated_text` selects by language instead.

### Severity is a constant dressed as signal

41 of 41 alerts report `UNKNOWN_SEVERITY`. The column is stored for
completeness and for the day TfNSW starts populating it, but **no feature may
be built on it** — a severity-weighted model input would be a constant.

## 3. How it is stored

Two tables, because one alert averages 22 selectors and a flat table would
store a 400-character description once per selector per poll.

```
service_alerts   -- ~41 rows.    PK: alert_id
alert_scopes     -- ~1,051 rows. PK: (alert_id, route_id, direction_id, stop_id, active_period_start)
alert_poll_log   -- one row per alert poll
```

Three decisions worth not relitigating:

- **Every key column is `NOT NULL` with a sentinel** (`''`, `-1`, `0`). SQLite
  treats NULLs in a composite primary key as distinct from each other, so a
  nullable key column silently stops deduplicating — here, worth about a
  thousand rows a poll. Same trap `stop_sequence` already carries a sentinel for.
- **`active_period_start` is in the key; `active_period_end` is not.** A
  publisher extending a trackwork window should update the row; an alert with
  two nightly windows is genuinely two scopes.
- **`first_seen_utc`/`last_seen_utc` are an independent signal.** `active_period`
  is the publisher's *claim*; the observed window is what the feed actually
  carried, and it is the fallback when that claim is absent or wrong. This is
  the real cost of a coarse poll cadence, and the reason not to raise
  `ALERTS_POLL_EVERY_N_POLLS` much past 15.

**`alert_poll_log` is a separate table, not a `feed` column on `poll_log`.**
`poll_log.poll_time_utc` is a PRIMARY KEY written with `INSERT OR REPLACE`, so
an alert poll sharing a timestamp would overwrite the trip-update row —
destroying the only signal `--status` and the volume checkpoint read. Widening
`poll_log` would be worse still: `CREATE TABLE IF NOT EXISTS` leaves the live
collector's existing table in place, so the widened insert would fail on *every*
trip-update poll.

## 4. Cadence

`ALERTS_POLL_EVERY_N_POLLS=15` — 15 × 120 s, so every 30 minutes.

**This is not a quota decision, and should not be written up as one.** Trip
updates burn 720 calls a day; alerts at N=15 add 48, against an allowance of
60,000 — 1.3%. Recording it as a budget constraint invites someone to raise it
to 60, and the actual cost of a coarse cadence is the resolution of the observed
activity window above. The reason for 30 minutes is that trackwork windows run
for hours, so finer polling records nothing more.

## 5. Running it

```bash
transit-poller --probe-alerts   # what the feed holds right now; writes nothing
transit-poller --once           # one trip-update poll and one alert poll
transit-poller --status         # alert and scope counts, plus both poll logs
transit-poller --no-alerts      # trip updates only, for a local run
```

## 6. The training-table flag is deliberately not built yet

`docs/07-training-table.md` still lists the active-alert feature as absent, and
that stays true until there is a usable overlap window.

Delay collection began 2026-09-03; alert collection began the day this shipped.
Adding `has_active_alert` now would give a column that is `False` across the
entire back-catalogue — not because there were no alerts, but because nothing
was looking. The chronological split would put that structural break *inside*
the training window, and the model would learn "alerts never occur before date
X": exactly the silent-score-inflation class `prediction/model/dataset.py`
exists to prevent. Rows predating alert collection must carry `pd.NA`, never
`False`.

When it is built, two constraints are already known:

- **Join on `route_short_name`, not `route_id`.** T1 spans 8 route ids; an
  alert naming 4 of them would flag half the trips on one line under identical
  conditions — a feature inconsistent with itself. Line level is also how a
  rider thinks, which is what the agent layer will ask.
- Predicate: `alert_scopes.route_short_name = event.route_short_name AND
  (start = 0 OR start <= t) AND (end = 0 OR t < end)`, falling back to the
  observed `[first_seen_utc, last_seen_utc]` window when the claimed period is
  unbounded.

## 7. From alerts to incidents

`transit-alerts audit --db <snapshot>` (`ingestion/alerts.py`) answers the two
questions under both the disruption label (`docs/08` §3.2 rule (a)) and the
reasons corpus: **which alerts are disruptions, and which alerts are the same
disruption.** Decided 2026-09-29, pending the supervisor's confirmation. It
opens the snapshot read-only.

### Which alerts are disruptions

Cause alone gets it wrong in both directions. Measured on the 22 non-maintenance
alerts naming T1 or T4 to 2026-09-28:

| Alert | Published cause | What the description says | By cause alone |
| --- | --- | --- | --- |
| 2026-09-15 23:54 (×2) | `UNKNOWN_CAUSE` | "no train running at 1:47am … due to trackwork" | wrongly a disruption |
| 2026-09-21 23:46 | `UNKNOWN_CAUSE` | "trackwork replacement buses will set down …" | wrongly a disruption |
| 2026-09-25 04:51 | `UNKNOWN_CAUSE` | police operation closing **streets**; bus connections affected | wrongly a disruption |
| 2026-09-21 16:11 | `UNKNOWN_CAUSE` | "Trains are not running between Bondi Junction and Central …" | a disruption — keep it |

So the rule reads the description, in order: a manual override; `MAINTENANCE`
cause; more than 24 h in the feed (a standing notice); planned-work wording
(`trackwork`, `planned work`, `installation works`) excludes; service-impact
wording (`extra travel`, `not running`, `delay`, `running late`, `resumed`, …)
includes; anything else is a notice. "Replacement bus" is deliberately **not**
a planned marker, because an unplanned incident can also say "buses are
replacing trains".

**The rule was written by reading those 22 alerts, so it is checked on alerts it
never saw.** Everything first seen from 2026-09-29 is marked `*` in the audit
(`--held-out` shows only those); the author reads them against the same
criteria and the rule's accuracy on them is reported. A wrong call is corrected
in the tracked `data/alert_overrides.csv` — alert id, yes/no, and a written
reason — never by editing the markers after seeing the data.

### Which alerts are the same disruption

TfNSW republishes an incident under a new `entity.id` whenever its scope or text
changes, and the old one leaves the feed. Two alerts are one incident when they
share a line, one leaves the feed within **35 minutes** (one 30-minute alert poll
plus slack) of the other appearing — a *handover* — and the causes match **or one
of them is `UNKNOWN_CAUSE`**, because Edgecliff was first posted as unknown and
then twice as police activity. Chains are merged transitively.

Overlap alone does not merge. Two different incidents can run on one line at the
same time with the same cause, and then both stay in the feed for their whole
length rather than handing over (found by Cursor Bugbot). The handover is checked
both ways round because it can happen within one poll: Edgecliff's two first
alerts were first seen together, and the unknown-cause one left at once.

An incident's cause is the **first specific cause any of its alerts names**, and
unknown only if none ever does: Edgecliff is police activity; Chatswood
(2026-09-25, three alerts, all unknown) stays other/unknown.

### What it gives, to 2026-09-28

| | |
| --- | --- |
| Alerts naming T1 or T4 | 46 (24 `MAINTENANCE`) |
| Standing notices (> 24 h in the feed) | 4 — station works, missing tactiles, the Tangara notice, a trackwork bus stand |
| Planned work or no train impact | 4 — the table above |
| **Incidents** | **8, on 5 Sydney dates** — 15 Sep (3), 21 Sep (2), 22 Sep, 25 Sep, 26 Sep |
| Cause groups | technical 4, network incident 3, other/unknown 1, weather/external 0 |

Those 8 incidents come from 14 alerts: Edgecliff and Chatswood are three alerts
each, North Sydney and Martin Place two each. The same rule over the data to
2026-09-24 gives 6 incidents on 3 dates, which is what `docs/08` §3.2 counted by
hand.
