# The Training Table

> Collection records what the feed *said*. Reconciliation works out what
> actually happened. This document is the contract between the two.

## Running it

```bash
transit-reconcile --report-only
```

```bash
transit-reconcile --split
```

Reads from whichever sources exist — the always-on collector's SQLite database
and the scheduled collector's CSV snapshots — and deduplicates across them, so
having both is redundancy rather than double counting. Writes
`data/training_table.csv` (`.parquet` if the output path says so), and with
`--split`, three more files partitioned by service date.

## The central idea

GTFS-Realtime reports a stop only while the train has yet to reach it. Once the
vehicle passes, that stop leaves the feed. So the **last observation naming a
stop event is the closest available proxy for what actually happened**, and
`stops_ahead_final` records how close to the event that last prediction was
made — one stop out is nearly an outcome, twenty stops out is a guess. Both are
kept; the column is what lets analysis tell them apart.

This is also why reconciliation is a separate step from collection. It can be
rerun, rewritten and got wrong as many times as necessary. Collection cannot.

## Schema

| Column | Meaning |
| --- | --- |
| `service_date` | Sydney service day (3am boundary), the split key |
| `trip_id`, `stop_id` | The stop event's identity |
| `route_id`, `route_short_name` | e.g. `NSN_2i`, `T1` |
| `stop_sequence` | Position within the trip, **from the static bundle** — null when unmatched |
| `scheduled_arrival_s` | Seconds after service-day start; null when unmatched |
| **`delay_s`** | **The target.** Arrival delay, falling back to departure |
| `arrival_delay_s`, `departure_delay_s` | The components, kept separately |
| `prev_stop_delay_s` | Delay at the preceding stop of the same trip |
| `stops_ahead_final` | How far out the *last* observation was made |
| `observation_count` | How many observations backed this event |
| `observed_at_utc` | When the final observation was taken |
| `hour_local`, `day_of_week`, `is_weekend`, `is_peak` | Sydney local time features |
| `schedule_matched` | Whether the static bundle knew this trip |

## Four things that are not obvious

### `stop_sequence` comes from the timetable, not the feed

TfNSW does not populate `stop_sequence` in the realtime feed — every raw
observation carries the `-1` sentinel. Stop order therefore has to come from the
static bundle's `stop_times.txt`, which is also the only source of scheduled
arrival times.

### The schedule join is lossy, and the loss rate drifts

A realtime `trip_id` looks like `162F.1396.159.32.A.8.90986110`, where
`1396.159.32` encodes the timetable and version the trip was planned under.
Trips already running when a new timetable is published keep the old version and
are absent from the current bundle. Measured against live data: **89% of trips
matched** (96% of rows), and every miss carried a superseded version.

Unmatched trips are **kept, not dropped**. Their delays are real observations,
and discarding a whole class of trips would bias the set toward whatever
timetable happened to be current. They lose `scheduled_arrival_s`, and their
stop ordering falls back to observation time — a train's later stops leave the
feed later, which recovers visit order well enough to preserve
`prev_stop_delay_s`.

The rate is reported on every run. **A falling number means the bundle has aged
— re-fetch it** with `python -m transit_rag.prediction.collection.routes --fetch`.

### A 24-hour delay is the feed, not a train

GTFS-Realtime sometimes republishes the previous day's run stamped with today's
`start_date`. Reconciliation then compares yesterday's stop times against
today's schedule and records a delay of roughly a full day. These are not slow
trains and they are not rare disruptions — they are an artifact of how the feed
identifies a trip.

`transit-train` drops them before the split, at
`quality.MAX_PLAUSIBLE_DELAY_S` = **7,200 s (2 hours)**. Past two hours a Sydney
Trains service is operationally a cancellation or a replacement, not a late
train, so the bound is a statement about what `delay_s` is allowed to *mean* —
decided from how the network runs, not from the shape of the tail. That
distinction matters: a threshold reverse-engineered from the data is one a
reviewer is right to distrust.

The data is nowhere near it. On the 204,628-row table of 2026-09-16:

| | |
| --- | --- |
| p99.99 of `delay_s` | 3,354 s (56 min) |
| Largest **plausible** delay | 4,394 s (73 min) |
| Next value above it | **79,422 s (22.1 h)** |
| Rows excluded | **7** (0.003%), all one trip |

Nothing falls between 73 minutes and 22 hours, so any bound from ~1.5 h to ~20 h
removes exactly the same rows — the result does not depend on where in that
range the number sits.

Two properties of *how* it is applied matter more than the number:

- **`prev_stop_delay_s` is bounded too.** The naive baseline predicts straight
  from that column, so an implausible value there is an implausible *baseline*
  prediction, and the model would be beating a strawman on those rows. No row
  currently has a poisoned feature and a clean target; the guard is there for
  the ghost trip whose first stop is corrupt and whose second is not.
- **It runs before the split.** Filtering after the boundary is drawn — or
  filtering only test — changes what each partition means and is
  indistinguishable from keeping the rows that flatter the result.

**It does not improve the reported result, and that is the point.** Measured on
the 14-date run of 2026-09-16 — not the current model, whose figures are in
`08-evaluation-plan.md` §2 — all seven rows landed in validation, so the bound
took validation MAE from **32.1 s to 14.5 s** while leaving that run's test MAE
(16.02 s), RMSE (45.11 s), baseline MAE (18.17 s), MASE (0.881) and the selected
hyperparameters bit-for-bit unchanged.
What it fixes is interpretability: without it, validation looks twice as hard as
test for no stated reason, and the true reason is six rows. `--keep-implausible`
reproduces the unfiltered figures so the exclusion stays auditable.

### The split is chronological, and splits whole days

A random shuffle would put observations from the same afternoon on both sides of
the boundary, and the model would score well by having already seen the
conditions it is asked to predict. That is the standard way time-series results
get quietly inflated, and it would invalidate the comparison against the
naive-persistence baseline.

Splitting on whole service dates rather than rows matters for the same reason: a
day straddling the boundary leaks identically, just less visibly.

## Reading the quality report

```
Rows: 252   service dates: 1 (2026-09-03 to 2026-09-03)
Distinct trips: 84
Rows per line: T1=158, T4=94
Observed within 1 stop(s) of the event: 182 (72%) — these are the reliable outcomes
Backed by more than one observation: 216 (86%)
Matched to the static timetable: 242 (96%)
Have a previous-stop delay: 168 (67%) — the first stop of each trip cannot have one
Delay seconds — median 0, p90 153, max 411
Rows per service date — min 252, median 252, max 252
```

- **Observed within 1 stop** is the honest size of the dataset. Rows beyond that
  are predictions being scored as outcomes and should be filtered or weighted.
- **Backed by more than one observation** shows whether polling was frequent
  enough to watch trains approach. It collapses when collection is sparse.
- **Rows per service date** flags collection gaps: a day far below the median
  gets an explicit warning, because a gap otherwise looks like an ordinary day
  with fewer rows.

## Known gaps

- **No alert flag yet, and deliberately so.** Service Alerts *are* now collected
  (see [`09-service-alerts.md`](./09-service-alerts.md)), but delay collection
  began 2026-09-03 and alert collection began later. Adding `has_active_alert`
  now would produce a column that is `False` across the whole back-catalogue —
  not because there were no alerts, but because nothing was looking — and the
  chronological split would put that structural break inside the training
  window. The flag lands once there is a usable overlap; rows predating alert
  collection must carry `pd.NA`, never `False`.
- **2026-09-11 to 09-15 have no schedule join, permanently.** TfNSW's static API
  serves only the era that is current; an era published and superseded inside
  that window was never fetched, so ~77,000 stop events across five service
  dates carry no `scheduled_arrival_s` and no `stop_sequence`. A daily archive
  now keeps every era (`docs/06-always-on-collector.md`), but the gap itself is
  unrecoverable and belongs in the write-up as a stated limitation.

  **`transit-train` drops these whole dates before the split**, at
  `quality.MIN_SCHEDULE_COVERAGE`; `--keep-schedule-blind` reproduces the
  unfiltered figures so the exclusion stays auditable. The rows remain in the
  table — their delays are real and reconciliation keeps them — but a partition
  built from them measures a nine-feature model on seven features.

  The exclusion does **not** flatter the result, and the check matters more than
  the claim. On the 22-date table of 2026-09-24 the filter cannot reach the test
  split at all: both settings score the identical 43,800 test rows against the
  identical baseline, and MASE moves **0.834 → 0.828**. What it fixes is which
  rows *select* the model, not which rows score it.

  Two caveats to state rather than bury. `stop_sequence` carries 3.1% of gain
  with the blind dates kept and 7.7% with them dropped, so citing its importance
  as the reason to drop them argues in a circle; `scheduled_arrival_s` is the
  least important feature in the model either way (0.4–0.5%). And the filter
  halves validation, from four dates to 09-20 and 09-21, one of which is a
  disruption day — so hyperparameter selection now rests on a two-day set whose
  mean delay is well above test.
- **`RTTA_*` trips are excluded upstream.** Out Of Service and Non Revenue
  movements never reach the table. A service *altered* beyond what the timetable
  can express may also be filed that way and go uncollected — unmeasured, and it
  belongs in the write-up as a stated limitation.
- **Peak boundaries are a modelling choice**, not a fact: weekdays 06:00–09:59
  and 15:00–18:59, set in `reconcile.py` as named constants so the write-up can
  state them.
