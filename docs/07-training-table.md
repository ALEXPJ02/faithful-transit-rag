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

## Three things that are not obvious

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

- **No alert flag.** The feature list includes an active-alert indicator, but
  nothing collects the Service Alerts feed yet. It is not in the table rather
  than being present and always false.
- **`RTTA_*` trips are excluded upstream.** Out Of Service and Non Revenue
  movements never reach the table. A service *altered* beyond what the timetable
  can express may also be filed that way and go uncollected — unmeasured, and it
  belongs in the write-up as a stated limitation.
- **Peak boundaries are a modelling choice**, not a fact: weekdays 06:00–09:59
  and 15:00–18:59, set in `reconcile.py` as named constants so the write-up can
  state them.
