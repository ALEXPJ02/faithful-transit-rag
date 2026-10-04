# The Disruption Label

> The ground truth that RQ2's detectors are scored against. For each line, T1 or T4,
> and each 15-minute window: was the line disrupted? Built on 2026-10-05 from the
> definition decided on 2026-09-28 ([`08-evaluation-plan.md`](./08-evaluation-plan.md)
> §3.2). The definition is pending the supervisor's confirmation, so every threshold is
> a parameter, and a change she asks for is a re-run.

## 1. The definition, as built

A line × window is **disrupted** if either rule holds:

- **(a)** an *incident* names the line while the feed carries it. Incidents come from
  `transit-alerts`: the alerts the incident rule calls disruptions, with their
  republications merged, overrides applied
  ([`09-service-alerts.md`](./09-service-alerts.md) §7);
- **(b)** at least a quarter of the line's services observed in the window are more
  than five minutes late.

| Parameter | Default | Where it comes from |
| --- | --- | --- |
| Window | 15 min | Agreed with the supervisor |
| Late | **more than** 300 s | TfNSW's on-time tolerance (Audit Office of NSW, 2017) |
| Share late | at least a quarter | Modelling choice, decided 2026-09-28 |
| Fewest services rule (b) may fire on | 5 | **Proposed 2026-10-05** (§3) |
| Observations used | final observation within 1 stop of the event, and \|delay\| ≤ 2 h | The delay model's own bounds (`quality.py`) |

## 2. How each rule is computed

**Rule (a) is timed by feed presence**, from `first_seen_utc` to `last_seen_utc`, never
by `active_period`. No unplanned alert has ever carried an end time. A window is marked
if it overlaps the incident's presence **on that line**. The span is per line because
Edgecliff named T1 in one alert, carried for one poll, and named T4 for six and a half
hours. Within a line, the span runs from the first alert to the last, so the gap a
republication leaves is still covered. An alert seen in a single poll marks one window.

**Rule (b) judges each service by its latest observation in the window.** A service is
one trip on one service date. A train 400 s late at 10:01 that recovers to 200 s by
10:10 was not late in the 10:00 window. This is the adaptation `docs/08` §3.2 names.
TfNSW judges a train once at its destination, and a window cannot wait for that. The
required count of late services is `ceil(share × n)`, counted rather than compared as
a ratio, so a threshold like 30% cannot drift in floating point.

**A window's service date turns over at 03:00 Sydney wall-clock time**, which is the
collector's own rule (`parsing.SERVICE_DAY_START_HOUR`). Measured on the 2026-10-05
snapshot, every stop event before 03:00 belongs to the previous service date and none
after it does. The subtraction happens on the wall clock. On 2026-10-04, when daylight
saving began, 03:30 AEDT therefore belongs to the 4th, as it does for the collector.

## 3. Which windows get a label

A window is labelled only if both rules could have fired on it. Everywhere else the
label is `pd.NA`, never `False`.

- **Alert collection was running.** Windows start at the first complete window after
  the first alert poll, which is 2026-09-15 00:45 Sydney time. Before that, rule (a)
  cannot be known. A label that meant "rule (b) only" on the early dates would change
  meaning partway through the chronological split, so those windows are not built.
  The audit also reports the longest gap between alert polls. On the 2026-10-05
  snapshot it is 30 minutes, so there was no outage.
- **At least one service of the line was observed.** With no train, there is nothing to
  be late and nothing to predict. On the 2026-10-05 snapshot, 134 windows had none, all
  overnight, and **none of them is an incident window**. A closure would put one here,
  and the audit counts them so that one cannot vanish unnoticed.

**Rule (b) cannot fire on fewer than five services.** *Proposed 2026-10-05, pending
confirmation with the rest of §3.2.* "A quarter of the line's services" is degenerate on
a thin window: at 02:00, one late train out of one is 100%. Five is the smallest count
at which one late train cannot reach a quarter on its own. It is an a priori bound, not
fitted to the data.

On the data to 2026-10-04, the rule with no minimum would also fire on 28 windows.
Twenty-six of them are between 01:00 and 04:30. The other two are T4 at 11:00 and 12:00
on 2026-09-22, during and just after the Bondi Junction repairs, when T4 ran three and
then four services.

Below the minimum, rule (b) is `False`, but the window keeps its label. Rule (a) can
still mark it, and the 11:00 window above is one rule (a) does mark. An incident thins
the service, so excluding thin windows from the population would drop exactly the
windows that matter. `--min-services 1` gives the rule as originally written.

## 4. What it gives

Snapshot of 2026-10-05, with the overrides of that date:

| Line | Windows | Labelled | Disrupted | Rule (a) | Rule (b) | Both |
| --- | --- | --- | --- | --- | --- | --- |
| T1 | 1,918 | 1,867 | **87 (4.7%)** | 74 | 40 | 27 |
| T4 | 1,918 | 1,835 | **80 (4.4%)** | 37 | 72 | 29 |

A base rate near 4.5% is a usable label. The first version of rule (a) put 100% of
windows in the positive class (`docs/08` §3.2).

## 5. Where the two rules disagree, and why both are kept

**Rule (b) confirms four incidents, touches a fifth, and never fires in the other seven.**

| Incident | Line | Windows | Of them, rule (b) | Highest share late |
| --- | --- | --- | --- | --- |
| Edgecliff, 21 Sep | T4 | 26 | 24 | 100% |
| North Sydney, 21 Sep | T1 | 22 | 12 | 57% |
| Harris Park, 27 Sep | T1 | 17 | 14 | 65% |
| Bondi Junction repairs, 22 Sep | T4 | 7 | 5 | 83% |
| Chatswood, 25 Sep | T1 | 9 | 1 | 26% |
| The other seven | T1, T4 | 1–9 each | **0** | at most 22% |

Two things keep rule (b) silent through a real incident:

- Many alerts are posted after the event, saying "... earlier", and stay in the feed
  through the recovery.
- T1 is a long line with many branches. An incident on one branch, such as Marayong or
  a person on the tracks at Central, never puts a quarter of the whole line's services
  more than five minutes late.

**Rule (b) also fires where no incident was posted.** This happens on 13 T1 windows
and 43 T4 windows. Most of the T4 ones fall in one week. T4's share of stop events more
than five minutes late was 3.2–4.4% each day from 26 Sep to 1 Oct, against 1.0–1.8%
from 23 to 25 Sep. On 29 Sep rule (b) marks seven T4 windows between 08:45 and 14:00.
The only unplanned T4 alert that day was first seen at 17:06, reporting urgent repairs
at Bondi Junction as completed. Rule (b) also found an incident the alert rule had
missed: Harris Park, which TfNSW published as `MAINTENANCE` and which is now in by
override ([`09`](./09-service-alerts.md) §7).

So the two rules measure different things: the operator's account and the passengers'
experience. Rule (a) alone would miss T4's late week. Rule (b) alone would miss seven
of the twelve incidents. That is the case for the union. It is also why detection is
worth reporting **against each rule separately** as well as against the union. A
detector built on delay features can only be expected to find what rule (b) sees.

## 6. Running it

```bash
transit-label --db data/delay_observations_20261005.db                  # audit; writes nothing
transit-label --db data/delay_observations_20261005.db \
              --out data/disruption_labels.csv                          # also write the labels
transit-label --db ... --min-services 1                                 # the rule with no minimum
transit-label --db ... --late-share 0.3 --late-threshold-s 180          # a threshold she might ask for
```

The audit prints windows and rule counts per line, where the rules agree, the windows
with no observed service (and whether any is an incident window), what the minimum
removes, each incident's windows, and disrupted windows per service date. The snapshot
is opened read-only.

## 7. Not built yet

- **The cancellation check** (`docs/08` §3.2). It re-runs rule (b) counting cancelled
  and skipped-stop services as late, on the dates from 2026-09-29. A cancelled trip has
  no stop updates, so placing it in the windows it should have run needs its timetabled
  stop times, from the static bundle of its service date.
- **The classifier's target and features** (`docs/08` §3.3). The target is whether
  the line is disrupted in the next 30 minutes, taken from these window labels. The
  features are aggregates over the windows up to the prediction time, never the
  windows the label comes from.
