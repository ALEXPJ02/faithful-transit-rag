# The Reasons Stage

> RQ1 Objective 3: the likely cause of a disruption, grounded in TfNSW's past alerts,
> and the RQ2 scoring that says whether retrieval helps. Built on 2026-10-05 on the
> alert corpus and guarded retrieval of [`10-retrieval.md`](./10-retrieval.md) §5.

## 1. What it is given, and what it never sees

An incident's own alert carries its cause, and that cause is the answer being scored
([`08-evaluation-plan.md`](./08-evaluation-plan.md) §3.5). So the reasons stage is
given a **situation** built from the delay feed alone (`agent/situation.py`). It
holds the line, the moment of the incident's first alert in Sydney time, and how many
services were observed in the 30 minutes before. It also holds how many were late and
how late the worst was, when last reported. Last comes **where**: the stations where
services were reported late at any point in that half hour, named the way past alerts
and riders name them ("Central", not `2000331`).

> T1, 13:45 on Sunday 27 September 2026 (Sydney time). In the 30 minutes before, 35
> services were observed. When last reported, 6 were more than five minutes late and
> the most delayed was 19 minutes behind. At some point in those 30 minutes, services
> were reported more than five minutes late at Parramatta (4), Harris Park (3),
> Lidcombe (3), Auburn (2), Granville (2).

The wording says which report each figure reads. Until 2026-10-05 it did not, so a
station could seem to hold more late services than the total
([`15`](./15-faithfulness-judge.md) §5).

That is the Harris Park signal repairs, with the operator's alert still unread. The
situation uses the label's own bounds: the final observation within one stop, and
|delay| ≤ 2 h. A service is judged by its latest observation, as in rule (b). Nothing
at or after the moment is read, and a test pins that the boundary is strict.

## 2. Retrieval, under both guards

`Situation.describe()` is the query. The alert index returns the five most similar
past **incidents**, one passage per incident, so republications cannot fill two
slots. Both guards of `08` §3.5 apply:

- `seen_before` admits only incidents first seen before this one;
- `exclude_incident` removes the incident itself, whatever its alert ids.

The run refuses an index built from different incidents or a different embedding
model, and names the rebuild command. An index that silently missed the newest
incidents would score retrieval on a corpus nobody chose.

## 3. The answer

The model answers through structured output validated against `ReasonAnswer`: a
cause group, one or two sentences, and the alert ids it relied on. A group outside
the four (`08` §3.2) cannot be returned. Citations are then checked against the
passages actually shown. An id that was not shown is counted as an **unsupported
citation**, because inventing evidence is the failure the faithfulness metric
exists to catch. A refusal or an API failure is recorded as a failed answer, which
counts as wrong.

## 4. The three systems (`08` §3.3)

| System | Sees |
| --- | --- |
| Most common cause, **time-aware** | The cause groups of incidents first seen before this one. A baseline allowed to count the future would leak more than the system it is compared with |
| The model, no retrieval | The situation |
| The model + retrieval (the RAG approach) | The situation and five guarded past incidents |

The model is the configured generation model (`ANTHROPIC_GENERATION_MODEL`,
currently `claude-sonnet-5`). **Its sampling cannot be fixed.** The current models
reject temperature, so `08` §3.5's "three runs at fixed temperature" becomes three
runs at the default sampling, reported as mean ± sd.

## 5. A development run, 2026-10-05

`transit-reasons --db data/delay_observations_20261005.db --model --repeats 3`, on all
12 incidents. Each incident is explained using only the incidents before it. This is a
development run on every incident, not the scored run on the test split.

| System | Accuracy | Macro-F1 |
| --- | --- | --- |
| Most common cause (time-aware) | **0.42** | **0.20** |
| `claude-sonnet-5`, no retrieval | 0.14 ± 0.05 | 0.11 ± 0.05 |
| `claude-sonnet-5` + retrieval | 0.28 ± 0.05 | 0.17 ± 0.03 |

The model rows are the mean ± sd of three runs at the default sampling. There were 72
answers, none failed, and **no citation of an alert that was not shown**. The run used
90k input and 15k output tokens. Retrieval recall@5 is **0.75**: for 9 of 12
incidents, at least one of the five past incidents shares the true group, and the
first incident has no past at all. A random draw of earlier incidents scores 0.74
(§7), so this is not evidence that the ranking works. Macro-F1 averages technical, network incident and
other/unknown, since no weather incident has occurred. Every answer is kept, locally,
in `data/reasons_dev_20261005.jsonl`. The run used the situation's earlier wording
(§1). A second run after the wording changed is the faithfulness judge's validation
set (`15` §4). Its accuracy, 0.17 ± 0.08 without retrieval and 0.25 ± 0.00 with it,
is in line with this one.

**At this size the three systems cannot be told apart.** Macro-F1 with 95% intervals
from resampling the 8 incident dates (`evaluation/stats.py`) is 0.20 [0.11, 0.39] for
the baseline, 0.11 [0.02, 0.25] without retrieval, and 0.17 [0.07, 0.35] with it. The
intervals overlap almost entirely.

**One run is not enough.** A single run on the same code scored 0.00 and 0.25 for the
two model systems, against 0.14 and 0.28 averaged over three. At n = 12 a single run
swings by more than the difference being measured, which is why `08` §3.5 asks for
repeats.

**What the answers show** (read from that file, not inferred from the scores):

- **Without retrieval the model abstains.** It answered other/unknown 30 times in 36,
  28 of them wrongly, typically saying the delay data alone indicates no specific
  cause. The delay pattern by itself says little about cause.
- **With retrieval it reasons by place, and that works for one kind of cause.** It was
  right in every run on three incidents: the 15 Sep North Shore fault, the North
  Sydney repairs and Harris Park. All three are technical faults at places with
  earlier technical incidents. It **never once identified a network incident**
  (police activity, an accident or a medical emergency, 5 of the 12), because those
  do not recur by place. Its answers were also more stable, with 10 of 12 incidents
  answered the same way in all three runs against 7 of 12 without retrieval.
- **The majority baseline still wins at n = 12**, because technical is the commonest
  group. Nothing at this size is significant, and none of it is a result. The scored
  run is on the test split at the cut-off.

**A decision this raises, for before the scored run.** The system prompt lets the
model answer other/unknown "when nothing indicates a cause". That is honest, but it
conflates *the evidence does not say* with the operator's own unknown cause, and
without retrieval it is most of what the model says. Whether to require the most
likely group instead is a development choice. It should be made on these incidents,
and never by looking at the test split.

## 6. Running it

```bash
transit-index build --source alerts --db data/delay_observations_20261005.db   # the index must match
transit-reasons --db data/delay_observations_20261005.db                       # free: cases, retrieval, baseline
transit-reasons --db ... --model --repeats 3 --out data/reasons_runs.jsonl     # three runs, every answer kept
transit-reasons --db ... --model --since 2026-10-14                            # only the incidents from a date
transit-reasons --db ... --sweep-k 1,2,3,4,5,6,8,10 --until 2026-10-12         # free: k against chance (§7)
```

## 7. The k sweep

`08` §3.5 freezes k on development incidents before the test dates are scored.
**Recall@k cannot choose k on its own.** Showing more incidents can only find more, so
recall always favours the largest k. `--sweep-k` therefore compares each k with
**chance**: the recall of the same number of earlier incidents drawn at random from the
pool the guards admit. Chance is exact, a hypergeometric "at least one", not simulated.
The sweep also shows the **on-cause share**, the mean share of the incidents shown that
share the true group, which is what each extra passage costs. The sweep retrieves once
per incident at the largest k, and a smaller k reads the top of the same ranking. It
calls no model, and it refuses to run if retrieval and the pool disagree about which
incidents were admissible.

**On the 12 development incidents, 2026-10-05:**

| k | Recall@k | Chance | Lift | On cause | Shown |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.33 | 0.37 | −0.04 | 0.36 | 0.9 |
| 2 | 0.50 | 0.56 | −0.06 | 0.32 | 1.8 |
| 3 | 0.75 | 0.67 | +0.08 | 0.42 | 2.5 |
| 4 | 0.75 | 0.71 | +0.04 | 0.42 | 3.2 |
| 5 | 0.75 | 0.74 | +0.01 | 0.40 | 3.8 |
| 6 | 0.75 | 0.75 | +0.00 | 0.42 | 4.2 |
| 8 | 0.75 | 0.75 | +0.00 | 0.41 | 5.0 |
| 10 | 0.75 | 0.75 | +0.00 | 0.40 | 5.4 |

- **0.75 is a ceiling, not a score.** Three incidents have no earlier incident of their
  group, so no k can help them: the first incident, the first network incident and the
  only other/unknown one. Recall reaches the ceiling at k = 3.
- **The ranking does no better than chance.** At every k the lift is within one
  incident (1/12 = 0.08) of a random draw, and at k = 1 and 2 it is slightly below.
  §5's recall@5 of 0.75 is chance's 0.74. It must never be reported without its chance.
- **Why:** the situation names stations, so the nearest past incidents are the nearest
  places. Place predicts a recurring technical fault and says nothing about a network
  incident (§5). On-cause share stays near 0.4 at every k, which is about the base
  rate.

**The proposed rule, for the author's confirmation.** At the cut-off, re-run the sweep
on every incident before the test split (`--until` the first test date). Freeze k as
the smallest value whose recall@k reaches the sweep's highest recall. Today that gives
3. The plan's 5 stays in force until the rule is confirmed and applied. A smaller k
might also help faithfulness, since retrieval lowered it in development (`15` §6). That
is a question for the validated judge, not something to assume.

## 8. Not built yet

- **The judge's validation.** The judge is built ([`15`](./15-faithfulness-judge.md)).
  Its Cohen's κ against the author waits on the author's labels for a blind sample.
- **Freezing k** at the cut-off, by the rule in §7, once the author confirms it.
- **The scored run**, on the incidents in the test split drawn at the cut-off, with the
  count of incidents and cause groups beside every number.
