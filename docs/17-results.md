# 17. The scored run, at the 2026-10-18 cut-off

Every figure on this page comes from one frozen file, written once after the cut-off.
The decisions they depend on were all made before the test split was drawn: the label
and its thresholds (`08` §5), the retrain window, and the reasons prompt (`13` §5). The
judge and its rubric were fixed by validation (`15` §4). The figures elsewhere in these
docs are development figures. These are the results.

**More models were compared afterwards, on the same file**
([`18`](./18-model-comparison.md)). The supervisor asked for the comparison after this
run. Its grids were frozen before any added model saw test, it reproduces every figure
on this page that it touches, and it replaces none of them. It does qualify one: the
delay model's gain over persistence, which a paired interval cannot tell from zero (§4).

**The scored run was made twice, and the second run is the result.** The first used the
alert rule as it stood on 2026-10-11. Reading its outputs showed that the rule's
held-out check had stopped on that date, although `09` §7 commits to continuing it to
the cut-off. Completed, the check found 18 alerts the rule had wrongly excluded. All
are urgent repairs that TfNSW published as `MAINTENANCE`, the Harris Park case `09` §7
had already decided. Each was corrected by an override written from the alert's own
text, and every stage that reads alerts was run again on the same frozen file. The
systems did not change: the code, the prompts, the k rule and the threshold rule are
the first run's. The delay model reads no alerts, so it was not re-run. §9 sets the
first run beside the second, so the correction's effect can be seen.

Both runs were made on 2026-10-20, by the chain rehearsed on a pretend cut-off on
2026-10-18.

## 1. The evaluation file

**The collector covered the cut-off day completely.** Between 00:00 on 18 October and
03:00 on the 19th (Sydney time), 802 of 802 trip polls succeeded, the longest gap
between them was 122 s, and 53 of 53 alert polls succeeded. The poller had been up since
2026-09-30 with no restart, and logged no error in the nine days before the pull.

**The snapshot** was taken on the instance by `transit-snapshot` at 23:06 UTC on 19
October. It held 696,618 stop events, 321 alerts and 9,611 scopes. Its sha256,
`f25587222e7391ca41801c6f0371d10ac8d8bb3f474cdd739bedbed101a40a43`, is identical on the
instance and locally, and `integrity_check` is ok. The 8 timetable eras not yet held
locally were pulled with it, and all 34 match the instance's checksums.

**The frozen file** is `transit-freeze --cutoff 2026-10-18` of that snapshot:

> `data/evaluation_20261018.db`, sha256
> `44e90956a60ae3140b71998413f057ce7070bafb59fa7091faa750f95b2532ac`

- **Removed:** the 21,785 stop events and 121 trip statuses of 19–20 October, 923 trip
  polls and 62 alert polls after 03:00 on the 19th, and 9 alerts first seen after it,
  with their 382 scopes.
- **Clipped:** 48 alerts still in the feed at the cut-off, to the last alert poll before
  it.
- **Checked:** the source's checksum did not move, and the frozen file's was re-checked
  after every later step of both runs.

**It holds 674,833 stop events over 46 service dates**, 2026-09-03 to 10-18: T1 403,185
and T4 271,648. 32,348 of 32,349 trip polls and all 1,624 alert polls succeeded. 18
October holds 8,531 stop events, about half a normal day. That is planned trackwork,
with buses replacing trains, as on 10 and 11 October. It is not a collection gap,
although `transit-reconcile` warns of one for 10 and 11 October.

## 2. The label, and the splits

**The label** (`transit-label --cancellation-check`) marks **237 of 3,170 T1 windows
(7.5%) and 216 of 2,944 T4 windows (7.3%) disrupted.** 34 incidents lie behind rule (a),
on 17 Sydney dates: 23 technical, 10 network incidents and 1 other/unknown. Weather
never occurs. The held-out check that produced 9 of them is in `09` §7.

**Counting cancelled and skipped-stop trains as late would still change the label.**
To 2026-10-04 that check changed no window. Over all dates from 2026-09-29, it would add
23 disrupted windows, 325 → 348, and 8 of them fall on the test dates. Before the
overrides it would have added 41. 19 of those lie inside the urgent repairs the
overrides made incidents, so the check had been pointing at them. The label stays as
decided, because it must mean the same on every date, and cancellations exist only
from 2026-09-29 (`08` §5 item 7). On those dates, rule (b) is a lower bound on what
TfNSW would count as late.

**Each stage drew its split once, by its own rule:**

| Stage | Train | Validation | Test |
| --- | --- | --- | --- |
| Detection, and the question set | 2026-09-14 to 10-07 (24 dates) | 10-08 to 10-13 (6) | **10-14 to 10-18 (5)** |
| Delay model | 09-03 to 10-06, less 09-11..15 (29) | 10-07 to 10-12 (6) | 10-13 to 10-18 (6) |
| Reasons | 26 incidents before 10-14, for the k sweep | | **8 incidents from 10-14** |

The delay model's split starts with collection, on 3 September, and drops the five
schedule-blind dates first. Detection's starts when alerts did. So the two test splits
differ by a day.

## 3. Detection (RQ2)

`transit-detect --db data/evaluation_20261018.db`. The test split has 904 windows whose
next 30 minutes are known, and **114 of them are positive**, against 7 on the
development split (`12` §6).

| Test, 114 positives | AP | 95% interval | Precision | Recall | F1 | False alarms / day | Recall, rule (a) | Recall, rule (b) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Persistence | 0.722 | [0.286, 0.829] | 89% | **78%** | **83%** | 2.2 | **86%** | **82%** |
| XGBoost | 0.745 | [0.285, 0.876] | 88% | 57% | 69% | **1.8** | 63% | 68% |
| Random forest | **0.799** | [0.305, 0.897] | 88% | 61% | 72% | 2.0 | 68% | 73% |
| XGBoost, no alert feature | 0.756 | [0.257, 0.854] | 75% | 67% | 70% | 5.2 | 74% | 75% |
| Random forest, no alert feature | 0.796 | [0.300, 0.889] | 86% | 60% | 70% | 2.2 | 68% | 69% |

The intervals come from resampling the 5 test dates. On validation the two detectors
tied, at AP 0.887 for the forest and 0.878 for XGBoost, so both are reported. The
agent's risk tool uses XGBoost.

- **The detectors rank windows slightly better than persistence**, at AP 0.80 and 0.75
  against 0.72. That is not significant: the three intervals overlap almost entirely.
- **Persistence is the better alarm.** At the thresholds chosen on validation, its F1 is
  83% against 69–72%, and its recall 78% against 57–61%, at the same precision. AP
  scores the ranking over every threshold, and F1 one operating point. On these dates
  the detectors' validation thresholds were too cautious.
- **Persistence is stronger than any deployable baseline** (`12` §4). It reads the
  current window's own label, rule (a)'s hindsight included. So each incident the
  overrides added strengthened it, and its AP rose from 0.59 in the first run (§9).
- **The operator's alert adds nothing measurable on this split.** Without it, test AP
  is 0.76 and 0.80, the same as with it.

**Lead time, on the eight test incidents:**

| | Flagged | Before the alert | Median lead |
| --- | --- | --- | --- |
| Persistence | 8 of 8 | 3 | −4 min |
| XGBoost and random forest | 6 of 8 | 3 | 0 min |
| XGBoost, no alert feature | 6 of 8 | 4 | +3 min |
| Random forest, no alert feature | 6 of 8 | 3 | 0 min |

A lead time is good to about 30 minutes, one alert poll (`12` §5). Every system flags
most incidents about when the operator posts them. XGBoost without the alert feature
flags slightly earlier on what it catches, as the no-alert detectors did on validation
in development.

## 4. Delay prediction (RQ1 Objective 2)

`transit-train --table data/training_table_20261018.csv`. The training table reads no
alerts, so this stage is the same in both runs. 5 schedule-blind dates and 14 rows over
two hours late were dropped first. The model is XGBoost, with depth 8, chosen on
validation MAE.

| Test, 6 dates | Rows | MAE | RMSE | MAPE* |
| --- | --- | --- | --- | --- |
| Naive persistence | 72,728 | 22.8 s | **66.5 s** | 81% |
| XGBoost | 72,728 | **22.0 s** | 78.8 s | **56%** |

\*MAPE covers only the 51% of rows with a non-zero delay.

**The model beats persistence by 3.7% on MAE, MASE 0.963, and loses on RMSE.**
- **The gain cannot be told from zero.** On these dates, the paired date-resampled
  interval added afterwards runs from −3.0 to +2.7 s (`18` §6).
- **The loss explains part of the RMSE result.** The model is trained on absolute error,
  which rewards many small gains over avoiding a few large misses. That is not the whole
  story: LightGBM, trained on the same loss, beats persistence on RMSE too (`18` §6).
- **The margin narrowed from development.** On the 17 schedule-covered dates to
  2026-09-24, MASE was 0.828. On the scored test dates the margin is much narrower.
- **Where the error sits.** Peak MAE is 27.7 s and off-peak 18.5 s, and the previous
  stop's delay carries 82% of the gain.

**The 90% interval holds close to 90% on dates it never saw:**

| How late the train already is | Test rows | Conformal | ± validation MAE |
| --- | --- | --- | --- |
| First stop | 5,183 | 86.5% | 54.9% |
| On time (≤ 60 s) | 50,956 | 88.8% | 87.5% |
| Late 1–5 min | 16,899 | 89.7% | 55.9% |
| Late > 5 min | 4,873 | 89.9% | 30.0% |
| All | 77,911 | **88.9%** | 74.9% |

It falls slightly short overall, and three of its eight bins fall below 87%: T1's first
stops (85.2%), T1 trains on time (86.8%), and T4 trains already more than 5 minutes late
(84.7%). Conformal coverage assumes the test rows are exchangeable with the calibration
rows, and the test dates are the six after them (`08` §6). A plain ± MAE margin would
have covered 30% of trains already more than 5 minutes late.

## 5. Reasons (RQ1 Objective 3, RQ2)

### k, frozen by the rule

`transit-reasons --until 2026-10-14 --sweep-k 1,2,3,4,5,6,8`, on the 26 incidents before
the test dates:

| k | Recall@k | Chance | Lift | On cause | Shown |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.38 | 0.43 | −0.04 | 0.40 | 1.0 |
| 2 | 0.62 | 0.65 | −0.04 | 0.44 | 1.9 |
| **3** | **0.88** | 0.78 | **+0.10** | 0.48 | 2.8 |
| 4 | 0.88 | 0.84 | +0.05 | 0.50 | 3.6 |
| 5 | 0.88 | 0.86 | +0.02 | 0.47 | 4.4 |
| 6 | 0.88 | 0.88 | +0.01 | 0.46 | 5.2 |
| 8 | 0.88 | 0.88 | +0.00 | 0.44 | 6.6 |

**k = 3**, the smallest k at the highest recall (`13` §7), as in the first run and the
rehearsal. Its lift over chance is between two and three incidents in 26. At k = 1 and
2 the ranking is worse than a random draw, and from k = 5 it is no better.

### The scored run

`transit-reasons --since 2026-10-14 --k 3 --model --repeats 3`. The test split holds
**8 incidents, all technical, on 4 dates**. Five are the urgent repairs added by the
check (Lidcombe, Burwood, Como–Jannali, Jannali, Flemington). The other three are urgent
train repairs at Oatley and at Central, and "an issue with a freight train" at Mt
Kuring-gai that stopped T1 trains between Hornsby and Berowra on 18 October.

| System | Accuracy | Macro-F1, over technical alone | Answered other/unknown |
| --- | --- | --- | --- |
| Most common cause (time-aware) | **1.00** | 1.00 [1.00, 1.00] | 0 of 8 |
| `claude-sonnet-5`, no retrieval | 0.12 ± 0.12 | 0.21 [0.00, 0.43] | **21 of 24** |
| `claude-sonnet-5` + retrieval, k = 3 | 0.08 ± 0.07 | 0.15 [0.00, 0.28] | 10 of 24 |

The model rows are the mean ± sd of three runs. The intervals come from resampling the
4 dates. Recall@3 is 0.88. There were 48 answers, none failed, and **none cited an alert
it was not shown**.

**The shortfall, as `08` §3.1 requires it to be stated:**
- **What the test split held:** 8 incidents of one cause group, on 4 dates.
- **What could not be computed:** cause macro-F1 across groups. Over one group it is
  that group's F1, so it cannot show whether a system tells causes apart.
- **Why the baseline wins:** technical was the commonest cause before the test dates, 15
  of 26 against 10 network incidents, and every test incident was technical.
- **What it does to the conclusions:** the scored run cannot rank the three systems on
  cause. It does show that neither model system named these technical faults from the
  delay feed: 3 right answers in 24 without retrieval, and 2 in 24 with it. With
  retrieval, the model's errors moved from abstaining to a confident wrong group.

**What the answers show** (read from `data/reasons_test_20261018_v2.jsonl`):
- **Without retrieval the model abstains**, 21 times in 24. It says the delay pattern
  alone points to no cause, as in development (`13` §5).
- **With retrieval it reads severity and place, and here both misled it.** The four
  incidents with heavy delays had 6 to 12 services more than five minutes late, and a
  worst delay of 20 to 30 minutes. For each, two of the three past incidents shown were
  police or emergency incidents, at Edgecliff, Bondi Junction or Town Hall. The model
  answered network incident in all 12 of those answers. The third incident shown was a
  technical one each time, which is why recall@3 counts a hit.
- **Where delays were light, it abstained,** in 10 of the 12 answers on the four
  incidents with at most three services late. That includes Burwood, where two of the
  three incidents shown were technical.
- **Three police and emergency incidents fill most of what is shown.** The Edgecliff
  incident of 21 September was shown for 5 of the 8 test incidents. The Bondi Junction
  incident of 6 October and the Town Hall emergency of 13 October were shown for 4 each.
  None of the nine urgent repairs added by the check was ever shown.
- **This is the development finding seen from the other side.** Place predicts a cause
  where faults recur in one place. Where a place's history is mostly network incidents,
  it predicts the wrong group, even with a fault in the evidence.

## 6. The orchestrator

`transit-questions ask --k 3`, on the 28 questions written from the frozen file before
any answer (`14` §7):
- each test date and line at 08:30 and 17:30;
- each test-date incident 15 minutes after its first sighting.

Every transcript records k = 3. All 28 ended normally, with 73 tool calls and no tool
error: line status 28, disruption risk 22, delay predictions 18 and past incidents 5.
The model is `claude-sonnet-5`, and the run used 168k input and 12k output tokens.

- **Every prediction came with its interval.** All 18 answers that used the delay
  tool state the 90% interval, unnarrowed, as the hard requirement asks (`14` §4).
- **At every incident, it found the operator's alert in the feed.** It named that
  alert's cause in all 8 incident answers, once under the wrong id. The line-status
  tool shows the alerts in the feed at that moment, by design. The reasons stage never
  sees the incident's own alert, so the two stages answer different questions.
- **It called every risk figure an estimate.** All 22 answers that used the risk tool
  give the detector's probability beside its threshold and call it an estimate or a
  forecast, as the prompt requires.

## 7. Faithfulness

`transit-judge judge --model claude-opus-4-6`, then `report`. 160 statements from 76
answers were judged, every one by Claude Opus 4.6 under rubric version 4,
`da0a503ea811`, at temperature 0. None failed.

| System | Statements | Faithful, by the judge | Cited | Validation set: judge | Validation set: author |
| --- | --- | --- | --- | --- | --- |
| Agent | 108 | **94%** | 12% | 88% | 88% |
| Model + retrieval | 26 | **15%** | 46% | 25% | 14% |
| Model, no retrieval | 26 | **46%** | 0% | 42% | 35% |

**These rates are upper bounds.** The judge passed its held-out round at κ 0.61 [0.38,
0.82], and every disagreement with the author ran its way (`15` §4). The validation
columns are from `15` §6.

- **Retrieval lowers faithfulness, as it did on the validation set**: 15% against 46%.
  Most of the 22 unsupported retrieval statements claim that the delays "mirror", or do
  not match, a past incident beyond what either alert describes. That is the
  comparison rule of version 4 (`15` §3).
- **The agent is mostly faithful,** 101 statements of 108. Two of its 7 unsupported
  statements cite a real alert id from the feed for another alert's content
  (`7e669567` and `8a035871`). Citation coverage counts both as cited, because it checks
  that an id was shown, not what the id says. The judge caught both. Another says no
  alert in the feed cites a live cause when the Lidcombe repair was there. The other
  four are:
  - an overreach about where delays were worst;
  - a causal link the judge read strictly;
  - an unknown-cause station notice called maintenance;
  - one statement judged without its neighbours (`15` §8).
- **Faithful is not right.** The reasons stage's more faithful system named 3 of 24
  causes.

## 8. What this says about the questions

**RQ1, feasibility.** Each of the three stages runs on real-time GTFS data for T1 and
T4, and each was scored against a baseline written first.
- **Objective 1, data retrieval: feasible.** Collection ran for 46 service dates, and
  32,348 of 32,349 trip polls succeeded. What it could not do is recover the timetable
  era of 11–15 September, which was superseded before the archive began. Those five
  dates have no schedule.
- **Objective 2, prediction with its margin: feasible, with small gains.**
  - The delay model beats persistence by 3.7% on MAE. A paired interval added afterwards
    cannot tell that gain from zero, though it can for LightGBM's 19% (`18` §7). Its 90%
    interval covers 88.9% on dates it never saw, where ± MAE covers 30% of trains
    already late.
  - The detectors rank disrupted windows slightly better than persistence, at AP
    0.75–0.80 against 0.72. Five test dates cannot separate them, and persistence is
    the better alarm at the chosen thresholds.
- **Objective 3, reasons grounded in retrieved evidence: not shown to work.**
  - From the delay feed and past alerts, the model named 3 of 24 technical faults
    without retrieval and 2 of 24 with it.
  - Retrieval finds past incidents by place, and on these dates place pointed to police
    incidents. It also lowered faithfulness, from 46% to 15% by an upper-bound judge.
  - The agent, which reads the operator's live alert, stated causes faithfully: 94% of
    its statements were supported.

**RQ2, evaluation.** The plan ran end to end on one frozen file. Every metric it names
was computed where the test split allowed. Two could not carry weight:
- cause macro-F1 across groups, because the test split held one cause group;
- any significance claim on detection, because over five dates the intervals overlap.

The held-out check of the alert rule, completed at the cut-off, found 19 wrong calls in
69 alerts, all excluding a real incident. That is the evaluation correcting its own
ground truth, by a criterion fixed before the test dates. §9 shows what it changed.

**What these results do not support:**
- that retrieval improves cause identification or faithfulness;
- that any detector beats persistence significantly;
- that the delay model's gain over persistence is distinguishable from zero (`18` §6);
- anything about weather or other/unknown causes on the test dates, which held none.

## 9. The first run, before the correction

The first run used `data/alert_overrides.csv` as it stood before the check was
completed, with two overrides: Harris Park and Croydon. Everything else was the same:
the frozen file, the code, the prompts, the rules, and the delay model.

| | First run, the rule as at 2026-10-11 | Corrected run, the result |
| --- | --- | --- |
| Incidents, all and on the test dates | 25 and 3 | 34 and 8 |
| Disrupted windows, T1 and T4 | 6.8% and 6.5% | 7.5% and 7.3% |
| Detection test positives | 86 | 114 |
| AP: persistence, XGBoost, random forest | 0.590, 0.742, 0.809 | 0.722, 0.745, 0.799 |
| F1: persistence, XGBoost, random forest | 74%, 70%, 72% | 83%, 69%, 72% |
| k, and the incidents it was frozen on | 3, on 22 | 3, on 26 |
| Reasons accuracy: baseline, no retrieval, retrieval | 1.00, 0.11, 0.00 | 1.00, 0.12, 0.08 |
| Answered other/unknown: no retrieval, retrieval | 8 of 9, 3 of 9 | 21 of 24, 10 of 24 |
| Agent questions, and its tool calls | 23 and 58 | 28 and 73 |
| Faithful, by the judge: agent, retrieval, no retrieval | 96%, 22%, 60% | 94%, 15%, 46% |

**The correction mostly moved persistence.** The incidents it added are rule (a)
windows, and persistence reads rule (a) from the current window's label. The
detectors' AP barely moved.

**Every conclusion in §8 holds in both runs.** In both, the detectors rank at least as
well as persistence and are not the better alarm. The reasons stage cannot be ranked
on cause, and retrieval moved its errors towards network incidents. The agent is
mostly faithful, and answers with retrieval are less faithful than answers without it.

## 10. Reproducing it

The chain, from the snapshot pulled after 03:00 Sydney time on 2026-10-19. Every scored
command reads the frozen file, and every command that reads alerts reads
`data/alert_overrides.csv` at this commit. `--snapshots` must name an empty folder, or
reconcile mixes in CSV snapshots that have no cut-off.

```bash
transit-freeze --db data/delay_observations_20261020.db --cutoff 2026-10-18 --out data/evaluation_20261018.db
transit-label --db data/evaluation_20261018.db --cancellation-check --out data/disruption_labels_20261018_v2.csv
transit-reconcile --db data/evaluation_20261018.db --snapshots <an empty folder> --out data/training_table_20261018.csv
transit-train --table data/training_table_20261018.csv \
    --model-out models/delay_model_20261018.joblib --metrics-out models/delay_model_20261018_metrics.json
transit-detect --db data/evaluation_20261018.db --out data/detection_table_20261018_v2.csv \
    --save models/detector_20261018_v2.joblib
transit-index build --source alerts --db data/evaluation_20261018.db
transit-questions write --db data/evaluation_20261018.db --out data/questions_test_v2.csv
transit-reasons --db data/evaluation_20261018.db --until 2026-10-14 --sweep-k 1,2,3,4,5,6,8
transit-reasons --db data/evaluation_20261018.db --since 2026-10-14 --k 3 --model --repeats 3 \
    --out data/reasons_test_20261018_v2.jsonl
transit-questions ask --questions data/questions_test_v2.csv --out-dir data/transcripts_test_v2 \
    --db data/evaluation_20261018.db --detector models/detector_20261018_v2.joblib \
    --delay-model models/delay_model_20261018.joblib --k 3
transit-judge judge --reasons data/reasons_test_20261018_v2.jsonl --transcripts data/transcripts_test_v2 \
    --model claude-opus-4-6 --out data/judged_test_20261018_v2.jsonl
transit-judge report --judged data/judged_test_20261018_v2.jsonl
```

The first run is the same chain at the commit before this one, without the `_v2`
suffix. Each step's full output, for both runs, is kept locally in
`data/logs_20261018/`. Re-running the label gave a byte-identical file. Station names
came from the newest timetable era, `gtfs_schedule_20261019_4ff2ee304795.zip`, which is
the default. A timetable describes the plan, not what ran, so it carries nothing from
after the cut-off into a result. The model answers vary between runs: they are sampled
at the default temperature, which is why each reasons system answers three times.
