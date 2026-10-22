# 18. More models, compared after the scored run

> The supervisor asked for more machine-learning models to be compared against each
> other, for the research part of the report. This page adds LightGBM, CatBoost and
> logistic regression to disruption detection, and LightGBM and CatBoost to delay
> prediction, on the frozen file the scored run read. It is **post-hoc**: it was asked
> for, built and run after the results in [`17`](./17-results.md) were known. It sits
> beside them and replaces none of them.
>
> **In short:**
> - **Delay:** LightGBM beats persistence clearly, at MASE 0.813, with a paired interval
>   clear of zero. It is also the model a choice on validation would have made (§6).
> - **Detection:** no model from any family beats persistence significantly, and a
>   linear model ranks windows as well as the boosted trees (§5).

## 1. What a comparison made now can claim

The scored run's test dates have been seen. A model added now could be tuned until it
looked good on them, and nothing in the numbers would show it. So this comparison
keeps everything the scored run fixed, and fixes the rest before it runs:

- **Data:** the frozen file `data/evaluation_20261018.db` (sha256 `44e90956…32ac`),
  its splits (`17` §2), and the same features (`12` §2, `07`).
- **Method:** each model is chosen on validation by the scored rule. Its grid is the
  same size as XGBoost's. Test is scored once.
- **Fixed before the run:** the grids, the rules and the code were merged in a pull
  request before any added model was scored on test. The run is made once. If a model
  had failed on real data, the code would have been fixed, never the grid, and the
  fix recorded here.
- **Corrected in review, before the run.** Review of that pull request found two places
  where LightGBM's grid did not mirror XGBoost's. Both were fixed before anything was
  scored:
  - its trees were capped one leaf short (2^depth − 1);
  - in detection, its leaf minimum counted rows where XGBoost's counts hessian.

  The grids are now derived in code from XGBoost's (`prediction/grids.py`), so they
  cannot drift from it again.
- **The scored rows do not move.** `--compare` appends to what the scored commands
  print and changes none of it (§3).

What it can then claim is how these models rank on these test dates (five for
detection, six for delay) under one protocol fixed in advance. It cannot promote a
model on its test score. The agent keeps the scored XGBoost detector and delay model:
its answers were scored with them (`17` §6), and a change would need its own disclosed
run (§7).

## 2. The models, and why each

**Detection** (RQ2) now spans the main families of tabular classifier:

| System | Family | Why it is in | |
| --- | --- | --- | --- |
| Persistence | Baseline | The next 30 minutes are disrupted if this window is (`12` §4) | scored |
| Logistic regression | Linear | Shows whether a tree ensemble earns its complexity over a linear model | added |
| Random forest | Bagged trees | Cottreau et al. (2025), as `08` §3.3 named | scored |
| XGBoost | Boosted trees, grown depth-wise | `08` §3.3; the agent's risk tool | scored |
| LightGBM | Boosted trees, grown leaf-wise | The supervisor named it. A different growth strategy from XGBoost's | added |
| CatBoost | Boosted symmetric trees, ordered boosting | The supervisor named it. Ordered boosting is designed against overfitting, which matters at a few thousand windows | added |

Every added model also runs **without the alert feature**, the ablation `08` §3.5
names, as the scored detectors do.

**Delay** (RQ1 Objective 2) adds LightGBM and CatBoost beside XGBoost and persistence.
No linear model is added there: the delay model's 321 stop ids are categories, and a
linear model would need one coefficient for each.

### The grids

Each boosting grid **mirrors XGBoost's on the parameters they share**: the same
depths, learning rates and leaf minimums, and the same number of candidates, so no
library is searched harder than another.
- **Leaf counts.** A leaf-wise LightGBM tree is held to `num_leaves = 2^depth`, every
  leaf a depth-wise tree of that depth can hold.
- **Derived, not copied.** The LightGBM and CatBoost grids are derived in code from
  XGBoost's (`prediction/grids.py`), and a test pins the values that were frozen.
- **Each library keeps its own design** where there is no shared parameter. LightGBM
  splits a category under its own rules, such as a minimum number of rows per category
  group, which XGBoost does not have. CatBoost encodes categories by ordered target
  statistics. Those differences are part of what is compared, so they keep their
  libraries' defaults rather than being tuned to resemble XGBoost after the fact.

| Detection, 3 candidates each | Candidates | Fixed |
| --- | --- | --- |
| XGBoost (scored) | depth 3/4/6, learning rate 0.05, `min_child_weight` 5/10/20 | 1,000 trees, early stopping on validation AP (`aucpr`) |
| Random forest (scored) | depth 6/10/unlimited, `min_samples_leaf` 5/10/20 | 500 trees |
| LightGBM | depth 3/4/6, `num_leaves` 8/16/64, learning rate 0.05, `min_child_weight` 5/10/20 | 1,000 trees, early stopping on validation AP, the only metric tracked |
| CatBoost | depth 3/4/6, learning rate 0.05 | 1,000 iterations, early stopping on validation PR-AUC |
| Logistic regression | C 0.01/0.1/1 | Median imputation with an indicator per missing column, then standard scaling |

| Delay, 4 candidates each | Candidates | Fixed |
| --- | --- | --- |
| XGBoost (scored) | depth 4/6/6/8, learning rate 0.10/0.10/0.05/0.05, `min_child_weight` 5/5/20/20 | Absolute-error loss, 2,000 trees, early stopping on validation MAE |
| LightGBM | depth 4/6/6/8, `num_leaves` 16/64/64/256, learning rate as XGBoost's, `min_child_weight` 5/5/20/20 | L1 loss, 2,000 trees, early stopping on validation MAE |
| CatBoost | depth 4/6/6/8, learning rate as XGBoost's | MAE loss, 2,000 iterations, early stopping on validation MAE; the line and stop are named as categories |

These choices need their reasons:

- **Leaf sizes.** `min_child_weight` is the least hessian a leaf may hold, and means
  the same in XGBoost and LightGBM, so LightGBM takes XGBoost's numbers. A row's
  hessian is at most 1 under either loss, so a leaf that meets the hessian floor meets
  a row floor of the same number too.
  - **In detection it is not a row count.** Under log loss a row adds p(1 − p), at
    most 0.25, and near the 6% base rate about 0.06. LightGBM's own row floor,
    `min_child_samples`, is set to the same number, which the hessian floor already
    exceeds, so it never binds.
  - **In delay it is.** Under absolute error both libraries give every row a hessian of
    1, so both floors match XGBoost's exactly.
  - **CatBoost has no leaf-size control.** Its symmetric trees have no minimum leaf
    size, so its grid varies depth and learning rate alone.
- **A missing line or stop.** XGBoost and LightGBM take a missing category natively.
  CatBoost rejects one, so it is named `<missing>` when CatBoost fits and predicts. The
  frozen table has none, so this changes nothing in the run.
- **Missing values in the linear model.** It cannot split on a missing value as the
  trees do. A window with no service is information, not a gap (`12` §2), so the
  indicator column carries it.
- **No reweighting.** As for the scored detectors, no model is reweighted for the base
  rate (`12` §5).

## 3. The protocol

**Selection is the scored rule, for every model.** The candidate with the best
validation AP (detection) or MAE (delay) wins. Early stopping watches the same metric
on validation. A detector's threshold is the best F1 on validation, placed midway
between two scores (`12` §3). Test is scored once.

**Two comparisons are added, because more models need them:**

- **The gap to persistence is paired.** Over five test dates, each system's own 95%
  interval is wide, and they overlap almost entirely (`17` §3). Comparing the intervals
  would hide a gap that holds on every draw. So each system is also reported as AP
  minus persistence's AP, resampling the same dates for both (`detect_cli.paired_gap`,
  over `evaluation/stats.py`). A hard day lowers both systems in the same draw, so
  the difference is measured more tightly than either AP. The delay gap is MAE minus
  persistence's, resampled the same way.
- **Each delay model gets its own 90% interval.** It is split conformal by line ×
  lateness, calibrated on that model's validation residuals (`08` §4). This shows
  whether the margin method holds for a model other than the one it was built for.

**Validation MAE is measured one way in the delay table**: from each model's own
predictions, XGBoost's included. XGBoost's selection itself read its float32 eval
metric, as in the scored run, which this comparison does not change.

**Re-runs reproduce on the same machine.** LightGBM runs with `deterministic` and
`force_row_wise`, and every model has a fixed seed. A test refits each detector and
checks that its scores are bit-identical, as the forest's are (`12` §5).

The boosters use every core, so the order in which threads sum a histogram follows the
thread count. On a machine with a different count, or a different processor, scores
can differ in their last bits. The scored XGBoost has the same property. The run is
made on an Apple M1 Pro with 8 cores, which reproduces the scored output byte for byte
(below).

**Checked before any added model was scored, on 2026-10-23.** The refactor that makes
room for `--compare` leaves the scored output unchanged:

- `transit-detect` without `--compare` reproduced the scored log of `17` §3 line for
  line. The only difference was the `--save` and `--out` lines it was deliberately not
  given.
- `transit-train --report-only` reproduced the report of `17` §4.
- The frozen file's checksum did not move.

**Both commands write nothing under `--compare`.** No table, no model, no metrics
file. `transit-detect --compare` refuses `--save` and `--out`, and
`transit-train --compare` refuses `--model-out` and `--metrics-out`, so neither can
overwrite what the agent reads. They only print, and the log is the record, as it is
for the scored run.

## 4. Running it

Once, after the pull request that fixed the grids is merged. The checks either side
are what make the run admissible:

```bash
shasum -a 256 data/evaluation_20261018.db
transit-detect --db data/evaluation_20261018.db --compare > data/logs_20261018/v3_06_detect_compare.log 2> data/logs_20261018/v3_06_detect_compare.err
diff <(sed '/^saved the XGBoost detector/,$d' data/logs_20261018/v2_06_detect.log | sed '$d') <(sed '/^Added for comparison/,$d' data/logs_20261018/v3_06_detect_compare.log | sed '$d')
transit-train --table data/training_table_20261018.csv --compare > data/logs_20261018/v3_05_train_compare.log 2> data/logs_20261018/v3_05_train_compare.err
diff <(grep -v '^INFO' data/logs_20261018/05_train.log) <(sed '/^Compared on the same/,$d' data/logs_20261018/v3_05_train_compare.log | sed '$d')
shasum -a 256 data/evaluation_20261018.db
```

- **The checksum** must read `44e90956…32ac` before and after.
- **Both `diff`s** must print nothing. Each compares everything the scored run printed
  with everything the new run printed before its first added line.
  - **Cut where the content changes, not at a line count.** A fixed `head -n` would
    pass a run that slipped an extra scored line in past its cut.
  - **Each `sed '$d'` drops a blank line.** It is the one printed before the cut.
- **Stderr is kept in its own file**, the `.err` beside each log. Written to a pipe,
  stdout is buffered, so a warning from an added fit could land above the scored rows
  and fail the `diff` on output that never changed. The warnings are kept, and read.

**The run** was made once, at 03:08 Sydney time on 23 October 2026, from `ee5fbd1`, the
merge of the pull request that froze the grids. Both `diff`s printed nothing, and the
checksum read `44e90956…32ac` before and after. Detection wrote nothing to stderr, and
delay only its progress lines. Detection took a minute, delay 26 minutes. The four
files are in `data/logs_20261018/`, as `v3_06_detect_compare` and
`v3_05_train_compare`, each a `.log` and an `.err`. One deviation: the local venv
predates the `transit-detect` script, so detection ran as `python -m
transit_rag.prediction.disruption.detect_cli`, the same `main()`.

## 5. Detection (RQ2)

The test split is `17` §3's: 904 windows whose next 30 minutes are known, **114 of them
positive, over 5 dates**. They are not spread evenly. **68 of the 114 fall on 15
October**, and 17 October has none.

| Test, 114 positives | Validation AP | Test AP | 95% interval | AP − persistence, paired | F1 | False alarms / day |
| --- | --- | --- | --- | --- | --- | --- |
| Persistence | 0.688 | 0.722 | [0.286, 0.829] | | **83%** | 2.2 |
| XGBoost (scored) | 0.878 | 0.745 | [0.285, 0.876] | +0.022 [−0.115, +0.151] | 69% | **1.8** |
| Random forest (scored) | **0.887** | **0.799** | [0.305, 0.897] | +0.076 [−0.054, +0.160] | 72% | 2.0 |
| LightGBM | 0.882 | 0.757 | [0.292, 0.878] | +0.035 [−0.095, +0.147] | 70% | 2.2 |
| CatBoost | 0.885 | 0.764 | [0.294, 0.883] | +0.042 [−0.079, +0.146] | 69% | 2.6 |
| Logistic regression | 0.864 | 0.780 | [0.315, 0.877] | +0.058 [−0.058, +0.157] | 69% | 3.6 |

| The same, without the alert feature | Validation AP | Test AP | 95% interval | AP − persistence, paired | F1 | False alarms / day |
| --- | --- | --- | --- | --- | --- | --- |
| XGBoost (scored) | 0.866 | 0.756 | [0.257, 0.854] | +0.033 [−0.128, +0.136] | 70% | 5.2 |
| Random forest (scored) | 0.868 | 0.796 | [0.300, 0.889] | +0.074 [−0.075, +0.175] | 70% | 2.2 |
| LightGBM | 0.864 | 0.750 | [0.248, 0.855] | +0.028 [−0.148, +0.141] | 67% | 3.6 |
| CatBoost | 0.869 | 0.793 | [0.307, 0.880] | +0.070 [−0.062, +0.167] | 72% | 3.8 |
| Logistic regression | 0.845 | 0.796 | [0.329, 0.879] | +0.073 [−0.041, +0.166] | 69% | 4.6 |

The scored rows are `17` §3's, reproduced. What is new for them is the paired gap.

- **No detector beats persistence, even paired.** All ten gaps are positive, from +0.02
  to +0.08, and every interval includes zero. Pairing narrows them a lot: the forest's
  own interval is 0.59 wide, and its gap's is 0.21. Five dates are not enough even so,
  when one of them holds 60% of the positives.
- **The random forest is still the best ranker,** at 0.799. It also had the best
  validation AP, so choosing on validation would pick it.
- **The linear model ranks windows as well as the boosted trees.** Logistic
  regression's test AP, 0.780, is above all three boosting libraries', which run from
  0.745 to 0.764 with the alert feature. The gap is within noise either way. On this
  split nothing shows the tree ensembles earning their complexity over a linear model.
- **Validation's ranking did not carry to test, apart from the forest at the top.**
  Logistic regression was last on validation and near the top on test. XGBoost was
  fourth on validation and last on test. Five test dates allow that much.
- **Persistence is still the better alarm.** At the thresholds chosen on validation,
  every detector's F1 is 67–72%, against persistence's 83%, as in `17` §3.
- **The operator's alert still adds nothing measurable.** Without it, CatBoost and
  logistic regression score higher on test, and LightGBM lower, all within noise.
- **Lead time does not separate the detectors from one another.**
  - **Flagging:** every detector, scored or added, flags 6 of the 8 test incidents.
    Persistence flags all 8, at a median of 4 minutes after the alert, because it reads
    each window's own label, rule (a)'s hindsight included (`12` §4).
  - **Median leads** for the detectors are 0 minutes. The exceptions are CatBoost and
    XGBoost without the alert feature, at +3 minutes with 4 of their 6 before the alert.
  - **Resolution:** a lead time is good to about 30 minutes (`12` §5), so no detector's
    differs from another's.
- **The log's "defined in fewer than every resample (100%)".** The log prints the share
  rounded.
  - **The exact count:** replaying the bootstrap's seeded date draws against each test
    date's positives shows that 1 resample of 2,000 drew 17 October, which has no
    positive target, five times.
  - **The share** is therefore 99.95%. The same is true of every AP interval in `17` §3.

## 6. Delay prediction (RQ1 Objective 2)

The compared test rows are `17` §4's, 72,728 of them over 6 dates, and so is the
persistence baseline.

| Test, 72,728 rows | Validation MAE | MAE | RMSE | MAPE* | MASE | MAE − persistence, paired |
| --- | --- | --- | --- | --- | --- | --- |
| Naive persistence | | 22.8 s | 66.5 s | 81% | | |
| XGBoost (scored) | 22.6 s | 22.0 s | 78.8 s | **56%** | 0.963 | −0.9 s [−3.0, +2.7] |
| **LightGBM** | **21.0 s** | **18.5 s** | **61.0 s** | 58% | **0.813** | **−4.3 s [−5.2, −3.3]** |
| CatBoost | 22.2 s | 20.2 s | 67.0 s | 63% | 0.884 | −2.6 s [−3.7, −1.1] |

\*MAPE covers only the 51% of rows with a non-zero delay. All three chose depth 8 and
learning rate 0.05. The paired gaps come from unrounded MAEs, so a gap can differ by
0.1 s from the difference of the rounded columns. XGBoost's −0.9 s is one such case.

- **LightGBM beats persistence clearly.** Its MAE is 19% lower, and the paired interval
  excludes zero by 3.3 seconds. **CatBoost does too**, by 12%. The scored XGBoost's 3.7%
  gain has a paired interval that includes zero, so on its own it is not distinguishable
  from persistence.
- **LightGBM also beats persistence on RMSE,** at 61.0 seconds against 66.5, where
  XGBoost loses (78.8). `17` §4 put XGBoost's RMSE loss down to its absolute-error
  training. LightGBM trains on the same loss and does not lose, so the loss alone does
  not explain it. Why XGBoost's large misses are larger was not investigated.
- **Validation agrees with test.** LightGBM also has the lowest validation MAE, 21.0
  seconds against 22.2 and 22.6. Choosing the library on validation, as the
  hyperparameters were chosen, would have picked it without looking at test. Unlike
  detection's ranking, this one holds on both, by a clear margin. No interval was
  computed for LightGBM against XGBoost directly. The plan set the gap to persistence as
  the comparison.
- **The tree cap binds for LightGBM and CatBoost.** The cap of 2,000 trees is the scored
  XGBoost's, kept for parity, and XGBoost stopped early (best iteration 1,690).
  - **LightGBM's** chosen candidate used every tree it was allowed (best iteration
    1,999).
  - **CatBoost's** best was at 1,985, nearer the cap than its 50-round patience, so the
    cap ended its fit too.

  More trees might improve both, so their figures are if anything conservative.
- **The 90% interval behaves the same for every model.** Overall it covers 88.9–89.2%,
  slightly short of 90%, as `17` §4 found for XGBoost. Calibrated on each model's own
  validation residuals:

  | How late the train already is | XGBoost | LightGBM | CatBoost |
  | --- | --- | --- | --- |
  | First stop | 86.5% | 86.4% | 86.5% |
  | On time (≤ 60 s) | 88.8% | 89.0% | 88.9% |
  | Late 1–5 min | 89.7% | 89.4% | 90.1% |
  | Late > 5 min | 89.9% | 91.8% | 91.7% |
  | All | 88.9% | 89.1% | 89.2% |

  The shortfall at first stops is the same for all three, so it belongs to the method
  and the data, not to one model. `17` §4 traces it to the test dates following the
  calibration dates.

## 7. What this changes in `17` §8

**RQ1 Objective 2, prediction with its margin: feasible, with a real gain from the
right library.**
- **What `17` §8 said:** "small gains". XGBoost beat persistence by 3.7% on MAE, and
  the paired interval here shows that gain is not distinguishable from zero.
- **What changes:** LightGBM, which a choice on validation would have made, beats
  persistence by 19% on MAE and on RMSE too, with a paired interval clear of zero.
- **What holds:** the conformal margin covers about 89% whichever model it wraps,
  slightly short of its 90%, as in `17` §4.

**RQ2, which model detects disruptions better: unchanged, on stronger evidence.**
- Five learners from three families (linear, bagged and boosted) now stand beside
  persistence. None beats it significantly, now on paired intervals as well as overlapping
  ones, and persistence stays the better alarm.
- A linear model ranks windows as well as any boosted tree.

**What these results do not support:**
- that any detector, from any of these families, beats persistence significantly on
  these five dates;
- that one boosting library detects better than another;
- that LightGBM beats XGBoost on delay by a margin with its own interval, which was not
  computed.

**The agent keeps the scored models.** Its answers were scored with them (`17` §6).
Moving its delay tool to LightGBM, the model a fresh choice on validation would make,
is future work, and it would need its own disclosed run.

## 8. Threats to validity

- **The comparison is post-hoc.** The scored results were known when these models were
  added.
  - **Mitigations:** the grids and rules were frozen in a merged pull request before
    any added model touched test, the run was made once, and the scored rows reproduced
    byte for byte.
  - **What they cannot undo:** which families to add was chosen knowing the scored
    results. The supervisor named LightGBM and CatBoost, and logistic regression was
    added as the linear reference. None was chosen for how it scored.
- **Five test dates, one of them dominant.** 15 October holds 68 of detection's 114
  positives, and 17 October none, which is why every detection interval is wide. Delay
  rests on six dates.
- **The comparisons are uncorrected.** There are ten detection gaps and three delay
  gaps, each at 95%.
  - **Detection:** no interval excludes zero, so a correction cannot change that
    conclusion.
  - **Delay:** the positive claims are the ones a correction could weaken. LightGBM's
    interval ends 3.3 s from zero, and CatBoost's only 1.1 s from it. A correction over
    the three delay gaps would widen each to a 98.3% interval, which was not computed.
    So CatBoost's claim is the weaker of the two.
- **Equal candidate counts are not equal tuning.** Each library keeps its own defaults
  where there is no shared parameter (§2), and the tree cap binds for two of the three
  boosters on delay.
- **Re-runs reproduce on the same machine** (§3). On another thread count or processor
  the last bits of a score can differ, as the scored run's can.
- **Importances are not compared.** Each library defines importance its own way, so only
  XGBoost's is printed, as in the scored run.
