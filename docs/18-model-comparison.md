# 18. More models, compared after the scored run

> The supervisor asked for more machine-learning models to be compared against each
> other, for the research part of the report. This page adds LightGBM, CatBoost and
> logistic regression to disruption detection, and LightGBM and CatBoost to delay
> prediction, on the frozen file the scored run read. It is **post-hoc**: it was asked
> for, built and run after the results in [`17`](./17-results.md) were known. It sits
> beside them and replaces none of them.

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
- **The scored rows do not move.** `--compare` appends to what the scored commands
  print and changes none of it (§3).

What it can then claim is how these models rank on these five test dates under one
protocol fixed in advance. It cannot promote a model. The agent keeps the scored
XGBoost detector and delay model, because choosing the best of several on test would
be choosing on test.

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

Each boosting grid **mirrors XGBoost's depths and learning rates**, so no library is
searched harder than another. A leaf-wise LightGBM tree is held to `num_leaves =
2^depth − 1`, the most a depth-wise tree of that depth can have.

| Detection, 3 candidates each | Candidates | Fixed |
| --- | --- | --- |
| XGBoost (scored) | depth 3/4/6, learning rate 0.05, `min_child_weight` 5/10/20 | 1,000 trees, early stopping on validation AP (`aucpr`) |
| Random forest (scored) | depth 6/10/unlimited, `min_samples_leaf` 5/10/20 | 500 trees |
| LightGBM | depth 3/4/6, `num_leaves` 7/15/63, learning rate 0.05, `min_child_samples` 5/10/20 | 1,000 trees, early stopping on validation AP, the only metric tracked |
| CatBoost | depth 3/4/6, learning rate 0.05 | 1,000 iterations, early stopping on validation PR-AUC |
| Logistic regression | C 0.01/0.1/1 | Median imputation with an indicator per missing column, then standard scaling |

| Delay, 4 candidates each | Candidates | Fixed |
| --- | --- | --- |
| XGBoost (scored) | depth 4/6/6/8, learning rate 0.10/0.10/0.05/0.05, `min_child_weight` 5/5/20/20 | Absolute-error loss, 2,000 trees, early stopping on validation MAE |
| LightGBM | depth 4/6/6/8, `num_leaves` 15/63/63/255, learning rate as XGBoost's, `min_child_samples` 5/5/20/20 | L1 loss, 2,000 trees, early stopping on validation MAE |
| CatBoost | depth 4/6/6/8, learning rate as XGBoost's | MAE loss, 2,000 iterations, early stopping on validation MAE; the line and stop are named as categories |

Three choices need their reasons:

- **Leaf sizes.** Under an absolute-error loss every row weighs the same, so XGBoost's
  `min_child_weight` counts rows. LightGBM's `min_child_samples` takes the same
  numbers. CatBoost's symmetric trees have no minimum leaf size, so its grid varies
  depth and learning rate alone.
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

**Re-runs reproduce.** LightGBM runs with `deterministic` and `force_row_wise`, and
every model has a fixed seed. A test refits each detector and checks that its scores
are bit-identical, as the forest's are (`12` §5).

**Checked before any added model was scored, on 2026-10-23.** The refactor that makes
room for `--compare` leaves the scored output unchanged:

- `transit-detect` without `--compare` reproduced the scored log of `17` §3 line for
  line. The only difference was the `--save` and `--out` lines it was deliberately not
  given.
- `transit-train --report-only` reproduced the report of `17` §4.
- The frozen file's checksum did not move.

**Both commands write nothing under `--compare`.** No table, no model, no metrics
file. They only print, and the log is the record, as it is for the scored run.

## 4. Running it

Once, after the pull request that fixed the grids is merged. The checks either side
are what make the run admissible:

```bash
shasum -a 256 data/evaluation_20261018.db
transit-detect --db data/evaluation_20261018.db --compare 2>&1 | tee data/logs_20261018/v3_06_detect_compare.log
diff <(head -n 50 data/logs_20261018/v2_06_detect.log) <(head -n 50 data/logs_20261018/v3_06_detect_compare.log)
transit-train --table data/training_table_20261018.csv --compare > data/logs_20261018/v3_05_train_compare.log
diff <(grep -v '^INFO' data/logs_20261018/05_train.log) <(head -n 48 data/logs_20261018/v3_05_train_compare.log)
shasum -a 256 data/evaluation_20261018.db
```

- **The checksum** must read `44e90956…32ac` before and after.
- **Both `diff`s** must print nothing:
  - the first 50 lines of the detection log are the scored detection output, before its
    `--save` and `--out` lines;
  - the first 48 lines of the delay log are the scored delay report. Its progress lines
    go to the terminal, not the log.

The results, and what they change in `17` §8, are added here by the pull request that
records the run.
