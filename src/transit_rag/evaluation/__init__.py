"""The RQ2 evaluation harness — which model detects disruptions and reasons better.

Four groups of measurement, scored separately because the workflow's stages
make different kinds of claim (``docs/08-evaluation-plan.md`` §3.4):

* **Detection** — average precision (primary), precision/recall/F1, lead time
  against the operator's own alert, and false alarms per day.
* **Reasons** — cause macro-F1 (primary) and retrieval recall@5, against
  most-common-cause and against the same LLM with no retrieval.
* **Explanation** — per-statement faithfulness and citation coverage, judged by
  an LLM whose agreement with a hand-labelled subsample is itself reported as
  Cohen's kappa.
* **Delay** — MAE/RMSE/MASE and interval coverage, supporting rather than
  headline.

Note what "faithful" does *not* mean here. An earlier framing defined it for a
prediction-grounded answer as "the stated margin matches the model's measured
MAE". ``docs/08`` §4 records why that does not survive contact with the data --
conditional error spans an order of magnitude, so a global margin is least
accurate exactly where the tool is asked -- and replaces it with a conformal
interval the answer must relay unnarrowed.
"""
