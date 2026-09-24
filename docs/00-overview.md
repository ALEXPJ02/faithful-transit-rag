# Overview — Predicting and Explaining T1/T4 Service Disruptions

> An agentic AI workflow that predicts and explains train service disruptions on
> Sydney Trains' T1 and T4 lines from real-time GTFS data, and the evaluation that
> says which models do it better.

## The problem

A disruption prediction is only useful if it arrives **early** and with a **credible
reason**. Those are two different problems, and the literature solves them separately:

| Stage | What is needed | Where it stands in the literature |
| --- | --- | --- |
| **Retrieve** | Live conditions, joined to the timetable | Agentic transport systems call tools well, but use static timetables only |
| **Predict** | A disruption flagged before the operator posts it | ML detectors work, but sit outside agentic workflows and outside GTFS-Realtime data |
| **Explain** | The operational cause, grounded in evidence | Retrieval-supported cause identification exists for cloud incidents, not for train disruptions |

No study builds and evaluates a workflow that does all three on the same live feed,
and evaluation is split between prediction error and text faithfulness.

## Research questions

> **RQ1 — Feasibility.** Investigate the feasibility of running machine
> learning-based prediction models on real-time GTFS data to predict or determine
> service disruptions, delays, and other transport events on Sydney Trains' T1 and
> T4 lines.

> **RQ2 — Evaluation.** Develop a comprehensive evaluation plan — datasets,
> performance metrics, baseline methods, experimental design — to determine which
> model detects the disruptions and their reasons better.

RQ1's objectives are the three stages above; RQ2 is how each is measured. See
[`08-evaluation-plan.md`](./08-evaluation-plan.md).

**Scope was narrowed on 2026-09-22:** trains only, service disruptions only, T1 and
T4 only, and Opal fare policy dropped. Retrieval stays — the corpus becomes past
service alerts, used to explain a disruption's cause.

## Glossary

| Term | Meaning |
| --- | --- |
| **GTFS** | General Transit Feed Specification — the static schedule format (`routes.txt`, `stops.txt`, `trips.txt`…) |
| **GTFS-Realtime (GTFS-R)** | The live companion feed: Trip Updates, Vehicle Positions, Service Alerts. Protobuf, not JSON |
| **Trip Update** | Per-trip predicted arrival/departure delay, for stops the vehicle has **not yet reached** |
| **Observation** | One predicted delay for one stop of one trip at one poll instant — what the collector stores |
| **Reconciliation** | The offline step turning raw observations into one row per *completed* stop event |
| **Service Alert** | An operator-published notice with a `cause`, an `effect` and a scope. The corpus RQ1 Objective 3 retrieves from, and the ground truth for a disruption's reason |
| **Faithfulness** | Whether every claim in an answer is supported by its evidence — a retrieved alert or a tool output |
| **Baseline (detection)** | Persistence — "the next window is disrupted if this one is" |
| **Baseline (reasons)** | Most-common-cause, and the same LLM **without** retrieval |
| **Baseline (model)** | Naive persistence — "this trip's delay at the next stop equals its last observed delay" |
| **LLM-as-judge** | Using a separate model to score answer faithfulness against retrieved evidence |

## Scope

**In scope**

- Sydney Trains **T1** (North Shore & Western) and **T4** (Eastern Suburbs & Illawarra).
- **Service disruptions**, with delay regression as the supporting output.
- Past T1/T4 **service alerts** as the retrieval corpus.
- Detection scored by average precision and lead time; reasons by cause macro-F1;
  explanations by faithfulness and citation coverage.

**Out of scope**

- Network-wide prediction, multi-modal journey planning, and roads/parking data.
- A seasonally robust model. The collection window is weeks, not years — see the
  honest limitation in [`04-implementation-plan.md`](./04-implementation-plan.md).
- Using TfNSW's Trip Planner API as a replacement for hand-rolled GTFS joins
  (see [`03-data-sources.md`](./03-data-sources.md) §4 — note the superseding
  banner at the top of that file; the Trip Planner decision still stands, the
  corpus recommendation does not).

## Document map

| Doc | Contents |
| --- | --- |
| [`00-overview.md`](./00-overview.md) | This file — problem, research question, glossary, scope |
| [`01-architecture.md`](./01-architecture.md) | System diagram, components, repo layout, conventions |
| [`02-tech-stack.md`](./02-tech-stack.md) | Technology choices with rationale and cost |
| [`03-data-sources.md`](./03-data-sources.md) | TfNSW dataset selection and access |
| [`04-implementation-plan.md`](./04-implementation-plan.md) | 13-week plan, current status, risks |
| [`05-setup-checklist.md`](./05-setup-checklist.md) | Getting keys, endpoints, and collection running |
| [`06-always-on-collector.md`](./06-always-on-collector.md) | The GCP e2-micro collector, and why Actions is not enough |
| [`07-training-table.md`](./07-training-table.md) | Reconciliation: observations to model-ready rows |
| [`08-evaluation-plan.md`](./08-evaluation-plan.md) | RQ1's objectives, RQ2's metrics, baselines and experimental design |
| [`09-service-alerts.md`](./09-service-alerts.md) | What the alerts feed sends, and how it is stored |

## Ground rules

- **One semester, one person.** 41029 + 41030 concurrently — roughly half the usual runway.
- **Free tiers and a bounded API budget.** ~USD $50–90 for the semester, almost all of it Claude API.
- **The evaluation harness is the deliverable.** If earlier phases slip, they get cut before it does.
- **Collection cannot be caught up.** TfNSW publishes no historical bus/train GTFS-R archive, so the model can only ever train on data collected while the project runs. A day not collected is a day gone.
