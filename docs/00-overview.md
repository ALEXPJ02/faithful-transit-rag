# Overview — Agentic RAG for Sydney Public Transport

> An agentic RAG system that answers Sydney public transport questions from static
> policy documents, live GTFS-Realtime conditions, and a bounded-accuracy delay
> model — and an evaluation harness that measures whether its answers are faithful.

## The problem

A Sydney rider's questions come in three kinds, and they have very different
volatility:

| Kind | Example | Where the answer lives | How fast it changes |
| --- | --- | --- | --- |
| **Stable policy** | "Can I get a refund if my train is delayed?" | Opal fare/policy PDFs | Months |
| **Current conditions** | "Is the T1 delayed right now?" | GTFS-Realtime feeds | Seconds |
| **Forward-looking** | "Will my 5:40 from Central be late?" | A prediction model | Doesn't exist yet |

No existing system answers all three. Transit-focused LLM systems handle static
GTFS well but ignore real-time conditions and prediction entirely. ML delay models
forecast well but sit outside any conversational agent. And agentic RAG evaluation
methods assume every claim traces back to a retrieved fact — which leaves no
defined meaning for "faithfulness" when part of an answer is a probabilistic
forecast rather than a verified fact.

## Research question

> How can automated faithfulness and hallucination evaluation be adapted to an
> agentic RAG system that answers transport queries using static policy retrieval,
> real-time GTFS-Realtime conditions, and a bounded-accuracy delay-prediction
> model — and how does this combined system perform against a static-retrieval-only
> baseline?

The evaluation methodology is the **object of study**; the system is the apparatus.
That framing is what keeps the project scoped to one semester.

## Glossary

| Term | Meaning |
| --- | --- |
| **GTFS** | General Transit Feed Specification — the static schedule format (`routes.txt`, `stops.txt`, `trips.txt`…) |
| **GTFS-Realtime (GTFS-R)** | The live companion feed: Trip Updates, Vehicle Positions, Service Alerts. Protobuf, not JSON |
| **Trip Update** | Per-trip predicted arrival/departure delay, for stops the vehicle has **not yet reached** |
| **Observation** | One predicted delay for one stop of one trip at one poll instant — what the collector stores |
| **Reconciliation** | The offline step turning raw observations into one row per *completed* stop event |
| **Faithfulness** | Whether every claim in an answer is supported by its evidence. For a prediction-grounded claim, this extends to whether the stated error margin matches the model's measured error |
| **Baseline (system)** | Static-retrieval-only RAG — no live tools, no prediction. What the full system is measured against |
| **Baseline (model)** | Naive persistence — "this trip's delay at the next stop equals its last observed delay" |
| **LLM-as-judge** | Using a separate model to score answer faithfulness against retrieved evidence |

## Scope

**In scope**

- Sydney Trains **T1** (North Shore & Western) and **T4** (Eastern Suburbs & Illawarra) for the prediction layer.
- Three Opal policy PDFs as the retrieval corpus.
- Delay **regression** (minutes late at next stop) — not disruption classification.
- A held-out QA set scored for retrieval precision/recall, faithfulness, and hallucination rate.

**Out of scope**

- Network-wide prediction, multi-modal journey planning, and roads/parking data.
- A seasonally robust model. The collection window is weeks, not years — see the
  honest limitation in [`04-implementation-plan.md`](./04-implementation-plan.md).
- Using TfNSW's Trip Planner API as a replacement for hand-rolled GTFS joins
  (see [`03-data-sources.md`](./03-data-sources.md) §4).

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

## Ground rules

- **One semester, one person.** 41029 + 41030 concurrently — roughly half the usual runway.
- **Free tiers and a bounded API budget.** ~USD $50–90 for the semester, almost all of it Claude API.
- **The evaluation harness is the deliverable.** If earlier phases slip, they get cut before it does.
- **Collection cannot be caught up.** TfNSW publishes no historical bus/train GTFS-R archive, so the model can only ever train on data collected while the project runs. A day not collected is a day gone.
