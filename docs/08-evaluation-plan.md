# The Evaluation Plan

> Closes outstanding supervisor items 4, 5 and 6 (`04-implementation-plan.md` §4).
> Written in the order Dr Ramezani asked for on 2026-09-09: **step 1, the research
> questions; step 2, how each one gets answered.** Every criterion below names a
> metric and a method, because "be more specific about the criteria, what you want
> to measure, and how" was the note.

The evaluation harness is the object of study, not a final step. If the build
overruns, scope comes out of the *system*, not out of this document.

## 1. The overarching question

> How can automated faithfulness and hallucination evaluation be adapted to an
> agentic RAG system that answers transport queries using static policy retrieval,
> real-time GTFS-Realtime conditions, and a bounded-accuracy delay-prediction model
> — and how does this combined system perform against a static-retrieval-only
> baseline?

It has two arms, and the decomposition follows them:

- **Methodological** — *how can evaluation be adapted*. Existing faithfulness
  measures assume every claim is grounded in a retrieved passage. One third of this
  system's claims are not: they are probabilistic forecasts. SQ4 is the adaptation.
- **Empirical** — *how does it perform against a baseline*. SQ5.

This supersedes the old RQ1/RQ2 split. Those were two candidate framings of the
whole project; what follows is a decomposition of the single merged question.

## 2. The five sub-questions

| | Question | Criterion | Metrics | Status |
| --- | --- | --- | --- | --- |
| **SQ1** | How accurately can a delay model trained only on self-collected realtime data predict next-stop delay, relative to a non-trivial baseline? | Point-forecast error beats naive persistence | MAE, RMSE, MAPE, MASE | **Answered** |
| **SQ2** | Does the policy retriever return the evidence a policy question needs? | Gold passage in top-k | Hit@k, MRR, context precision/recall | To build |
| **SQ3** | Are claims grounded in the evidence actually retrieved or observed, and attributed to the right source? | Every verifiable claim traceable to its true source | Faithfulness, hallucination rate, attribution accuracy, critical-error rate | To build |
| **SQ4** | When an answer rests on the model, is an uncertainty bound stated, undistorted, and empirically valid? | Stated · faithful to the tool · calibrated | Statement rate, margin fidelity, empirical coverage, calibration gap | **The novel one** |
| **SQ5** | How does the combined system compare against static-retrieval-only? | End-to-end quality across all three question kinds | All of the above, plus coverage and refusal appropriateness | To build |

## 3. SQ1 — prediction accuracy *(answered)*

Reported from `models/delay_model_metrics.json`, test split 2026-09-09, scored once.
Reproduced independently on 2026-09-14: MAE 17.915 s, matching to three decimals.

| | MAE | RMSE | MAPE | |
| --- | --- | --- | --- | --- |
| XGBoost | **17.92 s** | 52.45 s | 68.8% | |
| Naive persistence | 19.09 s | 53.73 s | 90.0% | |
| | | | | **MASE 0.938** |

**The honest reading.** The model beats persistence by 6%. That is a weak margin and
it will be reported as one. It is not a failure of method: on a network where the
median delay is zero, persistence is a genuinely strong baseline, and `baseline.py`
was written before the model precisely so this comparison could not be retrofitted.

The finding stands on its own — *a gradient-boosted model with nine features buys
little over "assume it stays as late as it was"* — and it is the answer to supervisor
item 5 about whether live-only collection was viable. It was: the pipeline is proven,
166,076 stop events over 12 service dates, and the accuracy is now measured rather
than assumed.

More importantly, SQ1 is **not the contribution**. It exists to give SQ4 a real
bounded-accuracy component to evaluate. A model that barely beats its baseline is
still a perfectly good object for a faithfulness study — arguably a better one, since
its uncertainty is large enough to matter.

### The reliability filter, stated for the write-up

Rows are kept only where `stops_ahead_final <= 1` — the last observation was within
one stop of the event. Rows without a previous-stop delay (~6%) are excluded from
*both* model and baseline, never one side. Split is chronological by whole service
date, 70/15/15, never random.

## 4. SQ2 — retrieval quality

**Criterion.** The passage needed to answer the question is in the retrieved set, and
the set is not padded with irrelevance.

| Metric | Definition | Target |
| --- | --- | --- |
| Hit Rate@k | Gold passage appears in top-k | report at k ∈ {3, 5, 10} |
| MRR | Reciprocal rank of the first gold passage | — |
| Context Precision | Ragas — proportion of retrieved chunks that are relevant | — |
| Context Recall | Ragas — proportion of gold evidence retrieved | — |

**Method.** 40 policy QA pairs, each carrying gold evidence as *document title + page
number*. Citation metadata is an ingestion contract, not an afterthought: a chunk that
cannot be attributed is unusable as evidence for SQ3, so ingestion emits it or the
chunk is rejected. Chunk size and k are the two tuned parameters; both are swept once,
on a development subset, and frozen before the test QA set is scored.

## 5. SQ3 — faithfulness and source attribution

Two failures are being separated here, and conflating them is the mistake this
project exists to avoid.

**5a. Grounding.** Is each verifiable claim supported by evidence the system actually
had?

- *Metrics*: Ragas Faithfulness; hallucination rate = unsupported claims / verifiable
  claims.
- *Method*: decompose the answer into atomic claims with an LLM, score each against
  the retrieved context.

**5b. Attribution.** Does the answer say *which of the three sources* a claim came
from — and is it right?

This is the agentic part, and it has a free ground truth most RAG evaluations do not
get: **the agent's own tool-call trace**. The harness knows whether the retriever,
the realtime tool, or the prediction tool produced the basis for an answer, so
attribution can be scored against fact rather than against a judge's impression.

- *Metrics*: source-attribution accuracy; a 3×3 confusion matrix over
  {policy, live, predicted}.
- **Critical-error rate** — the cell that matters: a *prediction* presented as an
  observed or retrieved fact. This is the dangerous failure, it is scored separately,
  and the target is zero.

## 6. SQ4 — prediction faithfulness *(the contribution)*

Nothing off the shelf measures this. Ragas asks whether a claim appears in a retrieved
passage; a forecast appears in no passage, so a correct, well-hedged prediction scores
as a hallucination and a reckless one scores the same. The adaptation is the point of
the project.

### 6.1 The three-part criterion

An answer grounded in the delay model is *prediction-faithful* when it is:

| | Measurement | Fails when |
| --- | --- | --- |
| **Stated** | **Margin Statement Rate** — % of prediction-grounded answers stating any bound | The agent says "your train will be 4 minutes late" full stop |
| **Undistorted** | **Margin Fidelity** — does the stated bound match the one the tool returned? | The tool returns ±90 s, the answer says "about a minute either way" |
| **Calibrated** | **Empirical Coverage** — does the stated interval contain the true delay at its nominal rate, on held-out data? | The bound is stated and faithfully relayed, and still wrong |

Margin Fidelity is, as far as this project's literature review found, unmeasured
anywhere: it is *hallucination of uncertainty* — the model correctly computing a bound
and the language layer quietly narrowing, rounding away, or dropping it. It is
separable from grounding, and it needs the tool trace to detect.

### 6.2 Why the margin cannot be the global MAE

The original formulation was *"is the stated margin consistent with the model's
measured MAE"*. Checked against the test split on 2026-09-14, that definition does not
survive contact with the data. If the agent always states ±17.92 s (the global MAE):

| Segment | n | Conditional MAE | Coverage by a global ±17.92 s |
| --- | --- | --- | --- |
| All rows | 14,669 | 17.9 s | 74.6% |
| \|delay\| 0–60 s | 11,073 | 7.8 s | **86.4%** |
| \|delay\| 60–180 s | 2,416 | 32.6 s | **41.5%** |
| \|delay\| > 180 s | 1,180 | 82.4 s | **31.3%** |
| 06:00 hour | 759 | 9.0 s | 84.6% |
| 22:00 hour | 516 | 59.9 s | 59.1% |

Conditional error spans **7.8 s to 82.4 s — a factor of 10.6**. A single global margin
is over-conservative on trains that are running fine and badly overconfident on trains
that are not.

The failure is worse than uneven: it is **anti-correlated with the question**. Nobody
asks whether their on-time train is on time. The prediction tool is invoked when a
rider suspects a delay — exactly the rows where a global margin covers 31% instead of
the ~75% it advertises. A metric that scored such an answer "faithful" would be
certifying the system's most misleading behaviour.

**This is itself a finding**, and it generalises past this project: *bounded-accuracy
claims in agentic systems must be bounded conditionally, because a global error rate
is least accurate where it is most load-bearing.*

### 6.3 The proposed fix — conformal prediction intervals

Rather than compare against MAE, the prediction tool returns a genuine interval, and
faithfulness is measured against that.

**Split conformal**, fitted on the validation split (2026-09-08), never on test:
compute absolute residuals, take the ⌈(n+1)(1−α)⌉/n quantile, and emit
`prediction ± q̂` for a nominal 90% interval. It is distribution-free, assumes only
exchangeability, and is roughly twenty lines against the existing artefact.

Because §6.2 shows conditional coverage is the whole problem, the plan uses
**Mondrian (class-conditional) conformal**: residual quantiles computed *within* bins
of hour-band × peak × route, so a 22:00 prediction carries a wider bound than an 06:00
one by construction. Coverage is then reported both marginally and per bin — the
per-bin table is the evidence that the fix worked.

Cost: one extra day of collection held out as a calibration split. Already covered —
12 service dates exist and the model used 7.

> **For discussion.** This changes the SQ4 metric from *"margin ≈ MAE"* to *"stated
> interval achieves nominal coverage"*. It is a strictly stronger claim and it is
> better supported by the data, but it is a change to what was proposed, so it needs
> sign-off rather than assumption.

### 6.4 Method

Replay held-out stop events through the full agent, phrased as natural rider questions
("I'm at Strathfield, is the next T1 to Central going to be late?"). For each:

1. Record the interval the prediction tool returned (from the trace).
2. Record the interval the answer stated (LLM extraction, human-validated subsample).
3. Record the true delay from the training table.

That yields all three measurements from one pass, and every one of them is checked
against a recorded fact rather than a judgement. `models/delay_model_metrics.json` is
already the contract the harness reads its thresholds from, so retraining updates the
metric without editing this plan.

## 7. SQ5 — the system against its baseline

Supervisor item: *what does "baseline" mean for the whole system?* Three arms, over
one QA set.

| | System | Purpose |
| --- | --- | --- |
| **B0** | Claude, no tools, no retrieval | Floor. Shows whether retrieval earns its place, and exposes parametric-memory answers about Opal fares |
| **B1** | **Static-retrieval-only RAG** | **The comparator named in the RQ.** Policy corpus only; no realtime, no prediction |
| **B2** | Full agentic system | The treatment |

The comparison is deliberately not a fair fight on live and predictive questions — B1
*cannot* answer them. That asymmetry is the result, and it is why **coverage** and
**refusal appropriateness** are scored alongside quality:

- **Coverage** — proportion of questions answered at all.
- **Refusal appropriateness** — when a system lacks the evidence, does it decline, or
  does it invent? A B1 that says "I don't have live service data" scores well; one
  that guesses a delay scores a critical error under §5b.

The interesting hypothesis is not that B2 beats B1 on live questions. It is that
**B1's hallucination rate rises on questions it cannot serve** — the measurable cost
of omitting the two volatile sources, which is the argument the whole system makes.

## 8. Datasets

| Set | n | Gold evidence | Source |
| --- | --- | --- | --- |
| Policy | 40 | Document + page | Opal Fares Business Rules, Opal Terms of Use, Fares & Ticketing brochure |
| Live-condition | 20 | Frozen GTFS-Realtime snapshot | Captured feed fixtures, replayed deterministically |
| Predictive | 20 | True delay from the held-out table | Test-split stop events |
| Multi-source | 15 | Two or more of the above | Hand-written, requires composition |
| Unanswerable / adversarial | 10 | Correct action is refusal | Out-of-scope lines, future dates, unknowable facts |
| **Total** | **105** | | |

**Determinism.** Live questions are scored against *frozen* feed snapshots, not the
live API. Re-running the harness a week later must produce the same numbers, which is
impossible against a feed that has moved on. Fixtures are committed; the realtime
client already separates fetching from parsing, so this needs no new abstraction.

**Provenance.** QA pairs are hand-written against the source documents by the author,
not LLM-generated, and each records its gold evidence at authoring time. LLM-generated
questions over the same corpus that will be retrieved from leaks the answer into the
question and inflates every retrieval metric.

## 9. Experimental design

- **Paired.** Every system sees the identical QA set. Differences are per-question
  paired, so **Wilcoxon signed-rank** is the test, not an unpaired comparison of means.
- **Repeats.** 3 runs per system per question at fixed temperature; report mean ± sd.
  A single run of a stochastic system is an anecdote.
- **Fixed before scoring.** Chunk size, k, prompts and the conformal α are frozen on a
  development subset before the test QA set is touched — once, as with the model's
  test split.
- **Judge validation.** The author hand-labels a **20% stratified subsample** and
  reports **Cohen's κ** between human and LLM judge. An unvalidated LLM judge is an
  unvalidated instrument, and every headline number here depends on one. If κ < 0.6
  the judge prompt is revised and the subsample re-scored — before the full run, not
  after seeing results.
- **Cost.** Haiku for judging, cached embeddings, ~105 questions × 3 systems × 3 runs.
  Inside the semester's USD 50–90 estimate.

## 10. Threats to validity

| Threat | Mitigation |
| --- | --- |
| 105 QA pairs is small | Report confidence intervals, not point estimates; κ-validate the judge |
| Author wrote both system and evaluation | Gold evidence fixed at authoring time; judge validated against a human subsample; the adversarial set is written to fail |
| 12 service dates is a short window | Stated as a limitation; no seasonal or incident-period claims made |
| `RTTA_*` movements are uncollected | ~40% of overnight trips are out-of-service and correctly excluded, but a service altered beyond timetable expression may also land there. Measurable now that full days exist — **measure it and state the number** |
| No alert-flag feature | Nothing polls Service Alerts, so the planned active-alert feature is absent rather than present-and-false. Either add alert collection or state its absence |
| Conformal assumes exchangeability | Rail delays are autocorrelated within a day; class-conditional binning mitigates but does not remove this. State it |

## 11. What this commits to building

In dependency order. The harness is Weeks 8–10 and is protected.

1. Ingestion → cited chunks *(blocks SQ2)*
2. Chroma retrieval *(blocks SQ2, SQ3)*
3. Realtime tools over the existing client *(blocks SQ3, SQ5)*
4. Agent loop — **the margin requirement is in the system prompt from the first
   version**, never bolted on, or SQ4 measures a retrofit
5. Conformal calibration on the validation split *(blocks SQ4)*
6. The QA set — 105 pairs, hand-written
7. The harness: Ragas + custom judge + the three SQ4 measurements
8. Judge validation, then the scored run

Items 5 and 6 do not depend on 1–4 and can be built in parallel with them. The QA set
is the long pole and is the thing most likely to be rushed; starting it early is the
main schedule protection available.
