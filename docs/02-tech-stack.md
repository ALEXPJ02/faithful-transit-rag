# Tech Stack

> Principle: **boring, cheap, and defensible in a viva**. Every choice has to survive
> the question "why not the obvious alternative?", and none of them may quietly
> outsource the part of the system the research question is about.

## At a glance

| Layer | Choice | Why |
| --- | --- | --- |
| Language | **Python 3.12** (floor 3.11) | The whole RAG/eval/ML ecosystem lives here |
| Generation | **Claude Sonnet** via the Anthropic API | Strong tool-use; the judge is a *different, cheaper* model to avoid self-grading |
| Eval judge | **Claude Haiku** | Judge calls dominate eval volume; a cheap judge is what makes a large QA set affordable |
| Embeddings | **Voyage `voyage-4-lite`** | Anthropic has no first-party embeddings API; 200M free tokens covers a 3-PDF corpus many times over |
| Agent | **Hand-rolled tool-use loop** (Anthropic SDK) | See §2 |
| Vector store | **Chroma**, embedded | The corpus is 3 static PDFs — a server-based store would be ceremony |
| Realtime | `gtfs-realtime-bindings` + TfNSW Open Data Hub | The feeds are protobuf; there is no JSON alternative |
| Prediction | **XGBoost** regressor | See §3 |
| Tool interface | **Anthropic MCP Python SDK** + FastAPI | Makes the tools callable from any MCP client, not just this agent |
| Evaluation | **Ragas** + a custom LLM-as-judge | Ragas covers standard RAG metrics; the prediction-faithfulness metric is novel and has to be hand-written |
| Lint/format | **Ruff** | One tool replacing flake8 + isort + black |
| Types | **mypy**, `disallow_untyped_defs` | Catches feed-parsing mistakes that unit tests miss |
| CI | **GitHub Actions** | Free and unmetered on public repos |

## 1. Two phases

**Phase 1 (Weeks 1–13, graded):** everything above, running locally. Public repo from
day one. A minimal Dockerfile is kept in sync from ~Week 4 but is *not* used for
daily development — containerising during the highest-uncertainty weeks slows
iteration, and the point of keeping it current is only to avoid the
"dockerise everything at the end" dependency-drift trap.

**Phase 1b — prediction layer:** the delay model. Separated because it has a
dependency the rest of the project doesn't: a collection window that has to already
be running. See §3.

**Phase 2 (post-grading, portfolio):** harden the Dockerfile into one production
image (FastAPI + MCP + baked-in Chroma index), deploy to Render's free tier, wire
GitHub Actions to run the eval harness as a regression check on every push.
Explicitly decoupled from the semester — it is a portfolio goal, not a grading
requirement.

**Chroma on ephemeral disk:** Render's free tier has no persistent disk, so the
index is **built into the Docker image at build time** (a `RUN` step) rather than
created at runtime. Legitimate precisely because the corpus is static; it would be
the wrong answer for a corpus that changes.

## 2. Why a hand-rolled agent, not LangChain

The research question is about how the agent decides between retrieval, live tools,
and prediction, and about how faithfully it reports each. A framework would bury
exactly that logic behind an abstraction, and every eval result would carry an
asterisk about the framework's prompt templates. Hand-rolling is also the stronger
portfolio signal: it demonstrates understanding of the tool-use loop rather than
familiarity with a library. Cost is not a factor either way.

## 3. Why XGBoost, and why the collector runs from day one

**The constraint that drives everything:** TfNSW's Open Data Hub has **no usable
historical archive** for bus or train GTFS-Realtime data. The "Historical GTFS and
GTFS Realtime" API has been Metro + Ferry only since 2020, and public requests for
bus/train history (2021–2025, still true as of a May 2026 forum thread) are
consistently answered with "collect the live feed going forward". Aggregate
on-time-running datasets exist back to 2010, but they are monthly percentage
rollups — no per-trip records, so nothing trainable.

Consequences, in order:

1. **Train on self-collected data.** The collector has to be running before anything
   else can be built. Its uptime is the project's critical path.
2. **Regression, not classification.** Disruptions are rare; in a multi-week window
   there would be too few positive cases. Delay regression uses every trip as signal.
3. **XGBoost, not an LSTM.** Sarhani & Voß (2024) found classical ML best for rail
   delay prediction *using open data alone* — this project's exact constraint.
   Boudabbous et al. (2026) get strong LSTM results, but on a city-scale feature
   pipeline with far more data than a semester of collection can produce.
4. **Report error margins, never point estimates.** Published accuracy is bounded
   (~10% error on dense routes, ~20% on sparse ones) and the field has no agreed
   evaluation protocol (Elliker et al., 2026). The system states its uncertainty,
   and the harness scores whether it does.

**Model spec:** delay regression (minutes late at next stop), T1 and T4 only.
Features: scheduled-vs-actual delta, hour, day-of-week, peak flag, delay at the
previous stop of the same trip, active-alert flag. Baseline: naive persistence.
Metrics: MAE, RMSE, MAPE against that baseline. Split: **time-based** 70/15/15 by
collection week — a random shuffle would leak future information into the training
set, which is the standard way time-series results get silently inflated.
Serving: a `joblib` artefact loaded in-process inside the MCP tool handler; a
separate model service would be infrastructure with no research payoff.

## 4. Dependency groups

Base install is the retrieval + agent path. Everything else is an extra, so the
collector can run in a minimal environment:

```bash
pip install -e ".[realtime]"                        # collector only — no model stack
pip install -e ".[dev,realtime,prediction]"         # normal development
pip install -e ".[dev,realtime,prediction,serve,evaluation]"   # everything
```

This is not tidiness for its own sake: the collector is the component most likely to
run somewhere small and unattended, and a dependency it doesn't need is a way for it
to fail to start.

## 5. Cost

~**USD $50–90 for the semester** (~AUD 75–140), almost entirely Anthropic API:
Sonnet generation across development queries plus Haiku judge calls across eval runs.
Embeddings (Voyage free tier), Chroma, Ragas, the TfNSW API, GitHub, and Phase 2
hosting are all effectively $0. Verified against Anthropic's and Voyage's own pricing
pages rather than third-party summaries.

Set a spend limit in the Anthropic Console regardless — an accidental loop over the
eval set is the realistic way this estimate gets blown.

## 6. Settled, with the alternatives on record

| Question | Decision | The case against, honestly |
| --- | --- | --- |
| Framework vs. hand-rolled | Hand-rolled | More code to write and maintain under a compressed timeline |
| Claude API vs. local LLM | Claude API | A local model would be free, but costs GPU/hosting overhead and weakens the eval story |
| Trip Planner API vs. GTFS joins | GTFS joins | The API is easier and returns fares too; kept as a fallback if joins eat too much of Weeks 4–7 |
| Network-wide vs. T1/T4 | T1/T4 | Narrower claim, but the only honest one given the collection window |
