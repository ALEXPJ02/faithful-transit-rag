# The Orchestrator

> RQ1's workflow as one agent. A hand-rolled Claude tool-use loop calls each stage as
> a tool and writes the answer, as of a moment in a frozen snapshot. Built on
> 2026-10-05 on the stages of [`11`](./11-disruption-labels.md)–[`13`](./13-reasons.md).

## 1. The loop

`agent/loop.py`. It is written by hand rather than with a framework, which is a
settled decision (`CLAUDE.md`), so that every request, tool call and result is
visible. It follows the SDK's documented shape:

- the assistant's whole `content` is appended each turn, which keeps any thinking
  blocks intact;
- all of a turn's tool results go back in one user message, each tied to its call's
  id;
- a failed tool goes back marked `is_error`.

A refusal or a truncated turn ends the loop and is recorded as it stands, never
retried blindly. Eight turns is the cap. A question needs at most one call per stage
per line, plus a follow-up or two.

**Everything the evaluation reads is recorded.** A `Transcript` keeps the question,
the moment, every tool call with its exact output, the answer, the stop reason and
the tokens. Tool-faithfulness (`08` §2, O1) compares the answer's claims with those
outputs, so they are stored as the model saw them, byte for byte.

## 2. The tools (`agent/tools.py`)

| Tool | Stage | Returns |
| --- | --- | --- |
| `line_status` | O1 | Services observed in the last 30 minutes. How many were late, and the worst delay, when last reported. The stations where services were late at any point in that half hour, and the operator's alerts in the feed |
| `disruption_risk` | O2 | The saved detector's probability that the line is disrupted in the next 30 minutes (`12`), its threshold and validation AP, and **whether the date was in its training** |
| `predict_delays` | O2 | For each train running now, its next station, how late it is, and its expected delay there **with the 90% interval** (`08` §4) |
| `similar_past_incidents` | O3 | The five past incidents most like the present, first seen before now, with the alert ids to cite (`13` §2) |

The schemas are `strict`, so a line other than T1 or T4 cannot be asked for. A handler
never raises into the loop. A failure comes back as an error result the model has to
report.

## 3. As of a moment

Every tool reads through a `SnapshotFeed` and one moment (`agent/feed.py`). Live
conditions are scored against frozen snapshots, never the live API (`08` §3.5), so a
re-run reproduces the answer's evidence exactly. Each tool sees only what existed by
the moment:

- **an alert** is in the feed if the last alert poll at or before the moment carried
  it, the same causal rule as the detector's alert feature;
- **retrieval** admits only incidents first seen before the moment;
- **risk** scores the last *complete* window, never the one still in progress.

The detector's own features are causal (`12` §2). If the moment falls on a date the
detector trained or was chosen on, the tool says so, because a detector asked about
a window it was fitted on reports what it memorised. Evaluation questions belong on
test dates. One residual is carried over from `10` §5: an alert republished with new
wording shows its final text.

## 4. The checked rules are in the prompt from the first version

These are never bolted on (`08` §7). The system prompt states the moment in Sydney
time and four rules the evaluation checks:

- live conditions only as a tool reported them;
- a probability as the estimate the tool returned, never rounded into certainty;
- **a predicted delay with its 90% interval, exactly as the tool returned it**, which
  is `01` §2's hard requirement;
- a cause only with the alert ids it rests on, otherwise said to be unknown;
- a failed tool said to have failed.

## 5. Two development questions

These were asked on 2026-10-05 against the live configured model (`claude-sonnet-5`)
and the 2026-10-05 snapshot. They are smoke tests of the integration, not results.
Their transcripts are local, in `data/transcripts/`.

**"Is the T1 disrupted right now, and why?" at 08:50 on 1 Oct.** The model called
`line_status`, which reported 13 of 68 services more than five minutes late, around
Central, Milsons Point and North Sydney. It found alert `ad28913f`, a person on the
tracks at Central, and cited it. It set aside the eleven other alerts in the feed,
which were planned maintenance and standing notices, naming three as examples of
alerts unrelated to the morning. It took 2 turns, with 4.5k tokens in and 292 out.

**"Is the T4 likely to be disrupted in the next half hour, and what has caused delays
like this before?" at 10:30 on 2 Oct, a test date.** The model called all three tools
in one turn. It gave the detector's 17.4% as a model estimate, flagged above the
14.4% threshold. The next half hour **was** disrupted: rule (b) marks T4's 10:30
window. It reported the live delays around Hurstville and Sutherland, and cited past
T4 incidents by alert id. It then declined to attribute the present delays to any of
those causes, because no current alert linked them. That is the behaviour rule three
asks for.

**"How late are T4 trains running right now, and how late will they be at their next
stops?" at the same moment, with the delay model.** `predict_delays` found 23 trains
running, predicted 21, and expected 5 to be more than five minutes late. The answer
stated **every** prediction with its interval exactly as returned, for example
"Caringbah 15.0 min [12.4, 17.6], Heathcote 9.2 min [6.6, 11.8]", and called them model
estimates with 90% intervals. With no alert to support one, it said the cause was not
known.

### How `predict_delays` decides what to predict

A train is running if it was observed in the ten minutes before the moment. Its delay
now is its last completed stop's, its latest reliable observation before the moment.
Its next stop and scheduled time come from the timetable, because the feed's
`stop_sequence` is always the sentinel. The era used is the newest fetched on or before
the service date, then up to two older ones. The features are assembled exactly as
training built them. A stop the model never saw is made missing explicitly rather than
coerced, because pandas will refuse that coercion in a later version. The interval is
the one calibrated beside the model: line × how late the train already is.

## 6. Running it

```bash
transit-detect --db data/delay_observations_20261005.db --save models/detector_20261005.joblib
transit-train --table data/training_table_20261005.csv \
    --model-out models/delay_model_20261005.joblib --metrics-out models/delay_model_20261005_metrics.json
transit-ask --db data/delay_observations_20261005.db --at 2026-10-02T10:30+10:00 \
    --detector models/detector_20261005.joblib \
    --delay-model models/delay_model_20261005.joblib --out data/transcripts/q.json \
    "Is the T4 likely to be disrupted in the next half hour?"
```

`--at` must carry its UTC offset. A naive time is ten or eleven hours ambiguous in
Sydney, and it is refused. Without `--detector`, the risk tool reports that it cannot
answer, and the answer has to say so.

## 7. Not built yet

- **The MCP server** (`mcp_server/`), which exposes the same tools to any MCP client.
- **Live mode**: the same tools over the realtime client instead of a snapshot.
- **Scoring the orchestrator**: a fixed question set on test dates, with
  tool-faithfulness and citation coverage from the judge
  ([`15`](./15-faithfulness-judge.md)). Four questions are already in the judge's
  validation set.
