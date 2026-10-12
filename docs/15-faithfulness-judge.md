# 15. The faithfulness judge, and its validation

RQ2 asks which system explains disruptions better, and an explanation can be right
for the wrong reasons. [`08-evaluation-plan.md`](./08-evaluation-plan.md) §3.4 scores
explanations with two metrics:

- **Faithfulness rate**: the share of statements supported by a retrieved alert or a
  tool output (Papageorgiou et al., 2025).
- **Citation coverage**: the share of statements that cite at least one alert that was
  actually shown (Huang et al., 2026).

`evaluation/judge.py` computes both, `evaluation/kappa.py` validates the judge against
the author, and `transit-judge` runs them. The judge asks *does the evidence support
this?*, never *is this right?*. A faithful wrong answer and an unfaithful right one are
both possible, and cause macro-F1 ([`13`](./13-reasons.md)) already measures the second
question.

## 1. What is judged, against what

**Per statement, never per answer.** An answer is split into sentences, and each one is
judged on its own. An answer-level verdict would forgive one invented clause inside
otherwise grounded text. The split is deterministic:

- a sentence ends at `.`, `!` or `?`, followed by a space and a capital, digit, quote
  or bracket;
- decimals ("15.0"), alert ids, and the abbreviations these answers use ("St.",
  "Mt.", "e.g.", "i.e.", "vs.", "approx.") do not end one;
- a line break does, so each list item is judged alone.

**The judge sees the system's evidence, and never the ground truth.**

| Answer | Evidence the judge sees |
| --- | --- |
| Reasons stage (`transit-reasons`) | The user prompt the model was sent, verbatim. Each record keeps it in `prompt` |
| Orchestrator (`transit-ask`) | The moment the agent was told it is, the question, and every tool output, verbatim from the transcript |

Neither system prompt holds a fact about the line, the delays or any incident. One
holds instructions and the four cause-group definitions. The other holds instructions
and the moment, which is included above in the system prompt's own words, so "this
morning" can be checked.

**Citation coverage needs no judge.** It is a string check: does the statement name an
8-hex-digit alert id that the evidence actually contained? A reasons answer cites in a
structured field rather than in its text, so the field applies to each of its
sentences. The agent cites inline. Its statements about live conditions rest on tool
outputs, which carry no alert id, so its coverage is low by design. Read coverage beside
faithfulness, never alone.

## 2. The judge

- **The model** is `ANTHROPIC_JUDGE_MODEL`, now `claude-opus-4-6`, chosen by validation
  (§4). The first judge, `claude-haiku-4-5`, failed it. The judge answers through
  structured output (`JudgeVerdict`: supported, and a one-sentence rationale).
- **Temperature 0.** The Python SDK's 1.x signatures dropped sampling parameters, but
  Opus 4.6, Sonnet 4.6 and Haiku 4.5 still accept them. The judge sends `temperature: 0`
  through `extra_body`, as the SDK's upgrade guide prescribes. Newer models reject any
  sampling parameter, and for those it is left out, with the run printing "default
  sampling: re-runs may differ". This is why the judge is not a newer model: a judge
  whose verdicts change between runs cannot be validated once and trusted afterwards.
- **A failed verdict is never guessed.** A refusal, a cut-off answer or an API error is
  recorded with `supported: null`, left out of the rate, and counted in `report`.
- **Every verdict names the rules it applied.** Each judged row carries the judge model
  and a 12-digit fingerprint of the judge's instructions, message and answer schema.
  `report` and `kappa` print the fingerprint, and they refuse a file that mixes two. A κ
  vouches only for rates computed under its fingerprint.

## 3. The rubric, and how it was revised

The author labels with exactly the rules the judge is given. This is the frozen
version, `e926f55c0cf7`:

> Supported means every factual claim in the statement is in the evidence or follows
> directly from it. Numbers, times, places, causes and alert ids must match.
>
> - Read each claim in its plain sense, as a careful reader would. A paraphrase that
>   keeps the evidence's meaning is supported: "up to 9 minutes late", "no more than 9
>   minutes late" and "the most delayed was 9 minutes behind" say the same thing.
> - A hedged claim ("likely", "may", "suggests") is supported only if the evidence
>   supports it at the strength claimed.
> - Delay figures show how many services were late, by how much, and where. They do not
>   show why. Attributing the delays to a cause that nothing in the evidence links them
>   to, such as a signal fault, congestion or police activity, is unsupported however
>   it is hedged. Saying that nothing points to a cause attributes none.
> - An alert or a past incident in the evidence can link a cause to the delays. The
>   statement must say what connects them, such as the same stations, that connection
>   must be in the evidence, and a past incident's cause must be offered as possible,
>   not as certain.
> - A statement that something is unknown or cannot be determined is supported when
>   the evidence does not determine it.
> - A statement with no factual claim, such as framing or a restated question, is
>   supported.
> - A claim the evidence does not contain is unsupported, even if it may be true. That
>   includes facts about the network, such as which line or area a station is on,
>   unless the evidence states them.
>
> Judge only against the evidence. Use no outside knowledge.

**The rubric was revised twice on 2026-10-05, before any author label existed.** Each
revision fixed a way the judge misapplied its own rules on the development set. The
development set is the 72 answers of the reasons run recorded in `13` §5, plus three
agent transcripts: 87 statements from 75 answers.

| Version | Fingerprint | Agent | Model + retrieval | Model, no retrieval | Why it was revised |
| --- | --- | --- | --- | --- | --- |
| 1 | `c3ea2f9d1d8f` | 90% | 50% | 74% | Called "no service more than 3 minutes late" unsupported when the worst was 3 minutes. Applied its outside-knowledge rule to some station geography and not to the rest. Read "none of the alerts cite a cause for these delays" as false because maintenance alerts name a cause |
| 2 | `37e52d36de8f` | 100% | 29% | 56% | Over-applied its new cause rule to statements that *no* cause is indicated. Had no rule for a cause carried by a past incident, which is exactly what retrieval exists to supply |
| 3 | `e926f55c0cf7` | 100% | 42% | 62% | Frozen |

The table shows faithfulness rates. Between versions, 16 verdicts and then 9 changed.
Citation coverage does not depend on the judge: 30%, 53% and 0% throughout.

**The rubric moved the retrieval system's rate by 21 points.** That is the strongest
argument for validating the judge before trusting it, and for freezing it before
scoring. The development numbers are not results.

**Two choices in the rubric, both confirmed by the supervisor:**

1. **Network geography is outside knowledge.** "Lindfield, Chatswood and Roseville are
   North Shore Line stations" is unsupported unless the evidence says so. A lenient
   rule would need a boundary between background and claim that two raters would draw
   differently, and the situation never tells the model which line a station is on.
2. **A past incident may carry a cause to the present**, if the statement says what
   connects them, that connection is in the evidence, and the cause is hedged. Without
   this rule, every retrieval-based explanation is unfaithful by construction, and the
   metric would penalise the reasons stage for doing what it is designed to do.

Changing either is a one-line edit to `JUDGE_SYSTEM`. It produces a new fingerprint,
and the sample must then be re-scored.

## 4. Validation: κ on a blind sample of fresh answers

`08` §3.5: the author hand-labels a 20% stratified subsample, and Cohen's κ against the
judge must reach 0.6 before the full run.

**The sample comes from answers generated after the rubric was frozen.** The rubric was
revised on the development statements, so validating it on them would partly measure
the revision. A fresh validation set was generated on 2026-10-05 and judged with
`e926f55c0cf7`. It holds 92 statements from 76 answers:

- a new `transit-reasons --model --repeats 3` run on the same 12 incidents, which
  recorded its prompts natively (72 answers);
- four new `transit-ask` questions, at moments not asked about before, which between
  them use all four tools (4 answers).

**Stratified by system and verdict.** A plain random 20% of mostly supported
statements could miss the unsupported ones entirely, and κ would then say nothing
about the case that matters. Each stratum gives 20% rounded up, so none is empty. The
sample has 21 statements:

| System | Judged supported | Judged unsupported |
| --- | --- | --- |
| Agent | 3 | 1 |
| Model + retrieval | 2 | 6 |
| Model, no retrieval | 4 | 5 |

**Blind.** The sheet `data/judge_sample_20261005.csv` holds each statement, its
evidence and an empty `human_supported` column. It leaves out the judge's verdict,
because a label made beside a verdict is anchored by it. The author marks each
statement `yes` or `no` against the rubric in §3, reading only the evidence beside it.
Then:

```bash
transit-judge kappa --judged data/judged_val_20261005.jsonl --labels data/judge_sample_20261005.csv
```

This prints κ, the raw agreement, and every disagreement with the judge's rationale. It
exits 2 below the floor. If κ < 0.6, the judge is revised, `transit-judge rescore`
re-judges exactly the labelled statements, and both κs are reported.

### The result, 2026-10-13

The author labelled all 21: 10 supported and 11 not, each with a note.

**Against `claude-haiku-4-5`, κ was 0.52**, below the floor. Its 95% bootstrap interval
over the statements is [0.13, 0.88], and 16 of 21 agreed. All five disagreements were
the judge misapplying its own rubric. None came from the rubric being unclear:

- **Congestion, twice.** It passed two statements naming congestion as the cause.
  The rubric gives congestion as its own example of an unsupported cause.
- **"Within 3 minutes".** It failed "all 13 services were within 3 minutes of
  schedule" when the most delayed was 3 minutes behind. The paraphrase rule covers this.
- **A hedged analogy.** It failed an analogy that named the same stations and was
  hedged, which the past-incident rule allows.
- **A current alert.** It failed a statement attributing delays to a current alert
  that says "due to urgent train repairs". Only a past incident needs hedging.

**The remedy was the one §8 named in advance: a stronger judge under the same frozen
rubric.** Rewriting the rubric around these 21 statements would have fitted κ to its own
sample. Of the stronger models, only Claude Opus 4.6 and Sonnet 4.6 still take
`temperature: 0`. Opus 4.6 was chosen before its κ was seen.

**Against `claude-opus-4-6`, κ is 0.71**, which passes: [0.36, 1.00], with 18 of 21 in
agreement. It still fails the statement attributing delays to the current Martin Place
alert. It also passes two analogies the author found unsupported: a "system-wide
knock-on effect" from an earlier incident, and a close match to the Edgecliff
incident. **The judge is now Opus 4.6.**

At 21 statements κ is noisy, and the interval still reaches below the floor. Round 2,
below, checks it on held-out statements. Labelling a blind sample of the scored run's
statements would also show whether κ holds on test answers. It is recommended if time
allows.

### Round 2: a held-out check, drawn 2026-10-13

Round 1's 21 statements rejected Haiku and led to Opus, so Opus's κ on them is not
independent of its choice. The author labels a second blind sample, with nothing an
earlier round drew:

```bash
transit-judge sample --judged data/judged_val_opus46_20261013.jsonl \
                     --exclude data/judge_labels_20261013.csv --share 0.4 --seed 1 \
                     --out data/judge_sample_round2_20261013.csv
```

It draws 30 of the 71 statements round 1 left, stratified by Opus 4.6's verdicts. The
share is 40% of each stratum, rounded up, which gives the 30 the author had time for:

| System | Judged supported | Judged unsupported |
| --- | --- | --- |
| Agent | 5 of 12 | none left |
| Model + retrieval | 5 of 12 | 7 of 16 |
| Model, no retrieval | 6 of 15 | 7 of 16 |

The agent's one unsupported statement was in round 1, so round 2 cannot test that case.

**The rule, fixed before any label exists.**
- **Round 2 alone gives κ ≥ 0.6:** the Opus 4.6 judge stands.
- **Below 0.6:** the judge has failed on held-out statements and is revised before the
  cut-off run (`08` §3.5).

Either way, round 2's κ is reported, and so is κ over all 51 statements, each with its
bootstrap interval.

## 5. Found along the way: the situation contradicted itself

The judge flagged an answer that said four late services were "clustered mostly at
Bondi Junction" because the situation read:

> … 38 services were observed and 4 were more than five minutes late. … Late services
> were seen at Bondi Junction (5), Edgecliff (1).

Both figures were right, but they counted different things. The late total reads each
service's **latest** report, as rule (b) does. The station counts read **every** report
in the half hour, counting a service at each station where it was late. A station could
therefore hold more late services than were late in total. This happened in 2 of the
12 incidents.

The measures were kept, and the wording now says which report each figure reads:

> T1, 13:45 on Sunday 27 September 2026 (Sydney time). In the 30 minutes before, 35
> services were observed. When last reported, 6 were more than five minutes late and
> the most delayed was 19 minutes behind. At some point in those 30 minutes, services
> were reported more than five minutes late at Parramatta (4), Harris Park (3),
> Lidcombe (3), Auburn (2), Granville (2).

The agent's `line_status` keys were renamed to match:
`more_than_5_min_late_when_last_reported`, `worst_delay_minutes_when_last_reported` and
`more_than_5_min_late_at_some_point_by_station`. The development run in `13` §5 used the
old wording. The validation set uses the new one.

## 6. What the validation set shows

These are rates under `e926f55c0cf7`. They are development figures, not results.

| System | Statements | Faithful, Opus 4.6 (validated) | Faithful, Haiku 4.5 (failed κ) | Cited |
| --- | --- | --- | --- | --- |
| Agent | 16 | 94% | 94% | 25% |
| Model + retrieval | 36 | **47%** | 19% | 64% |
| Model, no retrieval | 40 | 45% | 48% | 0% |

**The pattern flagged before validation did not survive it.** Under Haiku, retrieval
seemed to lower faithfulness, 19% against 48%. Under the validated judge, the two are
level at 47% and 45%. Haiku was failing the past-incident analogies the rubric allows.
Without the κ check, the evaluation would have reported a judge's misreading as a
finding about retrieval. The agent, held to its checked rules, stays at 94%.

## 7. Running it

```bash
transit-reasons --db ... --model --repeats 3 --out data/reasons_runs.jsonl   # records each prompt
transit-ask --db ... --at 2026-10-01T08:50+10:00 --out data/transcripts/q1.json "..."
transit-questions write --db data/evaluation_20261018.db --out data/questions_test.csv
transit-questions ask --questions data/questions_test.csv --out-dir data/transcripts_test \
                      --db data/evaluation_20261018.db --detector ... --delay-model ...
transit-judge judge  --reasons data/reasons_runs.jsonl --transcripts data/transcripts \
                     --out data/judged.jsonl                                   # calls the judge
transit-judge report --judged data/judged.jsonl                                # rates per system
transit-judge sample --judged data/judged.jsonl --out data/judge_sample.csv    # blind, to label
transit-judge sample --judged data/judged.jsonl --exclude data/judge_sample.csv \
                     --seed 1 --out data/judge_sample_2.csv                     # a second round
transit-judge kappa  --judged data/judged.jsonl --labels data/judge_sample.csv
transit-judge rescore --judged data/judged.jsonl --labels data/labels.csv \
                      --model claude-opus-4-6 --out data/rescored.jsonl         # the same sample, again
```

`sample` never writes over a sheet, because the author may have labelled it in place.
`--exclude` refuses a sheet naming statements the judged file lacks. If they were
ignored, a wrong sheet would exclude nothing.

`judge` refuses a reasons record without its prompt and names the fix. A prompt rebuilt
later could differ from the one sent, if the index has changed since. The development
run in `13` §5 predates the `prompt` field. Its prompts were rebuilt once, with the same
function, from each record's situation and retrieved passage ids. That was done only
after checking that the index still matched the snapshot and was built before the run
wrote its file. That rebuild is what made its development verdicts possible. The
validation set needed no rebuild.

The development and validation sets took about 430k input and 30k output Haiku tokens
across the four judge runs. Opus 4.6 then took 162k input and 11k output, about $1.10,
for the re-scored sample and the validation set.

## 8. Limits, and what is not built yet

- **A statement is judged without its neighbours.** "It suggests a technical cause"
  reaches the judge without the sentence that says what "it" is. Most reasons answers
  are one sentence (77 statements from 72 answers in the development set), but the
  agent's answers run to five. If κ fails on such statements, the fix is to show the
  whole answer as context.
- **One judge model, Opus 4.6.** It is validated together with the fingerprint
  `e926f55c0cf7`. A different model, or a changed rubric, needs its own κ. The
  `judge_model` and `judge_prompt` every judged row carries make the pair checkable.
- **The scored run** is judged at the cut-off with the fingerprint κ validated. A new
  fingerprint needs its own κ.
