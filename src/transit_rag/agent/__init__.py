"""The hand-rolled Anthropic tool-use loop.

Not a framework. The agent decides, per query, whether to retrieve from the
static corpus, call a live GTFS-Realtime tool, call the delay model, or some
combination — and it is required to state the model's error margin whenever
an answer rests on a prediction rather than a retrieved or observed fact.
"""
