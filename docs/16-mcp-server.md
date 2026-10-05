# 16. The MCP server

`transit-mcp` (`src/transit_rag/mcp_server/server.py`) serves the orchestrator's four
tools ([`14-agent.md`](./14-agent.md)) to any MCP client, such as Claude Code or
Claude Desktop. The tools are not a second implementation. Each MCP tool calls the
agent's own `ToolBox.call`, so a client and `transit-ask` get the same answer from the
same code.

## 1. What a client gets

| | As the agent has it |
| --- | --- |
| **Tools** | `line_status`, `disruption_risk`, `predict_delays`, `similar_past_incidents`, with the agent's descriptions from `TOOL_SPECS` |
| **Input** | `line`, which is `T1` or `T4`, and required. The enum is checked against the agent's `LINES` in a test |
| **Output** | The agent's JSON, as structured content |
| **A failure** | An MCP error result carrying the agent's message, for example that no detector is loaded. Never a traceback |
| **Instructions** | The agent's system prompt: the moment, and the rules the evaluation checks |

Every tool is marked read-only, idempotent and closed-world. It reads a frozen snapshot
and changes nothing.

**The instructions carry the rules because the evaluation checks them.** Probabilities
are stated as estimates, intervals are stated unnarrowed, and causes are given only with
alert ids. A client's model that never saw them could misstate what the tools return.

## 2. One moment per server

Like `transit-ask`, the server answers as of `--at` in a frozen snapshot. The tools see
what the snapshot recorded before then and nothing after. The live feed behind the tools
is not built (`14` §7), so the server replays a moment rather than reporting the
present. The same arguments as `transit-ask` choose the snapshot and the models, through
the same function (`agent/cli.py: build_toolbox`), so the two cannot drift apart.

## 3. Running it

It needs the `serve` extra. The SDK is `mcp` 2.x, where `FastMCP` became `MCPServer`.

```bash
uv sync --frozen --inexact --extra dev --extra realtime --extra prediction --extra rag --extra serve
```

To add it to Claude Code from the repo root, with every path derived and nothing to fill
in:

```bash
claude mcp add transit-rag -- "$(pwd)/.venv/bin/transit-mcp" \
  --db "$(pwd)/data/delay_observations_20261005.db" --at 2026-10-01T08:50+10:00 \
  --detector "$(pwd)/models/detector_20261005.joblib" \
  --delay-model "$(pwd)/models/delay_model_20261005.joblib"
```

Over stdio, standard output belongs to the protocol, so nothing in the server prints to
it. A startup error, such as a missing snapshot or an index that does not match it, goes
to standard error with a non-zero exit, and a test pins both.

**Checked live on 2026-10-05.** The SDK's own client launched the server as a subprocess
on the 20261005 snapshot at 08:50 on 1 October. It listed the four tools. `line_status`
returned the same 68 services and 13 late that the agent saw (`14` §5).
`disruption_risk` returned 0.869, flagged, for 08:30–08:45.
`similar_past_incidents` returned five earlier incidents.

## 4. Not built yet

- **Live mode**: the same tools over the realtime client instead of a snapshot (`14` §7).
- **An HTTP transport.** `MCPServer` can serve streamable HTTP without FastAPI, so the
  FastAPI surface in `02` may be unnecessary. That is a decision for the deployment
  phase, and nothing here depends on it.
