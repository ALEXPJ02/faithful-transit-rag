"""``transit-mcp`` -- the agent's tools, served to any MCP client.

    transit-mcp --db data/delay_observations_20261005.db --at 2026-10-01T08:50+10:00
    transit-mcp --db ... --at ... --detector models/detector_20261005.joblib \\
                --delay-model models/delay_model_20261005.joblib

It serves the four tools ``transit-ask`` gives its own loop (``agent/tools.py``) over
stdio, with the same names, inputs and descriptions. They answer as of ``--at`` in a
frozen snapshot, exactly as the agent's do (``docs/16``). The server's instructions are
the agent's system prompt, so a client's model is told the moment and the rules the
evaluation checks. A tool that fails returns an MCP error result carrying the message
the agent would have seen, never a traceback.

Over stdio, standard output belongs to the protocol. Nothing here prints to it, and a
startup error goes to standard error.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from typing import Any, Literal

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from transit_rag.agent.cli import add_toolbox_arguments, build_toolbox
from transit_rag.agent.loop import system_prompt
from transit_rag.agent.tools import TOOL_SPECS, ToolBox
from transit_rag.config import ConfigError

#: The lines a tool takes. ``agent.tools.LINES`` as a type, so the schema MCP derives
#: from the signature holds the same enum the agent's tools declare.
Line = Literal["T1", "T4"]

#: Every tool reads a frozen snapshot, changes nothing, and gives the same answer twice.
READ_ONLY = ToolAnnotations(read_only_hint=True, idempotent_hint=True, open_world_hint=False)


def _tool(tools: ToolBox, name: str) -> Callable[[Line], dict[str, Any]]:
    """One of ``tools`` as an MCP tool function, going through the agent's own dispatch."""

    def run(line: Line) -> dict[str, Any]:
        output, failed = tools.call(name, {"line": line})
        result: dict[str, Any] = json.loads(output)
        if failed:
            raise ToolError(str(result.get("error", output)))
        return result

    run.__name__ = name
    return run


def build_server(tools: ToolBox) -> MCPServer:
    """``tools`` under the agent's tool names and descriptions, with its rules as instructions."""
    server = MCPServer("transit-rag", instructions=system_prompt(tools.at))
    for spec in TOOL_SPECS:
        server.add_tool(
            _tool(tools, spec["name"]),
            name=spec["name"],
            description=spec["description"],
            annotations=READ_ONLY,
        )
    return server


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="transit-mcp",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_toolbox_arguments(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        tools = build_toolbox(args)
    except (FileNotFoundError, ValueError, ConfigError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    build_server(tools).run("stdio")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
