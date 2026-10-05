"""Tests for the MCP server (``mcp_server/server.py``).

The server runs in-process behind the SDK's own client, so these check what an MCP
client actually receives: the agent's tool names, descriptions and inputs, the
agent's rules as instructions, the agent's answer as structured content, and a
failure as an error result rather than a crash.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("mcp", reason="install the 'serve' extra")

import anyio
from mcp.client import Client
from mcp.types import TextContent

from transit_rag.agent.feed import SnapshotFeed
from transit_rag.agent.loop import system_prompt
from transit_rag.agent.tools import LINES, TOOL_SPECS, ToolBox
from transit_rag.mcp_server.server import build_server, main

AT = datetime(2026, 10, 1, 22, 50, tzinfo=UTC)  # 08:50 on 2 October in Sydney


def _tools() -> ToolBox:
    """Six T1 services ten minutes ago, two of them late, and no detector loaded."""
    events = pd.DataFrame(
        {
            "line": "T1",
            "service_date": "2026-10-02",
            "trip_id": [f"t{i}" for i in range(6)],
            "stop_id": "2000331",
            "stops_ahead": 0,
            "delay_s": pd.array([600, 420, 0, 0, 0, 0], dtype="Float64"),
            "observed_at": pd.to_datetime([AT - timedelta(minutes=10)] * 6, utc=True),
        }
    )
    polls = pd.Series(pd.date_range(AT - timedelta(hours=6), AT, freq="30min"))
    feed = SnapshotFeed(Path("unused.db"), events, [], polls, {"2000331": "Central"})
    return ToolBox(feed=feed, at=AT)


def _with_client(tools: ToolBox, check: Callable[[Client], Awaitable[None]]) -> None:
    async def run() -> None:
        async with Client(build_server(tools)) as client:
            await check(client)

    anyio.run(run)


def test_a_client_is_offered_the_agents_tools_under_the_agents_rules() -> None:
    async def check(client: Client) -> None:
        listed = (await client.list_tools()).tools
        assert [tool.name for tool in listed] == [spec["name"] for spec in TOOL_SPECS]
        for tool, spec in zip(listed, TOOL_SPECS, strict=True):
            assert tool.description == spec["description"]
            assert tool.input_schema["properties"]["line"]["enum"] == list(LINES)
            assert tool.input_schema["required"] == ["line"]
            assert tool.annotations is not None and tool.annotations.read_only_hint is True
        # The moment and the checked rules, exactly as the agent is told them.
        assert client.instructions == system_prompt(AT)

    _with_client(_tools(), check)


def test_a_tool_answers_a_client_exactly_as_it_answers_the_agent() -> None:
    tools = _tools()
    agent_sees, failed = tools.call("line_status", {"line": "T1"})
    assert not failed

    async def check(client: Client) -> None:
        result = await client.call_tool("line_status", {"line": "T1"})
        assert not result.is_error
        assert result.structured_content == json.loads(agent_sees)

    _with_client(tools, check)


def test_a_failure_is_an_error_result_with_the_message_the_agent_gets() -> None:
    tools = _tools()
    agent_sees, failed = tools.call("disruption_risk", {"line": "T1"})
    assert failed  # no detector is loaded

    async def check(client: Client) -> None:
        result = await client.call_tool("disruption_risk", {"line": "T1"})
        assert result.is_error
        message = result.content[0]
        assert isinstance(message, TextContent)
        assert json.loads(agent_sees)["error"] in message.text
        wrong_line = await client.call_tool("line_status", {"line": "T9"})
        assert wrong_line.is_error

    _with_client(tools, check)


def test_a_startup_error_goes_to_stderr_and_stdout_stays_the_protocols(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["--db", str(tmp_path / "absent.db"), "--at", "2026-10-01T08:50+10:00"]
    assert main([*argv, "--bundle", str(tmp_path / "absent.zip")]) == 1
    printed = capsys.readouterr()
    assert printed.out == ""
    assert printed.err.startswith("error: ")
