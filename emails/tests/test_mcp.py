"""Tests for the MCP integration in `emails.inbox_agent.mcp_tools`.

Two paths are exercised:

1. ``MCP_OFF=1`` returns the in-process LangChain tools (no subprocess; runs in CI).
2. The default path spawns the FastMCP stdio server in a subprocess and loads the
   `lookup_contact_history` tool through `langchain-mcp-adapters`. This second
   test is opt-in via ``-m integration`` to avoid spawning subprocesses by default.

Run only the offline check::

    uv run pytest emails/tests/test_mcp.py -q

Run including the subprocess check::

    uv run pytest emails/tests/test_mcp.py -q -m integration
"""

from __future__ import annotations

import shutil

import pytest
from langchain_core.tools import BaseTool

from emails.inbox_agent import mcp_tools
from emails.inbox_agent.contact_directory import format_contact_info


@pytest.fixture(autouse=True)
def _isolate_mcp_state(monkeypatch):
    """Each test gets a fresh tool cache + clean MCP_OFF/FAKE_LLM env."""
    monkeypatch.delenv("MCP_OFF", raising=False)
    monkeypatch.delenv("FAKE_LLM", raising=False)
    mcp_tools.clear_react_tools_cache()
    yield
    mcp_tools.clear_react_tools_cache()


def test_mcp_off_returns_in_process_tools(monkeypatch):
    """`MCP_OFF=1` skips the subprocess and returns LangChain tools directly."""
    monkeypatch.setenv("MCP_OFF", "1")

    tools = mcp_tools.get_react_tools()

    names = [t.name for t in tools]
    assert names == ["lookup_contact_history", "search_past_drafts"]
    for t in tools:
        assert isinstance(t, BaseTool)

    lookup = next(t for t in tools if t.name == "lookup_contact_history")
    expected = format_contact_info("priya@example.com")
    assert lookup.invoke({"email_address": "priya@example.com"}) == expected
    assert "Friend" in expected  # contact directory is wired up


def test_get_react_tools_is_cached(monkeypatch):
    """Second call should be a cache hit (same list instance)."""
    monkeypatch.setenv("MCP_OFF", "1")
    first = mcp_tools.get_react_tools()
    second = mcp_tools.get_react_tools()
    assert first is second


@pytest.mark.integration
def test_mcp_stdio_loads_tool():
    """Boots the FastMCP stdio server and loads `lookup_contact_history` over MCP.

    Requires `python` on PATH (it's the `uv` venv interpreter in CI).
    """
    if shutil.which("python") is None and shutil.which("python3") is None:
        pytest.skip("no python interpreter on PATH for MCP subprocess")

    tools = mcp_tools.get_react_tools()
    names = {t.name for t in tools}

    assert "lookup_contact_history" in names, f"got: {names}"
    assert "search_past_drafts" in names, "in-process tool should be appended"

    lookup = next(t for t in tools if t.name == "lookup_contact_history")
    assert isinstance(lookup, BaseTool)
