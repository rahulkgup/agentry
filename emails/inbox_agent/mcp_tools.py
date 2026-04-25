"""Load LangChain tools for the ReAct drafter: MCP `lookup_contact_history` + in-process `search_past_drafts`."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from .config import settings
from .llm_tools import lookup_contact_history, search_past_drafts

ROOT = Path(__file__).resolve().parents[2]

# Populated on first `get_react_tools()`; MCP subprocess should start once.
_react_tools_cache: list[BaseTool] | None = None


def get_react_tools() -> list[BaseTool]:
    """Tools for `create_react_agent` and manual `ToolNode`.

    - With MCP (default): stdio server for `lookup_contact_history` + `InjectedStore` search.
    - With FAKE_LLM or MCP_OFF=1: in-process `lookup_contact_history` (no subprocess).
    """
    global _react_tools_cache
    if _react_tools_cache is not None:
        return _react_tools_cache

    if settings.fake_llm or os.environ.get("MCP_OFF", "").lower() in ("1", "true", "yes"):
        _react_tools_cache = [lookup_contact_history, search_past_drafts]
        return _react_tools_cache

    connections = {
        "contacts": {
            "command": sys.executable,
            "args": ["-m", "emails.inbox_agent.mcp_contact_server"],
            "transport": "stdio",
            "cwd": str(ROOT),
        }
    }
    client = MultiServerMCPClient(connections=connections)
    mcp_tools: list[BaseTool] = asyncio.run(client.get_tools())
    # Store-backed tool must run in the graph's Python process
    _react_tools_cache = mcp_tools + [search_past_drafts]
    return _react_tools_cache


def clear_react_tools_cache() -> None:
    """Mostly for tests that toggle FAKE_LLM / MCP between runs."""
    global _react_tools_cache
    _react_tools_cache = None
