"""LangChain `@tool`-decorated functions exposed to LLMs.

- `tools.py`     → graph nodes (run as part of graph execution)
- `llm_tools.py` → LangChain tools (the LLM decides when to call these)

Phase 9: `lookup_contact_history` is also served by `mcp_contact_server` (stdio).
ReAct agents load tools via `mcp_tools.get_react_tools()` (MCP + `search_past_drafts`).
This module still exports in-process `lookup_contact_history` for `MCP_OFF`, tests, and
`inbox-agent tool-test`.
"""

from __future__ import annotations

from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedStore
from langgraph.store.base import BaseStore

from .contact_directory import format_contact_info


@tool
def lookup_contact_history(email_address: str) -> str:
    """Look up what we know about an email sender — relationship, last
    interaction, register (formal vs casual), any context that helps draft
    a reply in the right voice.

    Returns a short text summary, or "no history" if the sender is unknown.
    Always call this BEFORE drafting a reply to a person you don't recognize.
    """
    return format_contact_info(email_address)


@tool
def search_past_drafts(
    query: str,
    *,
    store: Annotated[BaseStore, InjectedStore()],
) -> str:
    """Semantically search past approved replies for examples that match the
    voice/structure you should use for a NEW similar email.

    Pass a short description of the incoming email (subject + intent) as the
    query. Returns up to 3 past replies, ranked by similarity.

    Use this when the user asks for help drafting a reply, especially when
    you want to match their established style for a kind of email.
    """
    items = store.search(("voice", "drafts"), query=query, limit=3)
    if not items:
        return "(no past drafts found in voice store)"
    lines = []
    for i, it in enumerate(items, 1):
        v = it.value or {}
        score = f"{it.score:.2f}" if it.score is not None else "—"
        lines.append(
            f"[{i}] score={score}  context: {v.get('text', '')}\n"
            f"    reply:\n      {v.get('reply', '').strip()}"
        )
    return "\n\n".join(lines)


TOOLS = [lookup_contact_history, search_past_drafts]
