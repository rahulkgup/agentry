"""MCP (stdio) server — exposes `lookup_contact_history` as a real out-of-process tool.

Run (from repo root):
  python -m emails.inbox_agent.mcp_contact_server

The LangGraph ReAct drafter connects over stdio via `langchain-mcp-adapters`.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from emails.inbox_agent.contact_directory import format_contact_info

mcp = FastMCP("inbox-contacts")


@mcp.tool()
def lookup_contact_history(email_address: str) -> str:
    """Look up what we know about an email sender — relationship, last
    interaction, register (formal vs casual), any context that helps draft
    a reply in the right voice.

    Returns a short text summary, or "no history" if the sender is unknown.
    Always call this BEFORE drafting a reply to a person you don't recognize.
    """
    return format_contact_info(email_address)


if __name__ == "__main__":
    mcp.run()
