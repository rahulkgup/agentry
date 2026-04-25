"""Deterministic "model" for offline evals and smoke tests (no API key).

In production, evals would compare a **real** :func:`~emails.inbox_agent.agents.triager_node`
to golden labels, or use an LLM-as-judge. This stub only teaches the harness pattern.
"""

from __future__ import annotations

from emails.inbox_agent.state import Classification, InboxState


def stub_triager_for_eval(state: InboxState) -> dict:
    """Same heuristics the smoke test has always used: @example.com + not noreply → needs_reply."""
    current = state.get("current")
    if current is None:
        return {}
    is_personal = (
        "@example.com" in current.sender_email and "noreply" not in current.sender_email
    )
    cls = Classification(
        category="needs_reply" if is_personal else "newsletter",
        urgency="normal",
        summary=f"stub summary for {current.subject}",
        intent="reply please" if is_personal else None,
        suggested_action=None if is_personal else "archive",
    )
    return {"classifications": {current.id: cls}}
