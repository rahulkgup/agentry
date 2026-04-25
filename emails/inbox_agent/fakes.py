"""Stub LLM nodes for `--fake-llm` / tests (no API key, no network)."""

from __future__ import annotations

from .state import Classification, Draft, InboxState


def fake_triager_node(state: InboxState) -> dict:
    current = state.get("current")
    if current is None:
        return {}
    is_personal = "@example.com" in current.sender_email and "noreply" not in current.sender_email
    cls = Classification(
        category="needs_reply" if is_personal else "newsletter",
        urgency="normal",
        summary=f"stub summary for {current.subject}",
        intent="reply please" if is_personal else None,
        suggested_action=None if is_personal else "archive",
    )
    return {"classifications": {current.id: cls}}


def fake_drafter_node(state: InboxState) -> dict:
    current = state.get("current")
    if current is None:
        return {}
    # Short body; retriage/critic paths are skipped when fake_llm in graph
    return {
        "drafts": {
            current.id: Draft(
                body=f"Hi {current.sender.split()[0]},\n\nSounds good!\n\nRahul",
            )
        }
    }


def fake_critic_node(state: InboxState) -> dict:
    """Always approve (graph may skip this node entirely when fake_llm)."""
    return {"critic_wants": "ok", "critic_note": ""}
