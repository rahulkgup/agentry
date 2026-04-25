"""Shared contact "CRM" data used by the in-process tool and the MCP stdio server.

Keeping one source of truth so Phase 9's MCP `lookup_contact_history` matches
what you previously called from `llm_tools` in-process.
"""

from __future__ import annotations

CONTACTS: dict[str, str] = {
    "priya@example.com": (
        "Friend. Mutual contact: Sam. Met at PyCon 2024. "
        "Last interacted 3 weeks ago about ML infra hiring. Casual register."
    ),
    "alvarez.rentals@example.com": (
        "Landlord at current apartment (2.5 years). Strict on deadlines. "
        "Prefers concise emails with explicit dates. Formal register."
    ),
    "receipts@stripe.com": (
        "Automated. No-reply sender. Don't draft a personal reply."
    ),
}


def format_contact_info(email_address: str) -> str:
    """Return the same string the LLM tools always saw."""
    info = CONTACTS.get(email_address.lower())
    if info is None:
        return f"No history for {email_address}."
    return f"Contact info for {email_address}: {info}"
