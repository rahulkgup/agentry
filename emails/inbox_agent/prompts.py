from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .config import settings

_VOICE_DIR = Path(__file__).resolve().parent / "voice_examples"

TRIAGER_SYSTEM = """\
You are an inbox triage assistant for {user_name}.

Classify the email into exactly one category:
- needs_reply: a real human is waiting on a response from {user_name}.
- fyi: informational, no response expected (receipts, confirmations, status updates).
- newsletter: subscriptions, digests, marketing newsletters.
- promo_or_spam: promotional, sales, cold outreach, or low-quality bulk mail.
- calendar: meeting invites, reschedules, calendar notifications.

Be conservative with `needs_reply` — only choose it when a reply is genuinely warranted.
Automated "no-reply" senders, marketing, and notifications are NEVER `needs_reply`.

Also assign:
- urgency: high (deadline today/tomorrow, urgent ask), normal (standard reply expected), low.
- summary: one sentence, factual.
- intent: if needs_reply, what the sender actually wants (in plain English).
- suggested_action: if not needs_reply, what to do with it.

Return ONLY the structured object.
"""

DRAFTER_SYSTEM = """\
You are drafting a reply that will be saved as a Gmail draft for {user_name} to review.

Voice / style notes from {user_name}:
{voice_notes}

Rules:
- Address the sender directly. Be warm but efficient.
- Match the length to the request: short questions get short answers.
- If you don't know something, leave a clear `[TODO: ...]` placeholder rather than making it up.
- Do NOT include a subject line — Gmail handles that for replies.
- Sign off as "{user_name}" (first name only) unless the voice notes say otherwise.
- Plain text only. No markdown, no HTML.

Return ONLY the structured object.
"""


def triager_system() -> str:
    return TRIAGER_SYSTEM.format(user_name=settings.user_name)


def drafter_system() -> str:
    return DRAFTER_SYSTEM.format(
        user_name=settings.user_name,
        voice_notes=settings.user_voice_notes,
    )


def voice_file_examples_block() -> str:
    """Load optional `*.txt` samples from `voice_examples/` (Backlog: few-shot anchoring)."""
    if not _VOICE_DIR.is_dir():
        return ""
    files = sorted(_VOICE_DIR.glob("*.txt"))
    if not files:
        return ""
    parts: list[str] = []
    for p in files:
        text = p.read_text().strip()
        if text:
            parts.append(f"— Example ({p.name}):\n{text}")
    if not parts:
        return ""
    return (
        "\nReal reply samples to match in tone and structure (not verbatim):\n"
        + "\n\n".join(parts)
        + "\n"
    )


def voice_examples_block(items: Sequence[object]) -> str:
    """Format past-reply Store items as a few-shot examples block.

    Items are `Item` objects from `store.search()` — each has `.value` with
    `{"text": <context>, "reply": <past reply>}` keys.
    """
    if not items:
        return ""
    parts = [
        f"Here are {len(items)} examples of how I've replied to similar emails before. "
        "Match the voice / structure / sign-off — don't copy verbatim.\n"
    ]
    for i, it in enumerate(items, 1):
        v = getattr(it, "value", {}) or {}
        parts.append(
            f"Example {i}:\n"
            f"  Incoming context: {v.get('text', '')}\n"
            f"  My reply:\n    {v.get('reply', '').strip()}\n"
        )
    return "\n".join(parts) + "\n"
