from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field

Category = Literal["needs_reply", "fyi", "newsletter", "promo_or_spam", "calendar"]
Urgency = Literal["high", "normal", "low"]
CriticWants = Literal["ok", "revise"]


class Email(BaseModel):
    """A single fetched email. Kept lean — body is plain text."""

    id: str
    thread_id: str | None = None
    sender: str
    sender_email: str
    subject: str
    body: str
    received_at: str  # ISO 8601 string

    def header(self) -> str:
        return f"From: {self.sender} <{self.sender_email}>\nSubject: {self.subject}"


class Classification(BaseModel):
    """Structured output from the triager agent."""

    category: Category
    urgency: Urgency
    summary: str = Field(description="One-sentence summary of what this email is about.")
    intent: str | None = Field(
        default=None,
        description="If category=needs_reply, what the sender wants from you.",
    )
    suggested_action: str | None = Field(
        default=None,
        description="If not needs_reply, the action to take (e.g. 'archive', 'read later').",
    )


class Draft(BaseModel):
    """Structured output from the drafter agent."""

    body: str = Field(description="Plain text reply body, ready to be sent as a Gmail draft.")
    notes: str | None = Field(
        default=None,
        description="Optional notes to the user about why you drafted it this way.",
    )
    retriage: bool = Field(
        default=False,
        description="True if the triager got this wrong and it should be re-classified.",
    )
    retriage_reason: str | None = Field(
        default=None,
        description="What the triager misunderstood (shown to triager on the second pass).",
    )


class CriticVerdict(BaseModel):
    """Structured output from the self-critic node."""

    wants: CriticWants
    note: str = Field(
        default="",
        description="If revise: concrete fixes. If ok: empty or brief confirmation.",
    )


def _merge_dict(left: dict, right: dict) -> dict:
    """Reducer: shallow-merge dicts so parallel branches can each contribute keys."""
    return {**(left or {}), **(right or {})}


class InboxState(TypedDict, total=False):
    """Graph state. Runs once per cron invocation."""

    emails: list[Email]
    queue: list[Email]
    current: Email | None
    classifications: Annotated[dict[str, Classification], _merge_dict]
    drafts: Annotated[dict[str, Draft], _merge_dict]
    saved_draft_ids: Annotated[dict[str, str], _merge_dict]
    summary: str
    dry_run: bool
    # Re-triage (drafter → triager, max one loop)
    triager_hint: str | None
    retriage_loop: int
    # Critic loop
    critic_wants: CriticWants | None
    critic_note: str | None
    drafter_attempts: int
