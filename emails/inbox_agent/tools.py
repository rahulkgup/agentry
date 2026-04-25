"""I/O nodes: Gmail fetch/save, queue management, terminal summary.

Sample fixtures live here too — they're only consumed by `fetch_emails_node`
in dry-run mode, so it's not worth a separate file.
"""

from __future__ import annotations

import base64
from collections import Counter
from email.message import EmailMessage
from typing import TYPE_CHECKING

import uuid

from langgraph.config import get_store, get_stream_writer
from langgraph.types import Command, interrupt
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .config import settings
from .state import Draft, Email, InboxState

VOICE_NAMESPACE = ("voice", "drafts")

if TYPE_CHECKING:
    from googleapiclient.discovery import Resource

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.compose",
]

console = Console()


# ---------------------------------------------------------------------------
# Dry-run fixtures
# ---------------------------------------------------------------------------

SAMPLE_EMAILS: list[Email] = [
    Email(
        id="msg_001",
        thread_id="thread_001",
        sender="Priya Patel",
        sender_email="priya@example.com",
        subject="Quick favor — coffee chat next week?",
        body=(
            "Hey! Hope you're well. I'm trying to break into ML infra and a mutual "
            "friend (Sam) said you'd be a great person to chat with. Any chance you "
            "have 20 mins next week? Tue or Thu afternoon both work for me.\n\nThanks!\nPriya"
        ),
        received_at="2025-04-24T09:14:00Z",
    ),
    Email(
        id="msg_002",
        thread_id="thread_002",
        sender="Stripe",
        sender_email="receipts@stripe.com",
        subject="Your receipt from Acme Inc",
        body="Thanks for your payment of $19.00. View receipt online.",
        received_at="2025-04-24T07:02:00Z",
    ),
    Email(
        id="msg_003",
        thread_id="thread_003",
        sender="The Pragmatic Engineer",
        sender_email="newsletter@pragmaticengineer.com",
        subject="Issue #312: How Big Tech does on-call",
        body="In this week's issue: on-call rotations, incident response...",
        received_at="2025-04-24T06:00:00Z",
    ),
    Email(
        id="msg_004",
        thread_id="thread_004",
        sender="Dr. Alvarez (landlord)",
        sender_email="alvarez.rentals@example.com",
        subject="Lease renewal — please confirm by Friday",
        body=(
            "Hi Rahul,\n\nAttached is the renewal paperwork for the apartment. "
            "Rent will go up 3% next term. Please sign and return by this Friday "
            "so I can keep your unit reserved.\n\nBest,\nDr. Alvarez"
        ),
        received_at="2025-04-24T08:45:00Z",
    ),
    Email(
        id="msg_005",
        thread_id="thread_005",
        sender="LinkedIn Recruiter",
        sender_email="noreply@linkedin.com",
        subject="3 new jobs match your profile",
        body="Senior Engineer at FooCorp, Staff Engineer at BarCo...",
        received_at="2025-04-24T05:30:00Z",
    ),
    Email(
        id="msg_006",
        thread_id="thread_006",
        sender="Google Calendar",
        sender_email="calendar-notification@google.com",
        subject="Invitation: Sync with Eng team @ Mon Apr 28, 10:00 AM",
        body="You have been invited to the following event...",
        received_at="2025-04-23T22:00:00Z",
    ),
]


# ---------------------------------------------------------------------------
# Gmail client (lazy — dry-run never imports Google libraries)
# ---------------------------------------------------------------------------

def _gmail_service() -> Resource:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    creds: Credentials | None = None
    token_path = settings.gmail_token_path
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            creds_path = settings.gmail_credentials_path
            if not creds_path.exists():
                raise FileNotFoundError(
                    f"Gmail credentials not found at {creds_path}. "
                    "See README for setup, or use --dry-run."
                )
            flow = InstalledAppFlow.from_client_secrets_file(str(creds_path), SCOPES)
            creds = flow.run_local_server(port=0)
        token_path.write_text(creds.to_json())

    return build("gmail", "v1", credentials=creds, cache_discovery=False)


def _decode_body(payload: dict) -> str:
    if payload.get("mimeType") == "text/plain":
        data = payload.get("body", {}).get("data")
        if data:
            return base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")
    for part in payload.get("parts", []) or []:
        body = _decode_body(part)
        if body:
            return body
    return ""


def _parse_message(msg: dict) -> Email:
    headers = {h["name"].lower(): h["value"] for h in msg["payload"].get("headers", [])}
    from_raw = headers.get("from", "")
    if "<" in from_raw:
        sender_name, sender_email = from_raw.split("<", 1)
        sender_name = sender_name.strip().strip('"')
        sender_email = sender_email.rstrip(">").strip()
    else:
        sender_name = from_raw
        sender_email = from_raw

    body = _decode_body(msg["payload"]) or msg.get("snippet", "")

    return Email(
        id=msg["id"],
        thread_id=msg.get("threadId"),
        sender=sender_name or sender_email,
        sender_email=sender_email,
        subject=headers.get("subject", "(no subject)"),
        body=body[:8000],
        received_at=headers.get("date", ""),
    )


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

def fetch_emails_node(state: InboxState) -> dict:
    """Fetches unread emails (or fixtures in dry-run)."""
    writer = get_stream_writer()
    if state.get("dry_run"):
        emails = list(SAMPLE_EMAILS)[: settings.max_emails_per_run]
        writer({"step": "fetch_emails", "msg": f"loaded {len(emails)} fixtures (dry-run)"})
        return {"emails": emails, "queue": list(emails)}

    writer({"step": "fetch_emails", "msg": "calling Gmail API..."})
    service = _gmail_service()
    listing = (
        service.users()
        .messages()
        .list(userId="me", q="is:unread in:inbox", maxResults=settings.max_emails_per_run)
        .execute()
    )

    emails: list[Email] = []
    for stub in listing.get("messages", []):
        full = service.users().messages().get(userId="me", id=stub["id"], format="full").execute()
        emails.append(_parse_message(full))

    writer({"step": "fetch_emails", "msg": f"fetched {len(emails)} unread emails"})
    return {"emails": emails, "queue": list(emails)}


def build_draft_queue_node(state: InboxState) -> dict:
    """Run after all parallel_triagers finish — seeds the serial draft queue.

    In the parallel path, triagers classified all emails simultaneously.
    Now we need to feed needs_reply emails into the pop_next → drafter loop.
    This node collects them into `queue` so pop_next can iterate normally.
    """
    writer = get_stream_writer()
    emails = state.get("emails", [])
    classifications = state.get("classifications", {})
    needs_reply = [e for e in emails if classifications.get(e.id, None) and
                   classifications[e.id].category == "needs_reply"]
    writer({
        "step": "build_draft_queue",
        "msg": f"{len(needs_reply)}/{len(emails)} emails need a reply — queuing for drafter",
    })
    return {"queue": needs_reply, "current": None}


def pop_next_email_node(state: InboxState) -> dict:
    """Pops the next email off the queue and sets it as `current`."""
    writer = get_stream_writer()
    queue = list(state.get("queue", []))
    if not queue:
        writer({"step": "pop_next", "msg": "queue empty → routing to summary"})
        return {
            "current": None,
            "queue": [],
            "triager_hint": None,
            "retriage_loop": 0,
            "critic_wants": None,
            "critic_note": None,
            "drafter_attempts": 0,
        }
    nxt = queue.pop(0)
    writer({
        "step": "pop_next",
        "msg": f"next: {nxt.sender} — {nxt.subject[:50]} ({len(queue)} left)",
    })
    return {
        "current": nxt,
        "queue": queue,
        "triager_hint": None,
        "retriage_loop": 0,
        "critic_wants": None,
        "critic_note": None,
        "drafter_attempts": 0,
    }


def human_review_node(state: InboxState):
    """Pause for the human to approve / edit / skip the draft.

    Calls `interrupt(...)` which suspends the graph at this point. The payload
    is what the consumer sees as `__interrupt__` in the streamed output. To
    resume, the caller passes `Command(resume=<decision>)` and execution
    continues from this exact line — `decision` will be that resume value.

    Decision shape (caller's contract):
        {"action": "approve"}                              → save_draft runs
        {"action": "edit", "body": "<new draft text>"}     → overwrite, then save
        {"action": "skip"}                                 → bypass save_draft
    """
    writer = get_stream_writer()
    current = state.get("current")
    if current is None:
        return {}
    draft = state.get("drafts", {}).get(current.id)
    if draft is None:
        return {}

    writer({"step": "human_review", "msg": f"awaiting approval for: {current.subject[:50]}"})
    decision = interrupt(
        {
            "email": {
                "id": current.id,
                "from": f"{current.sender} <{current.sender_email}>",
                "subject": current.subject,
            },
            "draft_body": draft.body,
        }
    )

    # ---- Resumed: caller passed Command(resume=decision) ----
    if not isinstance(decision, dict):
        decision = {"action": "approve"}  # be lenient

    action = decision.get("action", "approve")
    writer({"step": "human_review", "msg": f"resumed with action={action!r}"})

    if action == "skip":
        # Don't save skipped drafts — that would teach the agent its mistakes.
        return Command(goto="pop_next")

    # Approve or edit → also push to long-term voice store as a future few-shot.
    final_body = decision["body"] if action == "edit" and decision.get("body") else draft.body
    classification = state.get("classifications", {}).get(current.id)
    intent = classification.intent if classification and classification.intent else ""
    summary = classification.summary if classification else ""
    context = f"{current.subject} | {intent or summary}"

    try:
        store = get_store()
    except (RuntimeError, LookupError):
        store = None
    if store is not None:
        store.put(
            VOICE_NAMESPACE,
            key=uuid.uuid4().hex,
            value={"text": context, "reply": final_body},
        )
        writer({"step": "human_review", "msg": "saved approved draft → voice store"})

    if action == "edit":
        return {"drafts": {current.id: Draft(body=final_body)}}
    return {}


def save_draft_node(state: InboxState) -> dict:
    """Creates a Gmail draft as a reply to `current`. No-op in dry-run."""
    current = state.get("current")
    if current is None:
        return {}
    draft = state.get("drafts", {}).get(current.id)
    if draft is None:
        return {}

    if state.get("dry_run"):
        return {"saved_draft_ids": {current.id: "DRY_RUN"}}

    service = _gmail_service()
    message = EmailMessage()
    message.set_content(draft.body)
    message["To"] = current.sender_email
    message["Subject"] = (
        current.subject if current.subject.lower().startswith("re:") else f"Re: {current.subject}"
    )

    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
    body: dict = {"message": {"raw": raw}}
    if current.thread_id:
        body["message"]["threadId"] = current.thread_id

    created = service.users().drafts().create(userId="me", body=body).execute()
    return {"saved_draft_ids": {current.id: created["id"]}}


def summary_node(state: InboxState) -> dict:
    """Final node — prints the morning brief and any drafts."""
    emails = state.get("emails", [])
    classifications = state.get("classifications", {})
    drafts = state.get("drafts", {})
    saved = state.get("saved_draft_ids", {})

    counts = Counter(c.category for c in classifications.values())
    summary_text = "\n".join(
        [
            f"Processed {len(emails)} email(s).",
            "",
            "By category:",
            *[f"  - {cat}: {n}" for cat, n in counts.most_common()],
            "",
            f"Drafts created: {len(saved)}",
        ]
    )

    table = Table(title="Inbox triage", show_lines=False)
    table.add_column("From", overflow="fold", max_width=24)
    table.add_column("Subject", overflow="fold", max_width=40)
    table.add_column("Category")
    table.add_column("Urgency")
    table.add_column("Draft?", justify="center")

    for email in emails:
        c = classifications.get(email.id)
        cat = c.category if c else "?"
        urg = c.urgency if c else "?"
        has_draft = "yes" if email.id in drafts else ""
        table.add_row(email.sender, email.subject, cat, urg, has_draft)

    console.print(Panel(summary_text, title="Morning brief", expand=False))
    console.print(table)

    emails_by_id = {e.id: e for e in emails}
    for email_id, draft in drafts.items():
        email = emails_by_id.get(email_id)
        if email is None:
            continue
        console.rule(f"Draft → {email.sender} ({email.subject})")
        console.print(draft.body)
        if draft.notes:
            console.print(f"\n[dim]notes: {draft.notes}[/dim]")

    return {"summary": summary_text}
