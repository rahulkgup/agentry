"""The two LLM-backed agents.

Agent A (triager) classifies an email; Agent B (drafter) writes a reply.
Both produce structured output via Pydantic models defined in `state.py`.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.config import get_store, get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, create_react_agent, tools_condition
from langgraph.types import Command, Send

from .config import settings
from .mcp_tools import get_react_tools
from .prompts import drafter_system, triager_system, voice_examples_block, voice_file_examples_block
from .sender_memory import cached_classification_if_ready, remember
from .state import CriticVerdict, Classification, Draft, InboxState

VOICE_NAMESPACE = ("voice", "drafts")
VOICE_TOP_K = 2


def _llm(temperature: float) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.llm_model,
        temperature=temperature,
        api_key=settings.openai_api_key,
    )


def triager_node(state: InboxState) -> dict:
    """Agent A — classifies the current email."""
    writer = get_stream_writer()
    current = state.get("current")
    if current is None:
        return {}

    sender = current.sender_email
    hint = (state.get("triager_hint") or "").strip()
    cached = cached_classification_if_ready(sender)
    if cached is not None and not hint:
        writer({
            "step": "triager",
            "msg": f"cache hit (3+ same labels) → {cached.category}",
        })
        return {"classifications": {current.id: cached}, "triager_hint": None}

    writer({"step": "triager", "msg": f"classifying via {settings.llm_model}..."})
    llm = _llm(temperature=0).with_structured_output(Classification)
    body_extra = f"\n\n(Note: drafter asked to re-triage: {hint})" if hint else ""
    messages = [
        SystemMessage(content=triager_system()),
        HumanMessage(
            content=f"{current.header()}\n\n--- body ---\n{current.body}\n--- end body ---{body_extra}"
        ),
    ]
    classification: Classification = llm.invoke(messages)  # type: ignore[assignment]
    remember(sender, classification.category)
    writer({
        "step": "triager",
        "msg": f"→ {classification.category} ({classification.urgency})",
    })
    return {"classifications": {current.id: classification}, "triager_hint": None}


def drafter_node(state: InboxState) -> dict:
    """Agent B — drafts a reply to the current email.

    Pulls top-K most-similar past replies from the long-term Store and
    injects them as few-shot examples. The Store is bound at compile time
    via `g.compile(store=...)`; here we access it through `get_store()`,
    which returns None if no store is bound (e.g. during tests).
    """
    writer = get_stream_writer()
    current = state.get("current")
    if current is None:
        return {}

    attempts = state.get("drafter_attempts", 0) + 1
    out_base: dict = {"drafter_attempts": attempts, "critic_wants": None, "critic_note": None}

    classification = state.get("classifications", {}).get(current.id)
    intent = classification.intent if classification and classification.intent else ""
    intent_line = f"\n\nWhat the sender wants: {intent}" if intent else ""
    cnote = (state.get("critic_note") or "").strip()
    revision_block = f"\n\nCritic asked for a revision: {cnote}\n" if cnote else ""

    # Long-term memory lookup — invisible in `updates` mode, exposed via custom events.
    examples_block = ""
    try:
        store = get_store()
    except (RuntimeError, LookupError):
        store = None
    if store is not None:
        query = f"{current.subject} | {intent or (classification.summary if classification else '')}"
        writer({"step": "drafter", "msg": f"voice search: {query[:60]}..."})
        items = store.search(VOICE_NAMESPACE, query=query, limit=VOICE_TOP_K)
        scores = [f"{it.score:.2f}" for it in items if it.score is not None]
        writer({
            "step": "drafter",
            "msg": f"got {len(items)} voice matches (scores: {scores})",
        })
        examples_block = voice_examples_block(items)
    else:
        writer({"step": "drafter", "msg": "no store bound — drafting without examples"})

    examples_block = voice_file_examples_block() + examples_block

    voice_hint = (
        "\nSet retriage=True only if the triager's category is clearly wrong. "
        "At most one re-triage per email is allowed by the graph."
    )
    writer({"step": "drafter", "msg": f"calling {settings.llm_model} for draft..."})
    llm = _llm(temperature=0.4).with_structured_output(Draft)
    messages = [
        SystemMessage(content=drafter_system() + voice_hint),
        HumanMessage(
            content=(
                f"{examples_block}{revision_block}"
                f"Draft a reply to this email.{intent_line}\n\n"
                f"{current.header()}\n\n"
                f"--- their message ---\n{current.body}\n--- end ---"
            )
        ),
    ]
    draft: Draft = llm.invoke(messages)  # type: ignore[assignment]
    writer({"step": "drafter", "msg": f"draft ready ({len(draft.body)} chars)"})
    if draft.retriage:
        loop = state.get("retriage_loop", 0) + 1
        out_base.update(
            {
                "drafts": {current.id: draft},
                "retriage_loop": loop,
                "triager_hint": (draft.retriage_reason or "Please re-classify this email.").strip(),
            }
        )
    else:
        out_base["drafts"] = {current.id: draft}
        out_base["retriage_loop"] = state.get("retriage_loop", 0)
    return out_base


# ---------------------------------------------------------------------------
# Agent B v2 — ReAct drafter that can call tools before composing the reply
# ---------------------------------------------------------------------------

REACT_DRAFTER_SUFFIX = """\

You have access to tools that help you draft a better reply:
- `lookup_contact_history`: ALWAYS call this first to learn about the sender.
- `search_past_drafts`: Call this with a description of the incoming email
  (subject + intent) to find similar past replies you can match in style.

After using any tools you need, write ONLY the final reply text — no preamble,
no quotes, no JSON. Just the reply that should be sent.
"""


def tool_aware_drafter_node(state: InboxState) -> dict:
    """Agent B v2 — a ReAct loop that calls tools before drafting.

    Wraps `create_react_agent` as a sub-graph executed inside this node.
    The inner agent loops `model → tool → model → ...` until the model
    returns an answer without further tool calls. We extract the final
    message and box it into our `Draft` shape so the rest of the parent
    graph (save_draft, human_review, etc.) is unchanged.
    """
    writer = get_stream_writer()
    current = state.get("current")
    if current is None:
        return {}

    classification = state.get("classifications", {}).get(current.id)
    intent = classification.intent if classification and classification.intent else ""
    summary = classification.summary if classification else ""

    user_msg = (
        f"From: {current.sender} <{current.sender_email}>\n"
        f"Subject: {current.subject}\n"
        f"Intent (from triager): {intent or summary}\n\n"
        f"--- their message ---\n{current.body}\n--- end ---"
    )

    tools = get_react_tools()
    writer({
        "step": "react_drafter",
        "msg": f"starting ReAct loop with {len(tools)} tools available",
    })

    model = _llm(temperature=0.4)
    agent = create_react_agent(
        model,
        tools=tools,
        prompt=drafter_system() + REACT_DRAFTER_SUFFIX,
    )

    result = agent.invoke({"messages": [HumanMessage(content=user_msg)]})
    messages = result["messages"]

    # Tally how many tool calls happened so we can narrate.
    tool_call_count = sum(
        len(getattr(m, "tool_calls", []) or []) for m in messages
    )
    writer({
        "step": "react_drafter",
        "msg": f"loop ended after {tool_call_count} tool calls, {len(messages)} total msgs",
    })

    final = messages[-1].content if messages else ""
    return {"drafts": {current.id: Draft(body=final)}}


# ---------------------------------------------------------------------------
# Agent B v3 — same ReAct loop, but assembled by hand (educational)
# ---------------------------------------------------------------------------

class _ReactState(TypedDict):
    """Inner state of the hand-built ReAct agent — just a message history."""

    messages: Annotated[list[BaseMessage], add_messages]


def _build_manual_react_agent(tools: list | None = None):
    """Reproduce `create_react_agent` from primitives (tools from MCP + store)."""
    t = tools if tools is not None else get_react_tools()
    model = _llm(temperature=0.4).bind_tools(t)

    def agent_node(state: _ReactState) -> dict:
        return {"messages": [model.invoke(state["messages"])]}

    g = StateGraph(_ReactState)
    g.add_node("agent", agent_node)
    g.add_node("tools", ToolNode(t))
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")
    return g.compile()


def manual_react_drafter_node(state: InboxState) -> dict:
    """Same as `tool_aware_drafter_node` but uses the hand-built ReAct loop.

    Functionally equivalent to `create_react_agent` — same node names, same
    edges, same termination rule. The point is to see all the seams so you
    know where to customize (e.g., insert a "did the model emit dangerous
    tool_calls?" check between agent and tools).
    """
    writer = get_stream_writer()
    current = state.get("current")
    if current is None:
        return {}

    classification = state.get("classifications", {}).get(current.id)
    intent = classification.intent if classification and classification.intent else ""
    summary = classification.summary if classification else ""

    user_msg = (
        f"From: {current.sender} <{current.sender_email}>\n"
        f"Subject: {current.subject}\n"
        f"Intent (from triager): {intent or summary}\n\n"
        f"--- their message ---\n{current.body}\n--- end ---"
    )

    writer({
        "step": "manual_react",
        "msg": "starting hand-built ReAct loop (bind_tools + ToolNode + tools_condition)",
    })

    agent = _build_manual_react_agent(get_react_tools())
    init_messages: list[BaseMessage] = [
        SystemMessage(content=drafter_system() + REACT_DRAFTER_SUFFIX),
        HumanMessage(content=user_msg),
    ]
    result = agent.invoke({"messages": init_messages})
    messages = result["messages"]

    tool_call_count = sum(
        len(getattr(m, "tool_calls", []) or []) for m in messages
    )
    writer({
        "step": "manual_react",
        "msg": f"loop ended after {tool_call_count} tool calls, {len(messages)} total msgs",
    })

    final = messages[-1].content if messages else ""
    return {"drafts": {current.id: Draft(body=final)}}


# ---------------------------------------------------------------------------
# Critic (self-critique before save)
# ---------------------------------------------------------------------------


def critic_node(state: InboxState) -> dict:
    """Re-reads the draft; asks the model to approve or request one revision."""
    writer = get_stream_writer()
    current = state.get("current")
    if current is None:
        return {"critic_wants": "ok", "critic_note": ""}
    draft = state.get("drafts", {}).get(current.id)
    if draft is None:
        return {"critic_wants": "ok", "critic_note": ""}

    writer({"step": "critic", "msg": f"reviewing draft ({len(draft.body)} chars)..."})
    crit_sys = (
        f"You are a strict self-critique for {settings.user_name}'s outgoing email replies. "
        "Decide: ok to send, or needs_revision. If revision, be specific. "
        "Be conservative: prefer ok if the draft answers the ask and sounds like the user."
    )
    llm = _llm(temperature=0).with_structured_output(CriticVerdict)
    v: CriticVerdict = llm.invoke(  # type: ignore[assignment]
        [
            SystemMessage(content=crit_sys),
            HumanMessage(
                content=(
                    f"Subject: {current.subject}\n"
                    f"--- draft ---\n{draft.body}\n--- end ---\n"
                    f"Is this good to save, or should the drafter revise?"
                )
            ),
        ]
    )
    w = "revise" if v.wants == "revise" else "ok"
    writer({"step": "critic", "msg": f"→ {w}"})
    return {"critic_wants": w, "critic_note": v.note or ""}


# ---------------------------------------------------------------------------
# Phase 6 — Supervisor + peer-handoff patterns
# ---------------------------------------------------------------------------

_SUPERVISOR_PROMPT = """\
You are a routing supervisor for an email inbox agent.

You have already seen the triager's classification of the current email.
Decide which worker should handle this email next:

- "drafter"        : write a personal reply (use when category=needs_reply)
- "calendar_agent" : handle a meeting invite or calendar event
- "pop_next"       : skip — no action needed (fyi, newsletter, promo, etc.)

You MUST respond with ONLY one of the three strings above. No explanation.
"""


def supervisor_node(state: InboxState) -> Command:
    """Supervisor — LLM that routes each email to the right worker.

    Returns `Command(goto=worker)` to dynamically hand off control.
    This replaces the deterministic `_route_after_triage` conditional edge
    with an LLM that can reason about edge cases (e.g., an automated email
    that still warrants a reply, or a newsletter with a critical update).

    Key difference from conditional edges:
    - Conditional edge: router function defined statically at graph-build time
    - Command(goto): routing decision made dynamically at runtime, by the node
    """
    writer = get_stream_writer()
    current = state.get("current")
    if current is None:
        return Command(goto="pop_next")

    classification = state.get("classifications", {}).get(current.id)
    cat = classification.category if classification else "unknown"
    urgency = classification.urgency if classification else "unknown"
    summary = classification.summary if classification else "(no summary)"

    prompt = (
        f"Email from: {current.sender} <{current.sender_email}>\n"
        f"Subject: {current.subject}\n"
        f"Triager classification: category={cat}, urgency={urgency}\n"
        f"Summary: {summary}"
    )

    writer({"step": "supervisor", "msg": f"routing: {cat} — {current.subject[:40]}"})
    llm = _llm(temperature=0)
    response = llm.invoke([
        SystemMessage(content=_SUPERVISOR_PROMPT),
        HumanMessage(content=prompt),
    ])
    decision = response.content.strip().lower()

    valid = {"drafter", "calendar_agent", "pop_next"}
    if decision not in valid:
        for v in valid:
            if v in decision:
                decision = v
                break
        else:
            decision = "pop_next"

    writer({"step": "supervisor", "msg": f"→ {decision}"})
    return Command(goto=decision)


def make_calendar_agent_node(skip_review: bool):
    """Build calendar worker — all exits use `Command(goto=...)` (no static graph edge)."""

    def calendar_agent_node(state: InboxState) -> dict | Command:
        """Specialized worker for calendar invites; routing is 100% via Command."""
        writer = get_stream_writer()
        current = state.get("current")
        if current is None:
            return {}

        after_inline = "save_draft" if skip_review else "human_review"

        writer({"step": "calendar_agent", "msg": f"analyzing: {current.subject[:50]}"})

        decision_prompt = (
            "Does this calendar email require a personal RSVP reply from the recipient "
            "(i.e., accept/decline/tentative response expected)? "
            "Answer ONLY 'yes' or 'no'.\n\n"
            f"Subject: {current.subject}\n"
            f"From: {current.sender}\n"
            f"Body: {current.body[:300]}"
        )
        llm = _llm(temperature=0)
        decision = llm.invoke([HumanMessage(content=decision_prompt)]).content.strip().lower()
        needs_rsvp = decision.startswith("yes")

        writer({
            "step": "calendar_agent",
            "msg": f"needs_rsvp={needs_rsvp} → "
            f"{'drafter' if needs_rsvp else after_inline}",
        })

        if needs_rsvp:
            return Command(goto="drafter")

        draft_body = f"Noted: {current.subject}. Added to calendar."
        return Command(
            goto=after_inline,
            update={"drafts": {current.id: Draft(body=draft_body)}},
        )

    return calendar_agent_node


# ---------------------------------------------------------------------------
# Phase 7 — Send / map-reduce fan-out
# ---------------------------------------------------------------------------

def fan_out_node(state: InboxState) -> list[Send]:
    """Fan-out function used as a conditional edge — returns [Send(...), ...].

    This is the MAP step. Called after fetch_emails, it launches one
    parallel_triager per email. LangGraph runs them all in the same
    super-step. Their outputs merge via the `classifications` reducer.

    Used as `g.add_conditional_edges("fetch_emails", fan_out_node, ["parallel_triager"])`.
    """
    writer = get_stream_writer()
    emails = state.get("emails", [])
    writer({"step": "fan_out", "msg": f"fanning out {len(emails)} emails → parallel_triager"})
    return [
        Send("parallel_triager", {"current": email, "dry_run": state.get("dry_run", False)})
        for email in emails
    ]


def parallel_triager_node(state: dict) -> dict:
    """Worker node — classifies ONE email (its payload has only `current`).

    Runs concurrently alongside all other parallel_triager instances.
    Returns ONLY `classifications` so the reducer can merge safely.
    The parent InboxState already has `emails` from fetch_emails.
    """
    writer = get_stream_writer()
    current = state.get("current")
    if current is None:
        return {}

    cached = cached_classification_if_ready(current.sender_email)
    if cached is not None:
        writer({
            "step": "parallel_triager",
            "msg": f"cache hit → {cached.category} ({current.subject[:25]})",
        })
        return {"classifications": {current.id: cached}}

    writer({"step": "parallel_triager", "msg": f"classifying: {current.subject[:40]}"})
    llm = _llm(temperature=0).with_structured_output(Classification)
    messages = [
        SystemMessage(content=triager_system()),
        HumanMessage(
            content=f"{current.header()}\n\n--- body ---\n{current.body}\n--- end body ---"
        ),
    ]
    classification: Classification = llm.invoke(messages)  # type: ignore[assignment]
    remember(current.sender_email, classification.category)
    writer({
        "step": "parallel_triager",
        "msg": f"{current.subject[:30]} → {classification.category}",
    })
    return {"classifications": {current.id: classification}}
