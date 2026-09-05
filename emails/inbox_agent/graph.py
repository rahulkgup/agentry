"""LangGraph wiring for the two-agent inbox triager.

Flow:
    fetch_emails -> pop_next -> [queue empty?] -> summary -> END
                                       |
                                       v
                                   triager
                                       |
                              [needs_reply & allowed?]
                              /                       \
                          drafter                   pop_next (next email)
                             |
                          [critic / retriage]  (see build_graph)
                             |
                          save_draft
                             |
                          pop_next (next email)
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from typing import Iterator

from langchain_openai import OpenAIEmbeddings
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.store.sqlite import SqliteStore
from langgraph.types import RetryPolicy

from .agents import (
    critic_node,
    drafter_node,
    fan_out_node,
    make_calendar_agent_node,
    manual_react_drafter_node,
    parallel_triager_node,
    supervisor_node,
    tool_aware_drafter_node,
    triager_node,
)
from .config import settings
from tracing import traced
from .fakes import fake_critic_node, fake_drafter_node, fake_triager_node
from .state import InboxState
from .tools import (
    build_draft_queue_node,
    fetch_emails_node,
    human_review_node,
    pop_next_email_node,
    save_draft_node,
    summary_node,
)

DRAFTERS = {
    "simple": drafter_node,
    "react": tool_aware_drafter_node,
    "manual-react": manual_react_drafter_node,
}

CHECKPOINT_DB = "./checkpoints.sqlite"
STORE_DB = "./store.sqlite"
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIMS = 1536


def _route_after_pop(state: InboxState) -> str:
    return "summary" if state.get("current") is None else "triager"


def _route_after_triage(state: InboxState) -> str:
    current = state.get("current")
    if current is None:
        return "pop_next"

    classification = state.get("classifications", {}).get(current.id)
    if classification is None or classification.category != "needs_reply":
        return "pop_next"

    allow = settings.allowlist
    if allow and current.sender_email.lower() not in allow:
        return "pop_next"

    return "drafter"


def _make_route_after_drafter(*, use_critic: bool, parallel: bool, skip_review: bool):
    def route_after_drafter(state: InboxState) -> str:
        if not use_critic:
            return "save_draft" if skip_review else "human_review"
        if parallel:
            return "critic"
        current = state.get("current")
        d = (state.get("drafts", {}) or {}).get(current.id) if current else None
        if (
            d is not None
            and bool(getattr(d, "retriage", False))
            and state.get("retriage_loop", 0) <= 1
        ):
            return "triager"
        return "critic"

    return route_after_drafter


def _make_route_after_critic(*, skip_review: bool):
    def route_after_critic(state: InboxState) -> str:
        w = state.get("critic_wants") or "ok"
        if w == "revise" and state.get("drafter_attempts", 0) < 2:
            return "drafter"
        return "save_draft" if skip_review else "human_review"

    return route_after_critic


def _make_route_drafter_parallel(*, use_critic: bool, skip_review: bool):
    def route_drafter_parallel(state: InboxState) -> str:
        if not use_critic:
            return "save_draft" if skip_review else "human_review"
        return "critic"

    return route_drafter_parallel


def _critic_branch_map(skip_review: bool) -> dict[str, str]:
    m = {"drafter": "drafter", "save_draft": "save_draft"}
    if not skip_review:
        m["human_review"] = "human_review"
    return m


def _drafter_direct_branch_map(skip_review: bool) -> dict[str, str]:
    m = {"save_draft": "save_draft"}
    if not skip_review:
        m["human_review"] = "human_review"
    return m


def build_graph(
    checkpointer=None,
    store=None,
    drafter: str = "simple",
    skip_review: bool = True,
    use_supervisor: bool = False,
    parallel: bool = False,
    use_critic: bool = True,
):
    """`use_critic` is forced off when FAKE_LLM=1 in the environment."""
    g = StateGraph(InboxState)
    llm_retry = RetryPolicy(
        max_attempts=3, initial_interval=0.5, backoff_factor=2.0, jitter=True
    )

    if settings.fake_llm:
        use_critic = False

    drafter_fn = DRAFTERS.get(drafter, drafter_node)
    if settings.fake_llm:
        drafter_fn = fake_drafter_node
    tri = fake_triager_node if settings.fake_llm else triager_node
    crit = fake_critic_node if settings.fake_llm else critic_node

    if parallel:
        g.add_node("fetch_emails", traced("fetch_emails", fetch_emails_node))
        g.add_node("parallel_triager", traced("parallel_triager", parallel_triager_node))
        g.add_node("build_draft_queue", traced("build_draft_queue", build_draft_queue_node))
        g.add_node("pop_next", traced("pop_next", pop_next_email_node))
        g.add_node("drafter", traced("drafter", drafter_fn), retry=llm_retry)
        g.add_node("save_draft", traced("save_draft", save_draft_node))
        g.add_node("summary", traced("summary", summary_node))
        g.add_edge(START, "fetch_emails")
        g.add_conditional_edges("fetch_emails", fan_out_node, ["parallel_triager"])
        g.add_edge("parallel_triager", "build_draft_queue")
        g.add_edge("build_draft_queue", "pop_next")
        g.add_conditional_edges(
            "pop_next",
            _route_after_pop,
            {"triager": "drafter", "summary": "summary"},
        )
        if not skip_review:
            g.add_node("human_review", traced("human_review", human_review_node))
            g.add_edge("human_review", "save_draft")
        if use_critic:
            g.add_node("critic", traced("critic", crit), retry=llm_retry)
            g.add_conditional_edges(
                "drafter",
                _make_route_drafter_parallel(
                    use_critic=use_critic, skip_review=skip_review
                ),
                {"critic": "critic"},
            )
            g.add_conditional_edges(
                "critic",
                _make_route_after_critic(skip_review=skip_review),
                _critic_branch_map(skip_review),
            )
        else:
            g.add_conditional_edges(
                "drafter",
                _make_route_drafter_parallel(
                    use_critic=False, skip_review=skip_review
                ),
                _drafter_direct_branch_map(skip_review),
            )
        g.add_edge("save_draft", "pop_next")
        g.add_edge("summary", END)
        return g.compile(checkpointer=checkpointer if checkpointer is not None else InMemorySaver(), store=store)

    g.add_node("fetch_emails", traced("fetch_emails", fetch_emails_node))
    g.add_node("pop_next", traced("pop_next", pop_next_email_node))
    g.add_node("triager", traced("triager", tri), retry=llm_retry)

    if use_supervisor:
        g.add_node("supervisor", traced("supervisor", supervisor_node), retry=llm_retry)
        g.add_node("drafter", traced("drafter", drafter_fn), retry=llm_retry)
        g.add_node("calendar_agent", traced("calendar_agent", make_calendar_agent_node(skip_review=skip_review)))
        g.add_edge("triager", "supervisor")
    else:
        g.add_node("drafter", traced("drafter", drafter_fn), retry=llm_retry)
        g.add_conditional_edges(
            "triager",
            _route_after_triage,
            {"drafter": "drafter", "pop_next": "pop_next"},
        )

    g.add_node("save_draft", traced("save_draft", save_draft_node))
    g.add_node("summary", traced("summary", summary_node))
    if not skip_review:
        g.add_node("human_review", traced("human_review", human_review_node))
        g.add_edge("human_review", "save_draft")
    if use_critic:
        g.add_node("critic", traced("critic", crit), retry=llm_retry)

    g.add_edge(START, "fetch_emails")
    g.add_edge("fetch_emails", "pop_next")
    g.add_conditional_edges(
        "pop_next",
        _route_after_pop,
        {"triager": "triager", "summary": "summary"},
    )

    if use_supervisor and use_critic:
        g.add_conditional_edges(
            "drafter",
            _make_route_after_drafter(
                use_critic=use_critic, parallel=False, skip_review=skip_review
            ),
            {"triager": "triager", "critic": "critic"},
        )
        g.add_conditional_edges(
            "critic",
            _make_route_after_critic(skip_review=skip_review),
            _critic_branch_map(skip_review),
        )
    elif use_supervisor and not use_critic:
        g.add_conditional_edges(
            "drafter",
            _make_route_after_drafter(
                use_critic=False, parallel=False, skip_review=skip_review
            ),
            _drafter_direct_branch_map(skip_review),
        )
    elif not use_supervisor and use_critic:
        g.add_conditional_edges(
            "drafter",
            _make_route_after_drafter(
                use_critic=use_critic, parallel=False, skip_review=skip_review
            ),
            {"triager": "triager", "critic": "critic"},
        )
        g.add_conditional_edges(
            "critic",
            _make_route_after_critic(skip_review=skip_review),
            _critic_branch_map(skip_review),
        )
    else:
        g.add_conditional_edges(
            "drafter",
            _make_route_after_drafter(
                use_critic=False, parallel=False, skip_review=skip_review
            ),
            _drafter_direct_branch_map(skip_review),
        )

    g.add_edge("save_draft", "pop_next")
    g.add_edge("summary", END)

    return g.compile(checkpointer=checkpointer if checkpointer is not None else InMemorySaver(), store=store)


def _store_index() -> dict:
    return {
        "embed": OpenAIEmbeddings(
            model=EMBED_MODEL, api_key=settings.openai_api_key
        ),
        "dims": EMBED_DIMS,
        "fields": ["text"],
    }


@contextmanager
def open_graph(
    thread_id: str,
    checkpoint_db: str = CHECKPOINT_DB,
    store_db: str = STORE_DB,
    with_store: bool = True,
    drafter: str = "simple",
    skip_review: bool = True,
    use_supervisor: bool = False,
    parallel: bool = False,
) -> Iterator[tuple]:
    with ExitStack() as stack:
        cp = stack.enter_context(SqliteSaver.from_conn_string(checkpoint_db))
        store = None
        if with_store:
            store = stack.enter_context(
                SqliteStore.from_conn_string(store_db, index=_store_index())
            )
            store.setup()
        graph = build_graph(
            checkpointer=cp,
            store=store,
            drafter=drafter,
            skip_review=skip_review,
            use_supervisor=use_supervisor,
            parallel=parallel,
        )
        yield graph, {"configurable": {"thread_id": thread_id}}
