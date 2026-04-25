"""Studio entrypoint — exposes compiled graphs to `langgraph dev`.

IMPORTANT: Do NOT pass a checkpointer here. LangGraph Platform/Studio injects
its own checkpointer automatically. Passing InMemorySaver or SqliteSaver causes
a startup error. The platform handles persistence; your code handles logic.

Each variable is picked up by langgraph.json and shown in the Studio sidebar.
"""

from __future__ import annotations

from emails.inbox_agent.graph import build_graph

# Standard (sequential, simple drafter) — the baseline
inbox_graph = build_graph(checkpointer=False)

# With supervisor routing
inbox_supervisor = build_graph(checkpointer=False, use_supervisor=True)

# With parallel triage via Send
inbox_parallel = build_graph(checkpointer=False, parallel=True)

# With interactive HITL review
inbox_review = build_graph(checkpointer=False, skip_review=False)
