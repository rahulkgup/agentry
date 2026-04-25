"""Smoke test: ensure the graph compiles and routes correctly with stubbed agents.

We monkeypatch the LLM-backed nodes so this runs without network or API keys.
"""

from __future__ import annotations

import pytest

from emails.evals.stub_triager import stub_triager_for_eval
from emails.inbox_agent import graph as graph_module
from emails.inbox_agent import fakes
from emails.inbox_agent.state import Draft, InboxState


def _fake_drafter(state: InboxState) -> dict:
    current = state.get("current")
    if current is None:
        return {}
    return {"drafts": {current.id: Draft(body=f"Hi {current.sender.split()[0]},\n\nSounds good!\n\nRahul")}}


@pytest.fixture(autouse=True)
def _patch_agents(monkeypatch):
    monkeypatch.setattr(graph_module, "triager_node", stub_triager_for_eval)
    monkeypatch.setattr(graph_module, "drafter_node", _fake_drafter)
    monkeypatch.setattr(graph_module, "critic_node", fakes.fake_critic_node)
    # DRAFTERS dict is built at import time from the original references.
    # Patch it too so build_graph picks up our stubs when drafter="simple".
    monkeypatch.setitem(graph_module.DRAFTERS, "simple", _fake_drafter)


def test_dry_run_processes_all_fixtures():
    # build_graph() compiles with InMemorySaver, so we need a thread_id in config.
    graph = graph_module.build_graph()
    config = {"configurable": {"thread_id": "test-smoke"}}
    state = graph.invoke({"dry_run": True}, config=config)

    assert len(state["emails"]) == 6
    assert len(state["classifications"]) == 6
    assert all(eid in state["saved_draft_ids"] for eid in state["drafts"])
    assert state["summary"]
    needs_reply_ids = {
        eid for eid, c in state["classifications"].items() if c.category == "needs_reply"
    }
    assert needs_reply_ids == set(state["drafts"].keys())
