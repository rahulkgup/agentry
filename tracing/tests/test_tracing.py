"""Tests for tracing.traced: span creation, attribute extraction, exceptions."""

from __future__ import annotations

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

import tracing


@pytest.fixture
def spans(monkeypatch):
    """Install an in-memory exporter and point tracing._TRACER at it."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(tracing, "_TRACER", provider.get_tracer("test"))
    return exporter


def test_traced_names_the_span_after_the_node(spans):
    wrapped = tracing.traced("pop_next", lambda state: {})
    wrapped({})

    (span,) = spans.get_finished_spans()
    assert span.name == "node.pop_next"


def test_triager_span_carries_classification(spans):
    class Email:
        id = "e1"

    class Classification:
        category = "needs_reply"
        urgency = "high"

    state = {"current": Email()}
    result = {"classifications": {"e1": Classification()}}

    wrapped = tracing.traced("triager", lambda s: result)
    wrapped(state)

    (span,) = spans.get_finished_spans()
    assert span.attributes["agent.triage.category"] == "needs_reply"
    assert span.attributes["agent.triage.urgency"] == "high"


def test_critic_span_carries_verdict(spans):
    wrapped = tracing.traced("critic", lambda s: {"critic_wants": "revise"})
    wrapped({})

    (span,) = spans.get_finished_spans()
    assert span.attributes["agent.critic.verdict"] == "revise"


def test_drafter_span_carries_retriage_and_attempts(spans):
    class Email:
        id = "e1"

    class Draft:
        retriage = True

    state = {"current": Email()}
    result = {"drafts": {"e1": Draft()}, "drafter_attempts": 2}

    wrapped = tracing.traced("drafter", lambda s: result)
    wrapped(state)

    (span,) = spans.get_finished_spans()
    assert span.attributes["agent.retriage"] is True
    assert span.attributes["agent.drafter.attempts"] == 2


def test_node_without_gate_attributes_still_gets_a_span(spans):
    wrapped = tracing.traced("fetch_emails", lambda s: {"emails": []})
    wrapped({})

    (span,) = spans.get_finished_spans()
    assert span.name == "node.fetch_emails"
    assert not any(k.startswith("agent.") for k in span.attributes)


def test_exception_is_recorded_and_reraised(spans):
    def boom(state):
        raise ValueError("nope")

    wrapped = tracing.traced("drafter", boom)
    with pytest.raises(ValueError):
        wrapped({})

    (span,) = spans.get_finished_spans()
    assert span.status.status_code == trace.StatusCode.ERROR
    assert any(e.name == "exception" for e in span.events)
