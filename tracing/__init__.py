"""OpenTelemetry wiring for the agentry graphs.

Off by default: the OTel API returns a no-op tracer until a TracerProvider is
installed, so importing this module costs nothing when tracing isn't configured.
Set OTEL_EXPORTER_OTLP_ENDPOINT to turn it on (e.g. a local Jaeger or Langfuse
OTLP endpoint) — nothing else changes.

Semantic conventions: LLM call attributes use the OTel GenAI namespace
(`gen_ai.*`) where it applies. Gate decisions have no GenAI equivalent yet, so
they use a project-local `agent.*` namespace instead. Which attribute a node
emits is spelled out in `_GATE_ATTRIBUTES` below rather than guessed generically,
so the mapping stays honest about what each node actually decides.
"""

from __future__ import annotations

import functools
import os
from typing import Any, Callable

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

SERVICE_NAME = "agentry"
_TRACER = trace.get_tracer(SERVICE_NAME)


def configure_tracing() -> None:
    """Install a real TracerProvider if OTEL_EXPORTER_OTLP_ENDPOINT is set.

    Call once, at process start (CLI entrypoint). Safe to call when the env
    var is unset: it's a no-op, and get_tracer() keeps returning the OTel API's
    built-in no-op tracer.
    """
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint:
        return

    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter,
    )
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider(
        resource=Resource.create({"service.name": SERVICE_NAME})
    )
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    trace.set_tracer_provider(provider)

    global _TRACER
    _TRACER = trace.get_tracer(SERVICE_NAME)


# Node name -> function(state_before, result) -> dict[str, Any] | None.
# Each entry documents what that node's span attribute *means*, since the
# three gates below decide different things and shouldn't share one key.
def _current_id(state: dict) -> str | None:
    current = state.get("current")
    return getattr(current, "id", None)


def _triage_attrs(state: dict, result: dict) -> dict[str, Any]:
    cid = _current_id(state)
    classification = (result.get("classifications") or {}).get(cid) if cid else None
    if classification is None:
        return {}
    return {
        "agent.triage.category": classification.category,
        "agent.triage.urgency": classification.urgency,
    }


def _drafter_attrs(state: dict, result: dict) -> dict[str, Any]:
    cid = _current_id(state)
    draft = (result.get("drafts") or {}).get(cid) if cid else None
    attrs: dict[str, Any] = {}
    if draft is not None:
        attrs["agent.retriage"] = bool(getattr(draft, "retriage", False))
    if "drafter_attempts" in result:
        attrs["agent.drafter.attempts"] = result["drafter_attempts"]
    return attrs


def _critic_attrs(state: dict, result: dict) -> dict[str, Any]:
    wants = result.get("critic_wants")
    if wants is None:
        return {}
    return {"agent.critic.verdict": wants}


def _human_review_attrs(state: dict, result: dict) -> dict[str, Any]:
    # human_review_node mutates saved_draft_ids / drafts on approval; its
    # presence in the result is the outcome signal, since the node either
    # returns an update (approved, possibly edited) or leaves state alone.
    return {"agent.review.outcome": "approved" if result else "unchanged"}


_GATE_ATTRIBUTES: dict[str, Callable[[dict, Any], dict[str, Any]]] = {
    "triager": _triage_attrs,
    "drafter": _drafter_attrs,
    "critic": _critic_attrs,
    "human_review": _human_review_attrs,
}


def traced(name: str, fn: Callable) -> Callable:
    """Wrap a graph node so it emits one span per invocation.

    Applied at registration (`g.add_node(name, traced(name, fn))`) rather than
    inside each node body, so instrumentation stays in one place and node code
    never imports tracing.
    """

    @functools.wraps(fn)
    def wrapper(state, *args, **kwargs):
        with _TRACER.start_as_current_span(f"node.{name}") as span:
            try:
                result = fn(state, *args, **kwargs)
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                raise

            attr_fn = _GATE_ATTRIBUTES.get(name)
            if attr_fn is not None and isinstance(result, dict):
                for key, value in attr_fn(state, result).items():
                    span.set_attribute(key, value)
            return result

    return wrapper
