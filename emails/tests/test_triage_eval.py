"""Basic **eval** pattern: golden dataset + run model + assert metrics.

The "model" here is :func:`~emails.evals.stub_triager.stub_triager_for_eval` (free,
deterministic). The same pattern applies when you swap in ``triager_node`` and pay
for API calls: load examples, assert on structured output or use an LLM judge.

Run: ``uv run pytest emails/tests/test_triage_eval.py -v``
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from emails.evals.stub_triager import stub_triager_for_eval
from emails.inbox_agent.tools import SAMPLE_EMAILS

_GOLDEN = (
    Path(__file__).resolve().parent.parent
    / "evals"
    / "triage_golden.json"
)

_BY_ID = {e.id: e for e in SAMPLE_EMAILS}


def _load_golden() -> list[dict]:
    data = json.loads(_GOLDEN.read_text())
    return data["cases"]


def test_triage_golden_file_matches_stub():
    """Per-email accuracy against triage_golden.json (1 metric: category match)."""
    cases = _load_golden()
    n_ok = 0
    for row in cases:
        eid = row["email_id"]
        want = row["expected_category"]
        email = _BY_ID[eid]
        out = stub_triager_for_eval(
            {
                "current": email,
                "classifications": {},
            }
        )
        c = (out.get("classifications") or {})[eid]
        assert c.category == want, f"{eid}: want {want!r} got {c.category!r}"
        n_ok += 1
    assert n_ok == len(cases)


@pytest.mark.parametrize("row", _load_golden(), ids=lambda r: r["email_id"])
def test_triage_each_email_parametrized(row: dict):
    """Same checks as a table — one test row in the report per email (easier in CI)."""
    email = _BY_ID[row["email_id"]]
    out = stub_triager_for_eval({"current": email, "classifications": {}})
    got = (out.get("classifications") or {})[row["email_id"]].category
    assert got == row["expected_category"]


def test_triage_recall_style_metric():
    """Aggregate score (simple 'accuracy'): fraction of cases correct."""
    correct = 0
    for row in _load_golden():
        email = _BY_ID[row["email_id"]]
        out = stub_triager_for_eval({"current": email, "classifications": {}})
        if (out.get("classifications") or {})[row["email_id"]].category == row["expected_category"]:
            correct += 1
    total = len(_load_golden())
    acc = correct / total
    assert acc == 1.0, f"expected perfect stub accuracy, got {acc:.2%}"
