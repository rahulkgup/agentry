# Backlog

Status legend: `[ ]` todo · `[~]` in progress · `[x]` done · `[?]` parked / needs decision

---

## Done (rolled up 2026-04)

- [x] **0** — `calendar_agent` double-summary: static edge removed; all exits use `Command(goto=...)`.
- [x] **1** — Allowlist: documented in repo root `.env.example` (`ALLOWLIST_SENDERS`); code was already in `graph.py`.
- [x] **2** — Self-critic: `critic_node` + conditional edges; `drafter_attempts` cap.
- [x] **3** — Re-triage: `Draft.retriage` / `retriage_loop` / `triager_hint` + edge back to `triager` (once).
- [x] **4** — Few-shot files: `emails/inbox_agent/voice_examples/*.txt` + `voice_file_examples_block()` in `prompts.py`.
- [x] **5** — HITL: already had `human_review` + `review` CLI; optional enhancement: `$EDITOR` (parked).
- [x] **6** — `--fake-llm` + `inbox_agent/fakes.py` + `FAKE_LLM` env.
- [x] **7** — Parallel `Send` triage (was already implemented; this line kept for history).
- [x] **8** — Sender memory: `~/.inbox_agent/sender_memory.sqlite` + `sender_memory.py` (skip triager LLM after 3× same category).
- [x] **9 (learning plan)** — MCP: `mcp_contact_server.py` (stdio) + `mcp_tools.get_react_tools()`; `MCP_OFF=1` uses in-process tools.

---

## Next up

### [ ] Optional: open `$EDITOR` on `review` edit action
**Why:** Faster than pasting in the terminal for long edits.

---

## Later / ideas

### [ ] Daily digest output (instead of just terminal)
**Why:** A morning brief in your inbox / iMessage / Pushover is more actionable than a Rich table.

**How:**
- New `notify_node` after `summary`: formats the brief as markdown.
- Configurable sink: `stdout` (default), `email_to_self`, `pushover`, `imessage`.
- Pushover is easiest (one HTTP call, free for personal use).

---

### [ ] Anthropic / local model support
**Why:** Provider redundancy + cost flexibility (Claude Haiku, Ollama models).

**How:**
- Replace `ChatOpenAI(...)` with a `_get_llm(temperature=...)` factory in a new `llm.py`.
- Switch on `LLM_PROVIDER` env var: `openai` (default) | `anthropic` | `ollama`.
- Both langchain integrations exist; ~30 LOC change.

---

## Done (original list)

- [x] Two-agent skeleton: triager + drafter with conditional routing
- [x] Dry-run mode using bundled fixtures
- [x] Gmail OAuth (compose-only scope, no send capability)
- [x] Sender allowlist for safety
- [x] Rich terminal summary
- [x] Smoke test with stubbed LLMs
- [x] Pass `OPENAI_API_KEY` from settings to `ChatOpenAI` (was relying on env var)
