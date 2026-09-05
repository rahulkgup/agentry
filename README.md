# agentry

[![CI](https://github.com/rahulkgup/agentry/actions/workflows/ci.yml/badge.svg)](https://github.com/rahulkgup/agentry/actions/workflows/ci.yml)

Reference implementations of the [LangGraph](https://github.com/langchain-ai/langgraph)
patterns that only start to matter once an agent runs against real systems: explicit
graph state, durable execution, human review gates, safe tool access, and evals that
don't depend on live model calls.

Each subfolder is a self-contained app. They share one `pyproject.toml`, one `.venv`,
and one `.env`.

## What it demonstrates

- Agents built around explicit graph state, typed inputs, and predictable routing
  rather than free-form prompt chains.
- Production controls in the graph itself: retries, fallbacks, approval steps,
  auditability, and constrained tool use.
- Evals and tests that run locally and in CI without calling a live model, using
  injected fakes.
- Integration with real external systems through APIs, tools, and MCP.

## Projects

| Project | CLI | What it does |
|---|---|---|
| [`emails/`](./emails/) | `inbox-agent` | Email triage graph. Reads unread Gmail, classifies each message, and drafts a reply in your voice. It never sends: drafting and sending are separated by a human gate. |

### inbox-agent

The graph lives in [`graph.py`](./emails/inbox_agent/graph.py), with routing and
classification in [`agents.py`](./emails/inbox_agent/agents.py) and tool definitions in
[`tools.py`](./emails/inbox_agent/tools.py). Per-sender context is kept in
[`sender_memory.py`](./emails/inbox_agent/sender_memory.py), so replies to a recurring
correspondent carry what came before.

Tool access runs through both direct definitions and MCP
([`mcp_tools.py`](./emails/inbox_agent/mcp_tools.py)). The test suite covers graph
behavior, MCP wiring, and triage quality, and it runs offline against
[`fakes.py`](./emails/inbox_agent/fakes.py) rather than a live model, so evals stay
cheap and deterministic in CI.

## Setup

```bash
cp .env.example .env
# add your OPENAI_API_KEY and any project-specific keys

uv sync
```

## Running it

```bash
uv run inbox-agent run --dry-run
uv run pytest                      # every project
uv run pytest emails/tests         # one project
```

## Adding a project

1. Create the folder, e.g. `recipes/`, with an empty `__init__.py` and your package
   code under `recipes/recipe_agent/`.
2. Add dependencies to the root `pyproject.toml`.
3. Add a CLI entry under `[project.scripts]`, e.g.
   `recipe-agent = "recipes.recipe_agent.main:app"`.
4. Add the folder to `[tool.hatch.build.targets.wheel] packages`.
5. Run `uv sync`.

## Conventions

- One `pyproject.toml` at the root for all dependencies and CLI entrypoints.
- One `.env` at the root for shared secrets.
- One `.venv` at the root, managed by `uv`.
- Per project: source, tests, `README.md`, `BACKLOG.md`.
