# agentry

A hands-on lab for mastering [LangGraph](https://github.com/langchain-ai/langgraph) at the enterprise AI production level.

The goal is to build realistic agent systems that exercise the patterns needed in production: graph control flow, durable state, tool integration, human review, safety boundaries, evals, observability, and deployment discipline. Each subfolder is a self-contained app; they all share one `pyproject.toml`, one `.venv`, and one `.env`.

See [`ideas.md`](./ideas.md) for the running backlog of project ideas.

## Learning goals

- Build agents around explicit graph state, typed inputs, and predictable routing.
- Practice production controls: retries, fallbacks, approvals, auditability, and safe tool use.
- Add evals and tests that can run locally and in CI without relying on live LLM calls.
- Integrate real external systems through APIs, tools, and MCP-style interfaces.
- Keep each project small enough to understand, but realistic enough to expose enterprise concerns.

## Projects

| Project | CLI | Description |
|---|---|---|
| [`emails/`](./emails/) | `inbox-agent` | Production-style email triage graph that reads unread Gmail, classifies messages, and drafts replies in your voice without sending. |

## Setup (one-time)

```bash
cp .env.example .env
# edit .env with your OPENAI_API_KEY (and any project-specific keys)

uv sync
```

## Daily use

Run any project's CLI from anywhere in the repo:

```bash
uv run inbox-agent run --dry-run
uv run pytest                      # all tests across all projects
uv run pytest emails/tests         # one project's tests
```

## Adding a new project

1. Create a new folder, e.g. `recipes/`, with an empty `__init__.py` and your package code (e.g. `recipes/recipe_agent/`).
2. Add any new dependencies to the root `pyproject.toml` (the comment block keeps it organized by project).
3. Add a CLI entry under `[project.scripts]`, e.g. `recipe-agent = "recipes.recipe_agent.main:app"`.
4. Add the folder to `[tool.hatch.build.targets.wheel] packages`.
5. Run `uv sync`.

## Conventions

- **One `pyproject.toml`** at the root holds all deps and CLI entrypoints.
- **One `.env`** at the root for shared secrets.
- **One `.venv`** at the root, managed by `uv`.
- **Per-project:** source code, tests, `README.md`, `BACKLOG.md`.
- New project ideas land in `ideas.md` first.
