# agentry

A personal lab of small, real agent projects built on [LangGraph](https://github.com/langchain-ai/langgraph) — learning by building things I'd actually use. Each subfolder is a self-contained app; they all share one `pyproject.toml`, one `.venv`, and one `.env`.

See [`ideas.md`](./ideas.md) for the running backlog of project ideas.

## Projects

| Project | CLI | Description |
|---|---|---|
| [`emails/`](./emails/) | `inbox-agent` | Triages unread Gmail and drafts replies in your voice. |

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
