# inbox-agent

A two-agent [LangGraph](https://github.com/langchain-ai/langgraph) that runs every morning, triages your unread Gmail, and saves draft replies in your voice.

**It never sends.** Only `drafts.create` is used.

## Architecture

```
                    ┌────────────────┐
                    │  fetch_emails  │  Gmail API (or fixtures in --dry-run)
                    └────────┬───────┘
                             ▼
                ┌────────────────────────┐
                │   Agent A: Triager     │  classify + extract intent
                │   (structured output)  │
                └────────┬───────────────┘
                         │
              needs_reply │ else
                         ▼
                ┌────────────────────────┐
                │   Agent B: Drafter     │  writes reply in your voice
                └────────┬───────────────┘
                         ▼
                ┌────────────────┐
                │   save_draft   │  Gmail API
                └────────┬───────┘
                         ▼
                ┌────────────────┐
                │    summary     │  morning brief printed to terminal
                └────────────────┘
```

## Quick start

This is a member of the [`agentry`](../) uv workspace. Install and run from the repo root.

```bash
# 1. From the repo root, one-time setup
cd ..
cp .env.example .env       # add OPENAI_API_KEY etc.
uv sync                    # installs all workspace members into the shared .venv/

# 2. Try it on the bundled fixtures (no Gmail needed)
uv run inbox-agent run --dry-run

# 3. When ready, set up Gmail OAuth (see below) and run for real
uv run inbox-agent run
```

## Gmail setup (only needed for real runs)

1. Go to [console.cloud.google.com](https://console.cloud.google.com), create a project.
2. Enable the Gmail API.
3. Create OAuth client credentials (Desktop app), download JSON, save as `gmail_credentials.json`.
4. Add yourself as a test user under OAuth consent screen.
5. First run will open a browser to authorize. A `gmail_token.json` will be cached.

The agent only requests:
- `gmail.readonly` — to read unread messages
- `gmail.compose` — to create drafts (cannot send)

## Project layout

```
emails/
  __init__.py     # makes `emails` a Python package
  inbox_agent/
    state.py      # InboxState TypedDict + Pydantic models (Email, Classification, Draft)
    config.py     # Settings (loads .env from repo root)
    prompts.py    # System prompts for both agents
    agents.py     # Agent A (triager) + Agent B (drafter) — the LLM-backed nodes
    tools.py     # Gmail fetch/save, queue, summary printer, dry-run fixtures
    graph.py      # StateGraph wiring (conditional edges + the loop)
    main.py       # Typer CLI (registered at the root pyproject.toml)
  tests/
    test_graph.py # smoke test (stubs the LLM agents)
```

## Scheduling

Once it's working, schedule it with `launchd` or `cron`:

```bash
# crontab -e
0 7 * * * cd /Users/rahul/Projects/agentry && /opt/homebrew/bin/uv run inbox-agent run >> /tmp/inbox-agent.log 2>&1
```

## Safety

- `--dry-run` skips all Gmail writes and uses fixtures.
- `ALLOWLIST_SENDERS` in `.env` restricts whose emails get a drafted reply.
- The Gmail scope is `compose` only — sending is impossible.
