"""Typer CLI for inbox-agent."""

from __future__ import annotations

import os
import time
import uuid

import typer
from rich.console import Console
from rich.table import Table

from langgraph.types import Command

from .graph import CHECKPOINT_DB, build_graph, open_graph
from .state import Classification

app = typer.Typer(help="Two-agent LangGraph that triages your Gmail and drafts replies.")
voice_app = typer.Typer(help="Manage long-term voice memory (cross-thread store).")
app.add_typer(voice_app, name="voice")
console = Console()

VOICE_NAMESPACE = ("voice", "drafts")


@app.command()
def run(
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Use bundled fixtures and skip all Gmail writes."
    ),
    thread: str = typer.Option(
        "default", "--thread", help="Checkpoint thread id. Same id = same memory."
    ),
    inspect_after: bool = typer.Option(
        False, "--inspect", help="After the run, dump the checkpoint history."
    ),
    skip_review: bool = typer.Option(
        True, "--skip-review/--review",
        help="Skip HITL interrupts and auto-save all drafts (default). Use `--review` for interactive approval.",
    ),
    fake_llm: bool = typer.Option(
        False, "--fake-llm", help="Stub triager/drafter (no OpenAI, no Gmail writes)."
    ),
):
    """Fetch unread emails, triage them, draft replies. Runs once.

    By default runs non-interactively (suitable for cron). Pass `--review`
    to pause on each draft for human approval before saving.
    """
    if fake_llm:
        os.environ["FAKE_LLM"] = "1"
    try:
        with open_graph(thread, skip_review=skip_review) as (graph, config):
            result = graph.invoke({"dry_run": dry_run}, config=config)
            if result and result.get("summary"):
                pass
            if inspect_after:
                _dump_history(graph, config)
    finally:
        if fake_llm:
            os.environ.pop("FAKE_LLM", None)


@app.command()
def inspect(
    thread: str = typer.Option(..., "--thread", help="Thread to inspect."),
):
    """Read checkpoint history for a thread WITHOUT running the graph.

    Proves cross-process durability: this command opens the same SqliteSaver
    DB and walks `get_state_history` for `thread`. Run it in a fresh terminal
    after a `run --thread X --dry-run` and you'll see all the checkpoints
    from that earlier process.
    """
    with open_graph(thread) as (graph, config):
        latest = graph.get_state(config)
        if latest.created_at is None:
            console.print(
                f"[yellow]No checkpoints found for thread '{thread}'.[/yellow] "
                f"Did you run `inbox-agent run --thread {thread} --dry-run` yet?"
            )
            raise typer.Exit(1)
        _dump_history(graph, config)


def _dump_history(graph, config) -> None:
    """Print every checkpoint LangGraph wrote for this thread."""
    latest = graph.get_state(config)
    console.rule("Latest StateSnapshot")
    console.print(f"[bold]next nodes to run:[/bold] {latest.next or '() — graph is finished'}")
    console.print(f"[bold]checkpoint_id:[/bold] {latest.config['configurable']['checkpoint_id']}")
    console.print(f"[bold]state keys:[/bold] {sorted(latest.values.keys())}")
    console.print(f"[bold]created_at:[/bold] {latest.created_at}")

    table = Table(title="Checkpoint history (newest → oldest)", show_lines=False)
    table.add_column("step", justify="right")
    table.add_column("source")
    table.add_column("node queued (in .tasks)")
    table.add_column("keys it will write into next checkpoint", overflow="fold")
    table.add_column("checkpoint_id (last 12)", overflow="fold")

    for snap in graph.get_state_history(config):
        meta = snap.metadata or {}
        node_names = ", ".join(t.name for t in snap.tasks) or "—"
        keys_written = ", ".join(
            sorted({k for t in snap.tasks for k in (t.result or {})})
        ) or "—"
        table.add_row(
            str(meta.get("step", "?")),
            str(meta.get("source", "?")),
            node_names,
            keys_written,
            snap.config["configurable"]["checkpoint_id"][-12:],
        )
    console.print(table)


@app.command("time-travel")
def time_travel(
    thread: str = typer.Option(
        "auto", "--thread", help="Default 'auto' = time-travel-{unix_ts}."
    ),
):
    """Run once, then rewrite the Stripe classification mid-flight and resume.

    Demonstrates `update_state(..., as_node=...)` for branching from any past
    checkpoint. Defaults to a fresh thread per call so demos are reproducible
    against the persisted SqliteSaver.
    """
    if thread == "auto":
        thread = f"time-travel-{int(time.time())}"
    console.print(f"[dim]using thread: {thread}[/dim]")

    with open_graph(thread) as (graph, config):
        console.rule("[bold]1. ORIGINAL RUN[/bold] (Stripe classified as fyi → no draft)")
        state_a = graph.invoke({"dry_run": True}, config=config)
        drafts_a = sorted(state_a.get("drafts", {}).keys())
        console.print(f"original drafts: {drafts_a}")

        console.rule("[bold]2. WALK HISTORY[/bold] (find the snapshot to branch from)")
        branch_snapshot = next(
            (s for s in graph.get_state_history(config) if (s.metadata or {}).get("step") == 6),
            None,
        )
        if branch_snapshot is None:
            console.print("[red]Could not find step 6.[/red]")
            raise typer.Exit(1)
        branch_id = branch_snapshot.config["configurable"]["checkpoint_id"]
        console.print(f"branching from step 6, checkpoint_id ...{branch_id[-12:]}")
        console.print(
            "  state at branch: current=msg_002, "
            f"classifications keys={sorted(branch_snapshot.values.get('classifications', {}).keys())}"
        )

        console.rule("[bold]3. INJECT NEW STATE[/bold] via update_state(as_node='triager')")
        new_class = Classification(
            category="needs_reply",
            urgency="normal",
            summary="Receipt; want to ask about itemization.",
            intent="confirm receipt and ask for an itemized invoice",
        )
        branched_config = graph.update_state(
            branch_snapshot.config,
            {"classifications": {"msg_002": new_class}},
            as_node="triager",
        )
        new_id = branched_config["configurable"]["checkpoint_id"]
        console.print(f"new branch checkpoint: ...{new_id[-12:]}  (parent: ...{branch_id[-12:]})")
        console.print(
            "  as_node='triager' is what makes the conditional edge after triager re-fire"
        )

        console.rule("[bold]4. RESUME[/bold] graph.invoke(None, branched_config)")
        state_b = graph.invoke(None, config=branched_config)
        drafts_b = sorted(state_b.get("drafts", {}).keys())

        console.rule("[bold]5. DIFF[/bold]")
        table = Table(title="Drafts: original vs branched timeline")
        table.add_column("email_id")
        table.add_column("subject", overflow="fold", max_width=40)
        table.add_column("original")
        table.add_column("branched")
        emails = {e.id: e for e in state_b.get("emails", [])}
        for eid in sorted(set(drafts_a) | set(drafts_b)):
            in_a = eid in drafts_a
            in_b = eid in drafts_b
            marker_b = "yes (NEW)" if in_b and not in_a else ("yes" if in_b else "—")
            marker_a = "yes" if in_a else "—"
            table.add_row(eid, emails[eid].subject if eid in emails else "?", marker_a, marker_b)
        console.print(table)

        if "msg_002" in state_b.get("drafts", {}):
            console.rule("New Stripe draft (only exists in branched timeline)")
            console.print(state_b["drafts"]["msg_002"].body)

        console.print(
            f"\n[dim]Tip: the branched checkpoints are persisted in {CHECKPOINT_DB}. "
            f"Try `uv run inbox-agent inspect --thread {thread}` in another terminal.[/dim]"
        )


@app.command()
def watch(
    dry_run: bool = typer.Option(True, "--dry-run/--live"),
    thread: str = typer.Option("watch-demo", "--thread"),
    mode: str = typer.Option("updates", "--mode", help="updates | values | messages | custom | all"),
):
    """Stream the graph with stream_mode={mode}.

    `updates`   — per-node diffs (one chunk per super-step). Cheap, structured.
    `values`    — full state after each step. Big, useful for UI re-renders.
    `messages`  — one chunk per LLM token. Same UX as ChatGPT typing.
    `custom`    — what nodes emit via `get_stream_writer()`.
    `all`       — subscribe to updates+messages+custom in one feed.
    """
    with open_graph(thread) as (graph, config):
        console.print(f"[dim]thread: {thread}  |  stream_mode='{mode}'[/dim]\n")

        if mode == "messages":
            _stream_messages(graph, dry_run, config)
            return
        if mode == "all":
            _stream_all(graph, dry_run, config)
            return

        for chunk in graph.stream(
            {"dry_run": dry_run}, config=config, stream_mode=mode
        ):
            if mode == "updates":
                for node, payload in chunk.items():
                    if node == "__interrupt__":
                        for itr in payload or ():
                            val = getattr(itr, "value", itr)
                            console.print(
                                f"[bold red]⏸  INTERRUPT[/bold red] payload: {val}"
                            )
                        continue
                    keys = sorted((payload or {}).keys())
                    console.print(f"[bold cyan]→ {node:14}[/bold cyan] wrote: {keys}")
            elif mode == "values":
                console.print(
                    f"[bold cyan]state[/bold cyan]  keys: {sorted(chunk.keys())}  "
                    f"queue_len: {len(chunk.get('queue', []))}  "
                    f"drafts: {len(chunk.get('drafts', {}))}"
                )
            elif mode == "custom":
                # chunk is whatever a node wrote via get_stream_writer()
                step = chunk.get("step", "?") if isinstance(chunk, dict) else "?"
                msg = chunk.get("msg", chunk) if isinstance(chunk, dict) else chunk
                console.print(f"[bold yellow]·[/bold yellow] [bold]{step:14}[/bold] {msg}")


def _flush_token_buffer(node: str | None, buffer: list[str]) -> None:
    """Render an accumulated token buffer as a one-line summary."""
    if not node or not buffer:
        return
    total_chars = sum(len(t) for t in buffer)
    console.print(
        f"[dim]» {node:14} streamed {len(buffer):3} chunks "
        f"({total_chars} chars)[/dim]"
    )


def _stream_all(graph, dry_run: bool, config: dict) -> None:
    """Subscribe to updates+messages+custom and render a unified live feed."""
    buffer: list[str] = []
    buffer_node: str | None = None

    def flush() -> None:
        nonlocal buffer, buffer_node
        _flush_token_buffer(buffer_node, buffer)
        buffer, buffer_node = [], None

    for kind, payload in graph.stream(
        {"dry_run": dry_run},
        config=config,
        stream_mode=["updates", "messages", "custom"],
    ):
        if kind == "updates":
            flush()
            for node, p in payload.items():
                if node == "__interrupt__":
                    for itr in p or ():
                        val = getattr(itr, "value", itr)
                        console.print(
                            f"[bold red]⏸  INTERRUPT[/bold red] payload: {val}"
                        )
                    continue
                keys = sorted((p or {}).keys())
                console.print(f"[bold cyan]→ {node:14}[/bold cyan] wrote: {keys}")
        elif kind == "custom":
            flush()
            step = payload.get("step", "?") if isinstance(payload, dict) else "?"
            msg = payload.get("msg", payload) if isinstance(payload, dict) else payload
            console.print(f"[bold yellow]·[/bold yellow] [bold]{step:14}[/bold] {msg}")
        elif kind == "messages":
            chunk, meta = payload
            node = meta.get("langgraph_node", "?")
            text = _extract_token_text(chunk)
            if not text:
                continue
            if buffer_node and node != buffer_node:
                flush()
            buffer_node = node
            buffer.append(text)
    flush()


def _extract_token_text(chunk) -> str:
    """Pull human-readable text from a streamed message chunk.

    Two cases:
    - Plain text completion: `chunk.content` is the partial string.
    - Structured output via tool-calling: text lives in `tool_call_chunks[i].args`
      (partial JSON arguments).
    """
    text = getattr(chunk, "content", "") or ""
    if text:
        return text
    for tc in getattr(chunk, "tool_call_chunks", None) or []:
        text += tc.get("args") or ""
    return text


def _stream_messages(graph, dry_run: bool, config: dict) -> None:
    """Print LLM tokens as they arrive, with a header whenever the node changes."""
    last_node = None
    for chunk, metadata in graph.stream(
        {"dry_run": dry_run}, config=config, stream_mode="messages"
    ):
        node = metadata.get("langgraph_node", "?")
        text = _extract_token_text(chunk)
        if not text:
            continue
        if node != last_node:
            console.print(f"\n[bold cyan]→ {node}[/bold cyan]: ", end="")
            last_node = node
        console.print(text, end="")
    console.print()


@app.command()
def review(
    dry_run: bool = typer.Option(True, "--dry-run/--live"),
    thread: str = typer.Option(None, "--thread", help="Default: review-{ts}."),
):
    """Run the inbox interactively — pause on each draft for approve/edit/skip.

    Implements the full HITL loop:
      1. stream the graph until an `__interrupt__` chunk arrives
      2. pretty-print the draft, prompt for a decision
      3. resume with `Command(resume=decision)`
      4. repeat until stream ends without interrupt
    """
    if not thread:
        thread = f"review-{int(time.time())}"
    console.print(f"[dim]thread: {thread}[/dim]\n")

    with open_graph(thread, skip_review=False) as (graph, config):
        next_input: object = {"dry_run": dry_run}

        while True:
            paused_at: list = []
            for chunk in graph.stream(next_input, config=config, stream_mode="updates"):
                for node, payload in chunk.items():
                    if node == "__interrupt__":
                        paused_at.extend(payload or ())
                        continue
                    keys = sorted((payload or {}).keys()) if isinstance(payload, dict) else []
                    console.print(f"[dim]→ {node:14} wrote: {keys}[/dim]")

            if not paused_at:
                # Stream ended without an interrupt → graph reached END
                break

            # Sequential graph: at most one interrupt per pause.
            decision = _prompt_for_decision(paused_at[0].value)
            next_input = Command(resume=decision)

        console.rule("[bold green]Done[/bold green]")
        final = graph.get_state(config).values
        drafts = final.get("drafts", {})
        saved = final.get("saved_draft_ids", {})
        console.print(
            f"emails processed: {len(final.get('emails', []))}  "
            f"drafts created: {len(drafts)}  "
            f"saved (post-review): {len(saved)}"
        )


def _prompt_for_decision(payload: dict) -> dict:
    """Show the draft and ask the user what to do."""
    email = payload.get("email", {}) if isinstance(payload, dict) else {}
    body = payload.get("draft_body", "") if isinstance(payload, dict) else str(payload)

    console.rule("[bold yellow]Review draft[/bold yellow]")
    console.print(f"[bold]To:[/bold]      {email.get('from', '?')}")
    console.print(f"[bold]Subject:[/bold] {email.get('subject', '?')}")
    console.print()
    console.print(body)
    console.print()

    while True:
        action = (
            typer.prompt("Action: [a]pprove / [e]dit / [s]kip", default="a").strip().lower()
        )
        if action in ("a", "approve", ""):
            return {"action": "approve"}
        if action in ("s", "skip"):
            return {"action": "skip"}
        if action in ("e", "edit"):
            new_body = typer.prompt("New draft body (single line)", default=body)
            return {"action": "edit", "body": new_body}
        console.print("[yellow]Pick a / e / s.[/yellow]")


@app.command("async-watch")
def async_watch(
    thread: str = typer.Option("async-demo", "--thread"),
):
    """Same as `watch --mode updates` but using ainvoke + AsyncSqliteSaver.

    Demonstrates the async API surface. In production (FastAPI, etc.),
    every graph call should be async so LLM I/O doesn't block the event loop.
    """
    import asyncio

    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    async def _run():
        async with AsyncSqliteSaver.from_conn_string(CHECKPOINT_DB) as cp:
            graph = build_graph(checkpointer=cp)
            config = {"configurable": {"thread_id": thread}}
            console.print(f"[dim]thread: {thread}  |  async stream[/dim]\n")
            async for chunk in graph.astream(
                {"dry_run": True}, config=config, stream_mode="updates"
            ):
                for node, payload in chunk.items():
                    if node == "__interrupt__":
                        continue
                    keys = sorted((payload or {}).keys()) if isinstance(payload, dict) else []
                    if keys:
                        console.print(f"[bold cyan]→ {node:16}[/bold cyan] wrote: {keys}")

    asyncio.run(_run())


@app.command("parallel-demo")
def parallel_demo(
    thread: str = typer.Option(None, "--thread", help="Default: parallel-{ts}."),
):
    """Run inbox with Send-based parallel triage — all emails classified at once.

    Compares wall-clock time for sequential (pop_next loop) vs parallel (Send
    fan-out). All 6 triagers fire in the same super-step; results merge via
    the `classifications` reducer.
    """
    if not thread:
        thread = f"parallel-{int(time.time())}"
    console.print(f"[dim]thread: {thread}[/dim]\n")

    import time as _time

    console.rule("[bold]Sequential (pop_next loop)[/bold]")
    t0 = _time.monotonic()
    with open_graph(f"{thread}-seq") as (graph_seq, config_seq):
        for chunk in graph_seq.stream({"dry_run": True}, config=config_seq, stream_mode=["custom"]):
            _kind, payload = chunk
            if isinstance(payload, dict) and payload.get("step") in ("triager", "parallel_triager"):
                console.print(f"  [dim]{payload['msg']}[/dim]")
    seq_elapsed = _time.monotonic() - t0

    console.print()
    console.rule("[bold]Parallel (Send fan-out)[/bold]")
    t0 = _time.monotonic()
    with open_graph(f"{thread}-par", parallel=True) as (graph_par, config_par):
        for chunk in graph_par.stream({"dry_run": True}, config=config_par, stream_mode=["custom"]):
            _kind, payload = chunk
            if isinstance(payload, dict) and payload.get("step") in ("fan_out", "parallel_triager"):
                console.print(f"  [dim]{payload['msg']}[/dim]")
    par_elapsed = _time.monotonic() - t0

    console.rule("[bold]Timing[/bold]")
    console.print(f"Sequential: {seq_elapsed:.1f}s")
    console.print(f"Parallel:   {par_elapsed:.1f}s")
    speedup = seq_elapsed / par_elapsed if par_elapsed > 0 else 0
    console.print(f"Speedup:    {speedup:.1f}×")


@app.command("supervisor-demo")
def supervisor_demo(
    thread: str = typer.Option(None, "--thread", help="Default: supervisor-{ts}."),
):
    """Run with the LLM supervisor routing emails to specialized workers.

    Instead of a deterministic `_route_after_triage` conditional edge,
    an LLM supervisor decides each email's handler via Command(goto=...).
    Three possible routes: drafter / calendar_agent / pop_next.
    """
    if not thread:
        thread = f"supervisor-{int(time.time())}"
    console.print(f"[dim]thread: {thread}  |  use_supervisor=True[/dim]\n")

    with open_graph(thread, use_supervisor=True) as (graph, config):
        for chunk in graph.stream(
            {"dry_run": True}, config=config, stream_mode=["updates", "custom"]
        ):
            kind, payload = chunk
            if kind == "custom":
                step = payload.get("step", "?") if isinstance(payload, dict) else "?"
                msg = payload.get("msg", payload) if isinstance(payload, dict) else payload
                console.print(f"[bold yellow]·[/bold yellow] [bold]{step:16}[/bold] {msg}")
            elif kind == "updates":
                for node, p in (payload or {}).items():
                    if node == "__interrupt__":
                        continue
                    keys = sorted((p or {}).keys()) if isinstance(p, dict) else []
                    if keys:
                        console.print(f"[bold cyan]→[/bold cyan] [bold]{node:16}[/bold] wrote: {keys}")


@app.command("react-demo")
def react_demo(
    thread: str = typer.Option(None, "--thread", help="Default: react-{ts}."),
    manual: bool = typer.Option(
        False, "--manual",
        help="Use the hand-built ReAct loop instead of create_react_agent.",
    ),
):
    """Run the inbox with the ReAct drafter — see tool calls happen live.

    With `--manual`, swaps `create_react_agent` for a hand-built equivalent
    using `bind_tools` + `ToolNode` + `tools_condition`. Same trace shape,
    proves the prebuilt is just a convenience over the primitives.

    Streams the outer graph with `subgraphs=True` so we can also see the
    inner agent's per-step activity (model call, tool call, model call, ...).
    """
    if not thread:
        thread = f"react-{int(time.time())}"
    drafter_kind = "manual-react" if manual else "react"
    console.print(f"[dim]thread: {thread}  |  drafter={drafter_kind}[/dim]\n")

    with open_graph(thread, drafter=drafter_kind) as (graph, config):
        for ns, kind, payload in graph.stream(
            {"dry_run": True},
            config=config,
            stream_mode=["updates", "custom"],
            subgraphs=True,
        ):
            prefix = f"[{ns[-1].split(':')[0] if ns else 'root'}]" if ns else "[root]"
            if kind == "custom":
                step = payload.get("step", "?") if isinstance(payload, dict) else "?"
                msg = payload.get("msg", payload) if isinstance(payload, dict) else payload
                console.print(f"[bold yellow]·[/bold yellow] {prefix:8} {step:18} {msg}")
            elif kind == "updates":
                for node, p in payload.items():
                    if node == "__interrupt__":
                        for itr in p or ():
                            console.print(f"[bold red]⏸  INTERRUPT[/bold red] in {prefix}")
                        continue
                    keys = sorted((p or {}).keys()) if isinstance(p, dict) else []
                    console.print(f"[bold cyan]→[/bold cyan] {prefix:8} {node:18} wrote: {keys}")


@app.command("tool-test")
def tool_test(
    name: str = typer.Argument(..., help="Tool name (e.g. lookup_contact_history)."),
    arg: str = typer.Argument(..., help="Single string argument."),
):
    """Invoke a tool from llm_tools.TOOLS directly to verify it works.

    Bypasses the LLM. For tools that need a Store (via `InjectedStore`),
    we open the graph just to get a bound store and pass it explicitly to
    the underlying function. Useful for sanity-checking tools before
    plugging them into a ReAct agent in step 5.2.
    """
    from .llm_tools import TOOLS

    tool = next((t for t in TOOLS if t.name == name), None)
    if tool is None:
        names = [t.name for t in TOOLS]
        console.print(f"[red]No tool named {name!r}. Available: {names}[/red]")
        raise typer.Exit(1)

    # `tool.args` shows only the LLM-visible parameters (InjectedStore is hidden).
    visible_args = list(tool.args.keys())
    payload = {visible_args[0]: arg} if visible_args else {}
    console.print(f"[bold]→ {tool.name}({visible_args[0] if visible_args else ''}={arg!r})[/bold]\n")

    # Tool needs a store? Open the graph so we have one, then call `.func` directly.
    needs_store = "store" in (tool.func.__annotations__ if tool.func else {})
    if needs_store:
        with open_graph(thread_id="_tool_test") as (graph, _config):
            result = tool.func(**payload, store=graph.store)
    else:
        result = tool.invoke(payload)

    console.print(result)


@app.command()
def graph_png(out: str = "graph.png"):
    """Render the graph as a PNG (requires graphviz)."""
    png_bytes = build_graph().get_graph().draw_mermaid_png()
    with open(out, "wb") as f:
        f.write(png_bytes)
    console.print(f"Wrote {out}")


# ---------------------------------------------------------------------------
# voice — long-term memory CRUD via the bound SqliteStore
# ---------------------------------------------------------------------------

@voice_app.command("add")
def voice_add(
    context: str = typer.Option(
        ..., "--context", "-c",
        help="What the incoming email looked like (becomes the search key).",
    ),
    reply: str = typer.Option(
        ..., "--reply", "-r",
        help="The reply you wrote in your own voice.",
    ),
):
    """Save a (context, reply) pair as a future few-shot example.

    The `context` is what gets embedded — at draft time the store will be
    queried with the new incoming email, so make `context` look like a short
    description of an email.
    """
    with open_graph(thread_id="_voice_admin") as (graph, _config):
        store = graph.store  # the same SqliteStore we bound at compile()
        key = uuid.uuid4().hex
        store.put(
            VOICE_NAMESPACE,
            key=key,
            value={"text": context, "reply": reply},
        )
        console.print(f"[green]saved[/green] voice example {key[:8]}…")
        console.print(f"  context (embedded): {context}")
        console.print(f"  reply: {reply!r}")


@voice_app.command("search")
def voice_search(
    query: str = typer.Option(
        ..., "--query", "-q", help="Semantic query — find similar past emails."
    ),
    limit: int = typer.Option(3, "--limit", "-k"),
):
    """Find the top-k past replies whose context is most similar to `query`."""
    with open_graph(thread_id="_voice_admin") as (graph, _config):
        store = graph.store
        items = store.search(VOICE_NAMESPACE, query=query, limit=limit)
        if not items:
            console.print("[yellow]no voice examples yet — try `voice add` first.[/yellow]")
            return
        table = Table(title=f"Top {len(items)} matches for: {query!r}")
        table.add_column("score", justify="right")
        table.add_column("key")
        table.add_column("context", overflow="fold", max_width=40)
        table.add_column("reply", overflow="fold", max_width=50)
        for it in items:
            score = f"{it.score:.3f}" if it.score is not None else "—"
            table.add_row(score, it.key[:8] + "…", it.value.get("text", ""), it.value.get("reply", ""))
        console.print(table)


@voice_app.command("list")
def voice_list():
    """List ALL stored voice examples (no semantic search)."""
    with open_graph(thread_id="_voice_admin") as (graph, _config):
        store = graph.store
        items = store.search(VOICE_NAMESPACE, limit=100)
        console.print(f"[bold]{len(items)} voice example(s) in store.sqlite[/bold]")
        for it in items:
            console.print(f"  {it.key[:8]}… | {it.value.get('text', '')[:60]}")


@voice_app.command("demo")
def voice_demo():
    """A/B compare drafts WITHOUT vs WITH long-term voice memory.

    Same fixture inbox, two graphs, two threads. The only difference is
    whether a SqliteStore is bound at compile time. Compares the resulting
    drafts so you can see voice memory's impact directly.
    """
    ts = int(time.time())
    drafts_no_store: dict = {}
    drafts_with_store: dict = {}
    emails: dict = {}

    console.rule("[bold]Run A — WITHOUT voice store[/bold]")
    with open_graph(thread_id=f"voice-demo-no-{ts}", with_store=False) as (graph, config):
        state = graph.invoke({"dry_run": True}, config=config)
        drafts_no_store = state.get("drafts", {})
        emails = {e.id: e for e in state.get("emails", [])}

    console.rule("[bold]Run B — WITH voice store[/bold]")
    with open_graph(thread_id=f"voice-demo-yes-{ts}", with_store=True) as (graph, config):
        state = graph.invoke({"dry_run": True}, config=config)
        drafts_with_store = state.get("drafts", {})

    console.rule("[bold]A/B comparison[/bold]")
    common = sorted(set(drafts_no_store) & set(drafts_with_store))
    if not common:
        console.print("[yellow]No drafts produced in either run — nothing to compare.[/yellow]")
        return
    for eid in common:
        email = emails.get(eid)
        title = f"{email.sender} — {email.subject}" if email else eid
        console.rule(f"[bold cyan]{title}[/bold cyan]")
        table = Table(show_header=True, header_style="bold")
        table.add_column("WITHOUT voice (Run A)", overflow="fold")
        table.add_column("WITH voice (Run B)", overflow="fold")
        table.add_row(drafts_no_store[eid].body, drafts_with_store[eid].body)
        console.print(table)


if __name__ == "__main__":
    app()
