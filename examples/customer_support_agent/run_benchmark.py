"""
Leeroopedia Customer Support Benchmark Runner

Pipeline:
  0. Pre-build deterministic test corpus (200 tickets, 27 intent categories)
  1. Baseline agent (no KB) builds the customer support API and runs tests
  2. KB agent (with Leeroopedia MCP) builds the same API and runs tests

Usage:
  python run_benchmark.py               # run with Anthropic API key (default)
  python run_benchmark.py --bedrock     # run with AWS Bedrock
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import load_dataset
from dotenv import load_dotenv

# --- Paths (all relative to this file's directory) ---
BENCH_DIR = Path(__file__).resolve().parent
PROPOSAL_FILE = BENCH_DIR / "proposal.md"
MCP_CONFIG_PATH = BENCH_DIR / "leeroopedia_mcp_config.json"
KB_TOOLS_REF = BENCH_DIR / "leeroopedia_tools_reference.md"
WORKSPACES_DIR = BENCH_DIR / "workspaces"
RESULTS_FILE = BENCH_DIR / "benchmark_results.json"

# Load .env from project directory
load_dotenv(BENCH_DIR / ".env")

# --- Model config ---
# Anthropic API model ID (default)
ANTHROPIC_MODEL = "claude-opus-4-5-20251101"
# Bedrock cross-region inference profile ID (used with --bedrock)
BEDROCK_MODEL = "us.anthropic.claude-opus-4-5-20251101-v1:0"

# Resolved at runtime based on --bedrock flag
USE_BEDROCK = False
AGENT_MODEL = ANTHROPIC_MODEL

# No timeout — customer support tasks can take a while
AGENT_TIMEOUT = None

# High max-turns for agent (needs many tool calls to build full service)
AGENT_MAX_TURNS = 200

# Sandbox + execution enforcement: appended to the agent's system prompt.
# This is the strongest lever we have to prevent Claude from stopping early.
# Claude Code often writes files then declares "done" without running them.
# Every sentence here is battle-tested against that failure mode.
SANDBOX_SYSTEM_PROMPT = (
    "CRITICAL RULES — VIOLATION OF ANY RULE IS A FAILURE:\n\n"
    "1. SANDBOX: You must ONLY read, write, and explore files within your "
    "current working directory. Do NOT access files or directories outside "
    "your current working directory.\n\n"
    "2. EXECUTE EVERYTHING: You MUST actually execute every script you write "
    "using Bash. Do NOT just write code and stop. After writing a script, "
    "run it IMMEDIATELY using the Bash tool and check the output. "
    "If it fails, debug and fix it, then re-run until it succeeds. "
    "Writing code without executing it is NOT acceptable and counts as "
    "a FAILURE.\n\n"
    "3. DO NOT STOP EARLY: Do NOT stop until the FULL pipeline has "
    "completed end-to-end and you have printed the final results. "
    "You have up to 200 tool-call turns — use as many as you need. "
    "Do NOT summarize what you 'would do' or say 'you can run this'. "
    "YOU must run it. Do NOT give up after one error — debug and retry. "
    "The task is NOT complete until you have executed the code AND "
    "printed the final benchmark numbers.\n\n"
    "4. VERIFY RESULTS: Before stopping, confirm that you have actually "
    "printed concrete output numbers (not placeholders). If you see an "
    "error in the output, fix it and re-run. Do NOT stop on an error.\n\n"
    "REMEMBER: Your output will be evaluated by an automated judge. "
    "The judge checks whether code was EXECUTED and whether results "
    "were PRINTED. Just writing files is worth ZERO points."
)

# All 27 Bitext intents — each is its own routing category.
# Grouped here for readability, but they are all separate categories.
ALL_INTENTS = [
    # Refund-related
    "get_refund", "check_refund_policy", "track_refund",
    # Order-related
    "cancel_order", "place_order", "change_order",
    # Payment-related
    "payment_issue", "check_invoice", "get_invoice",
    "check_payment_methods", "check_cancellation_fee",
    # Account-related
    "create_account", "delete_account", "edit_account",
    "switch_account", "recover_password", "registration_problems",
    # Support-related
    "contact_customer_service", "contact_human_agent", "complaint", "review",
    # Shipping-related
    "track_order", "set_up_shipping_address", "change_shipping_address",
    # Delivery-related
    "delivery_options", "delivery_period",
    # Subscription-related
    "newsletter_subscription",
]


# =============================================================================
# Test corpus builder
# =============================================================================


def build_corpus(verbose: bool = False) -> dict:
    """Build the 200-ticket test corpus deterministically.

    Distributes tickets across all 27 intents as evenly as possible.
    200 / 27 = 7 remainder 11, so 11 intents get 8 tickets and 16 get 7.

    The intent field is NOT included in the output — the agent must
    determine the intent purely from the message text.
    """
    if verbose:
        print("Loading Bitext dataset from HuggingFace...", flush=True)

    ds = load_dataset(
        "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
        split="train",
    )

    if verbose:
        print(f"Dataset loaded: {len(ds)} examples", flush=True)
        intents = sorted(set(ds["intent"]))
        print(f"Unique intents ({len(intents)}): {intents}", flush=True)

    # Group all examples by intent
    by_intent = {intent: [] for intent in ALL_INTENTS}
    for row in ds:
        intent = row["intent"]
        if intent in by_intent:
            by_intent[intent].append(row["instruction"])

    # Sort each intent's examples for determinism
    for intent in ALL_INTENTS:
        by_intent[intent].sort()

    if verbose:
        for intent in ALL_INTENTS:
            print(f"  {intent}: {len(by_intent[intent])} examples available",
                  flush=True)

    # Distribute 200 tickets across 27 intents: 11 get 8, 16 get 7
    total_target = 200
    base_count = total_target // len(ALL_INTENTS)  # 7
    remainder = total_target % len(ALL_INTENTS)     # 11

    tickets = []
    ticket_num = 1
    for i, intent in enumerate(ALL_INTENTS):
        # First 'remainder' intents get one extra ticket
        count = base_count + (1 if i < remainder else 0)
        available = by_intent[intent]
        actual = min(count, len(available))
        for msg in available[:actual]:
            tickets.append({
                "ticket_id": f"T-{ticket_num:03d}",
                "message": msg,
                "expected_category": intent,
            })
            ticket_num += 1

    cat_counts = Counter(t["expected_category"] for t in tickets)

    metadata = {
        "total_tickets": len(tickets),
        "num_categories": len(ALL_INTENTS),
        "categories": ALL_INTENTS,
        "per_category": {intent: cat_counts.get(intent, 0)
                         for intent in ALL_INTENTS},
    }

    if verbose:
        print(f"\nCorpus built: {len(tickets)} tickets across "
              f"{len(ALL_INTENTS)} intent categories", flush=True)
        for intent in ALL_INTENTS:
            print(f"  {intent}: {cat_counts.get(intent, 0)} tickets",
                  flush=True)

    return {"tickets": tickets, "metadata": metadata}


def save_corpus(output_dir: Path, verbose: bool = False) -> Path:
    """Build and save the test corpus to output_dir."""
    corpus = build_corpus(verbose=verbose)
    output_dir.mkdir(parents=True, exist_ok=True)

    corpus_path = output_dir / "test_corpus.json"
    corpus_path.write_text(
        json.dumps(corpus["tickets"], indent=2), encoding="utf-8"
    )

    meta_path = output_dir / "corpus_metadata.json"
    meta_path.write_text(
        json.dumps(corpus["metadata"], indent=2), encoding="utf-8"
    )

    if verbose:
        print(f"\nSaved to {output_dir}:", flush=True)
        print(f"  test_corpus.json ({len(corpus['tickets'])} tickets)")
        print(f"  corpus_metadata.json")

    return corpus_path


# =============================================================================
# File helpers
# =============================================================================


def _save_response_report(filepath: Path, response_text: str):
    """Save the agent's response as a markdown report."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text(response_text, encoding="utf-8")


def _format_readable_log(raw_log_lines: List[str]) -> List[str]:
    """Convert raw stream-json lines into a human-readable narrative."""
    # Pass 1: map tool_use_id -> tool_name
    tool_id_to_name: Dict[str, str] = {}
    parsed_events: List[dict] = []

    for raw in raw_log_lines:
        if not raw.strip():
            continue
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue
        parsed_events.append(event)
        if event.get("type") == "assistant":
            for block in event.get("message", {}).get("content", []):
                if block.get("type") == "tool_use":
                    tid = block.get("id", "")
                    tname = block.get("name", "?")
                    if tid:
                        tool_id_to_name[tid] = tname

    # Pass 2: format events into readable lines
    lines: List[str] = []
    RESULT_LIMIT = 1500

    for event in parsed_events:
        etype = event.get("type", "")

        if etype == "system" and event.get("subtype") == "init":
            model = event.get("model", "?")
            tools = event.get("tools", [])
            lines.append("=" * 70)
            lines.append(f"SESSION INIT  Model: {model}  Tools: {len(tools)}")
            lines.append("=" * 70)
            lines.append("")

        elif etype == "assistant":
            for block in event.get("message", {}).get("content", []):
                btype = block.get("type", "")
                if btype == "text":
                    text = block.get("text", "").strip()
                    if text:
                        lines.append("[ASSISTANT TEXT]")
                        for tline in text.split("\n"):
                            lines.append(f"  {tline}")
                        lines.append("")
                elif btype == "tool_use":
                    tool_name = block.get("name", "?")
                    lines.append(f"[TOOL CALL] {tool_name}")
                    tool_input = block.get("input", {})
                    input_str = json.dumps(tool_input, indent=2)
                    if len(input_str) > 500:
                        input_str = input_str[:500] + "..."
                    for iline in input_str.split("\n"):
                        lines.append(f"  {iline}")
                    lines.append("")

        elif etype == "user":
            content = event.get("message", {}).get("content", [])
            if isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") != "tool_result":
                        continue
                    tool_use_id = block.get("tool_use_id", "")
                    is_err = block.get("is_error", False)
                    origin = tool_id_to_name.get(tool_use_id, "?")
                    status = "ERROR" if is_err else "OK"
                    rc = block.get("content", "")
                    if isinstance(rc, list):
                        rc = "\n".join(
                            i.get("text", "")
                            for i in rc
                            if isinstance(i, dict) and i.get("type") == "text"
                        )
                    preview = str(rc)[:RESULT_LIMIT]
                    lines.append(f"[TOOL RESULT] {origin} -> {status}")
                    for rline in preview.split("\n"):
                        lines.append(f"    {rline}")
                    lines.append("")

        elif etype == "result":
            cost = event.get("total_cost_usd", 0)
            duration = event.get("duration_ms", 0)
            lines.append("=" * 70)
            err = event.get("is_error", False)
            lines.append(f"RESULT: {'ERROR' if err else 'SUCCESS'}")
            lines.append(f"  Duration: {duration/1000:.1f}s  Cost: ${cost:.4f}")
            lines.append("=" * 70)

    return lines


def _save_claude_log(filepath: Path, raw_log_lines: List[str]):
    """Save raw stream-json log and a human-readable .txt version."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text("\n".join(raw_log_lines), encoding="utf-8")

    readable_path = filepath.with_suffix(".txt")
    readable_lines = _format_readable_log(raw_log_lines)
    readable_path.write_text("\n".join(readable_lines), encoding="utf-8")


def _save_results(results: List[Dict]):
    """Save benchmark results to disk."""
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)


# =============================================================================
# Claude Code CLI runner
# =============================================================================


def _build_env() -> Dict[str, str]:
    """Build environment for Claude Code CLI.

    Two modes:
      - Anthropic API (default): uses ANTHROPIC_API_KEY from .env.
      - Bedrock (--bedrock flag): sets CLAUDE_CODE_USE_BEDROCK=1 and
        passes AWS_BEARER_TOKEN_BEDROCK.

    Both modes get the same normalized PATH and forwarded API keys.
    """
    env = os.environ.copy()

    if USE_BEDROCK:
        # Bedrock mode: authenticate via AWS bearer token
        env["CLAUDE_CODE_USE_BEDROCK"] = "1"
        env["AWS_REGION"] = "us-east-1"
        bearer_token = os.getenv("AWS_BEARER_TOKEN_BEDROCK", "")
        if bearer_token:
            env["AWS_BEARER_TOKEN_BEDROCK"] = bearer_token
    else:
        # Anthropic API mode (default): authenticate via API key
        env.pop("CLAUDE_CODE_USE_BEDROCK", None)
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            print("ERROR: ANTHROPIC_API_KEY not set in .env. "
                  "Use --bedrock flag for Bedrock mode.", flush=True)
            sys.exit(1)
        env["ANTHROPIC_API_KEY"] = api_key

    # Normalize PATH: ensure miniconda Python is first, then standard paths.
    # This prevents agents from accidentally using different Python versions.
    conda_bin = "/home/ubuntu/miniconda3/bin"
    standard_paths = [
        conda_bin,
        "/home/ubuntu/miniconda3/condabin",
        "/usr/local/sbin",
        "/usr/local/bin",
        "/usr/sbin",
        "/usr/bin",
        "/sbin",
        "/bin",
    ]
    # Preserve any existing non-standard PATH entries (e.g. npm/node)
    current_path = env.get("PATH", "").split(":")
    extra_paths = [p for p in current_path if p not in standard_paths and p]
    # Build normalized PATH: conda first, then extras, then standard
    normalized_path = standard_paths + extra_paths
    env["PATH"] = ":".join(dict.fromkeys(normalized_path))

    # Pass OpenAI API key so both agents have the same key
    openai_key = os.getenv("OPENAI_API_KEY", "")
    if openai_key:
        env["OPENAI_API_KEY"] = openai_key

    # Pass Leeroopedia API key for MCP config env expansion
    leeroopedia_key = os.getenv("LEEROOPEDIA_API_KEY", "")
    if leeroopedia_key:
        env["LEEROOPEDIA_API_KEY"] = leeroopedia_key

    return env


def _preinstall_common_packages():
    """Pre-install packages that both agents will need.

    This saves each agent from spending tool-call turns on pip install,
    and ensures they use the same package versions.
    """
    packages = [
        "fastapi",
        "uvicorn",
        "openai",
        "openai-agents",
        "datasets",
        "pydantic",
        "httpx",
        "sse-starlette",
    ]
    print("Pre-installing common packages...", flush=True)
    cmd = [
        "pip", "install", "--quiet", "--disable-pip-version-check",
    ] + packages
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=120,
    )
    if result.returncode == 0:
        print(f"  Installed: {', '.join(packages)}", flush=True)
    else:
        print(f"  Warning: pip install failed: {result.stderr[:200]}",
              flush=True)


def _print_live_event(event: Dict[str, Any]):
    """Print detailed live output of each Claude stream-json event."""
    etype = event.get("type", "")

    if etype == "system" and event.get("subtype") == "init":
        model = event.get("model", "?")
        tools = event.get("tools", [])
        mcp = event.get("mcp_servers", [])
        mcp_names = [s.get("name", "?") for s in mcp] if mcp else []
        print(
            f"    [init] model={model}  tools={len(tools)}  "
            f"mcp={mcp_names or 'none'}",
            flush=True,
        )

    elif etype == "assistant":
        for block in event.get("message", {}).get("content", []):
            btype = block.get("type", "")
            if btype == "text":
                text = block.get("text", "").strip()
                if text:
                    print("    [assistant]", flush=True)
                    for line in text.split("\n"):
                        print(f"      {line}", flush=True)
            elif btype == "tool_use":
                name = block.get("name", "?")
                inp = block.get("input", {})
                print(f"    [tool_call] {name}", flush=True)
                inp_str = json.dumps(inp, indent=2)
                for line in inp_str.split("\n"):
                    print(f"      {line}", flush=True)

    elif etype == "user":
        content = event.get("message", {}).get("content", [])
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    is_err = block.get("is_error", False)
                    status = "ERROR" if is_err else "OK"
                    rc = block.get("content", "")
                    if isinstance(rc, list):
                        rc = "\n".join(
                            i.get("text", "")
                            for i in rc
                            if isinstance(i, dict) and i.get("type") == "text"
                        )
                    rc = str(rc)
                    preview = rc[:500]
                    truncated = "..." if len(rc) > 500 else ""
                    print(f"    [tool_result] {status}", flush=True)
                    for line in preview.split("\n"):
                        print(f"      {line}", flush=True)
                    if truncated:
                        print(
                            f"      ... ({len(rc)} chars total)", flush=True
                        )

    elif etype == "result":
        cost = event.get("total_cost_usd", 0)
        dur = event.get("duration_ms", 0)
        err = event.get("is_error", False)
        status = "ERROR" if err else "SUCCESS"
        print(
            f"    [result] {status}  {dur/1000:.1f}s  ${cost:.4f}",
            flush=True,
        )


def run_claude_cli(
    prompt: str,
    model: str,
    workspace: str,
    mcp_config: Optional[Path] = None,
    append_system_prompt: Optional[str] = None,
    timeout: Optional[int] = None,
    max_turns: int = AGENT_MAX_TURNS,
) -> Dict[str, Any]:
    """Run Claude Code CLI and stream output.

    Prompt is piped via stdin (positional arg hangs in non-TTY).
    Returns dict with: response, success, error, cost_usd, time_seconds,
                       raw_log_lines, tool_call_count, input_tokens,
                       output_tokens.
    """
    cmd = [
        "claude",
        "--print",
        "--model", model,
        "--output-format", "stream-json",
        "--verbose",
        "--dangerously-skip-permissions",
        "--max-turns", str(max_turns),
        "--effort", "high",
    ]

    if mcp_config and mcp_config.exists():
        cmd.extend(["--mcp-config", str(mcp_config)])

    if append_system_prompt:
        cmd.extend(["--append-system-prompt", append_system_prompt])

    env = _build_env()

    raw_lines: List[str] = []
    assistant_texts: List[str] = []
    result_text = ""
    total_cost = 0.0
    is_error = False
    tool_call_count = 0
    input_tokens = 0
    output_tokens = 0

    start = time.time()

    proc = subprocess.Popen(
        cmd,
        cwd=workspace,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.PIPE,
        text=True,
        env=env,
    )

    # Write prompt to stdin, then close to signal EOF
    proc.stdin.write(prompt)
    proc.stdin.close()

    # Drain stderr in background to avoid deadlocks
    stderr_lines: List[str] = []

    def _drain_stderr():
        for line in proc.stderr:
            stderr_lines.append(line)

    stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
    stderr_thread.start()

    try:
        for line in proc.stdout:
            line = line.rstrip("\n")
            if not line:
                continue
            raw_lines.append(line)

            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            etype = event.get("type", "")
            _print_live_event(event)

            if etype == "assistant":
                for block in event.get("message", {}).get("content", []):
                    if block.get("type") == "text":
                        assistant_texts.append(block.get("text", ""))
                    elif block.get("type") == "tool_use":
                        tool_call_count += 1

            if etype == "result":
                result_text = event.get("result", "")
                total_cost = event.get("total_cost_usd", 0.0)
                is_error = event.get("is_error", False)
                usage = event.get("usage", {})
                input_tokens = usage.get("input_tokens", 0)
                output_tokens = usage.get("output_tokens", 0)

        proc.wait(timeout=timeout)
        stderr_thread.join(timeout=5)

    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        stderr_thread.join(timeout=5)
        is_error = True

    elapsed = time.time() - start

    # Fallback: sum per-turn tokens if result didn't include them
    if input_tokens == 0 or output_tokens == 0:
        cum_in, cum_out = 0, 0
        for raw in raw_lines:
            try:
                ev = json.loads(raw)
                if ev.get("type") == "assistant":
                    usage = ev.get("message", {}).get("usage", {})
                    cum_in += usage.get("input_tokens", 0)
                    cum_out += usage.get("output_tokens", 0)
            except json.JSONDecodeError:
                continue
        if input_tokens == 0:
            input_tokens = cum_in
        if output_tokens == 0:
            output_tokens = cum_out

    response = result_text or "\n".join(assistant_texts)
    success = proc.returncode == 0 and not is_error

    return {
        "response": response,
        "success": success,
        "error": (
            ""
            if success
            else (result_text if is_error else f"exit code {proc.returncode}")
        ),
        "cost_usd": total_cost,
        "time_seconds": round(elapsed, 2),
        "raw_log_lines": raw_lines,
        "tool_call_count": tool_call_count,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


# =============================================================================
# Single-phase runner (baseline or with_kb)
# =============================================================================


def _run_one_phase(
    label: str,
    task_text: str,
    model: str,
    workspace: str,
    task_ws: Path,
    corpus_dir: Optional[Path] = None,
    mcp_config: Optional[Path] = None,
    timeout: Optional[int] = AGENT_TIMEOUT,
) -> Dict[str, Any]:
    """Run one agent phase in an isolated /tmp sandbox.

    1. Create empty temp dir under /tmp
    2. Copy corpus files (test_corpus.json) into it
    3. Run agent with cwd = that temp dir
    4. Copy generated files back to real workspace
    5. Clean up temp dir
    """
    print(f"\n  --- {label} Starting ---", flush=True)
    sandbox_dir = tempfile.mkdtemp(prefix=f"bench_{label}_")
    print(f"  Sandbox: {sandbox_dir}", flush=True)

    # Copy pre-built corpus into sandbox so the agent can use it
    if corpus_dir:
        for fname in ["test_corpus.json", "corpus_metadata.json"]:
            src = corpus_dir / fname
            if src.exists():
                shutil.copy2(src, Path(sandbox_dir) / fname)
                print(f"  Copied {fname} into sandbox", flush=True)

    try:
        metrics = run_claude_cli(
            prompt=task_text,
            model=model,
            workspace=sandbox_dir,
            mcp_config=mcp_config,
            append_system_prompt=SANDBOX_SYSTEM_PROMPT,
            timeout=timeout,
        )

        # Copy files from sandbox to real workspace
        real_ws = Path(workspace)
        real_ws.mkdir(parents=True, exist_ok=True)
        for item in Path(sandbox_dir).iterdir():
            if item.name.startswith(".claude"):
                continue
            dest = real_ws / item.name
            if item.is_dir():
                shutil.copytree(item, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dest)
    finally:
        shutil.rmtree(sandbox_dir, ignore_errors=True)

    # Save logs and response
    prefix = label.lower().replace(" ", "_")
    _save_response_report(
        task_ws / f"{prefix}_response.md", metrics["response"]
    )
    _save_claude_log(
        task_ws / f"{prefix}_claude.log", metrics["raw_log_lines"]
    )

    print(
        f"  --- {label} Done: {metrics['time_seconds']}s, "
        f"${metrics['cost_usd']:.4f}, "
        f"{metrics['tool_call_count']} tool calls, "
        f"{metrics['input_tokens']}+{metrics['output_tokens']} tokens ---",
        flush=True,
    )
    return metrics


# =============================================================================
# Main benchmark: baseline -> with_kb
# =============================================================================


def run_proposal_benchmark():
    """Main pipeline:
      Phase 0: Pre-build deterministic test corpus
      Phase 1: Baseline agent (no KB) builds and tests the customer support API
      Phase 2: KB agent (with Leeroopedia MCP) builds and tests the same API

    Both agents receive the same pre-built test_corpus.json.
    """
    if not PROPOSAL_FILE.exists():
        print(f"ERROR: {PROPOSAL_FILE} not found")
        sys.exit(1)

    # --- Phase 0: Pre-build test corpus ---
    print(f"\n{'='*60}")
    print("Phase 0: Building deterministic test corpus")
    print(f"{'='*60}")
    corpus_dir = save_corpus(BENCH_DIR, verbose=True)
    corpus_parent = corpus_dir.parent
    print(f"  Corpus saved to: {corpus_parent}")

    _preinstall_common_packages()

    # Read proposal file (used for both baseline and with_kb)
    proposal_text = PROPOSAL_FILE.read_text(encoding="utf-8")
    print(f"Loaded proposal.md ({len(proposal_text)} chars)")

    # Read KB tools reference to append to the with_kb prompt
    kb_tools_text = ""
    if KB_TOOLS_REF.exists():
        kb_tools_text = KB_TOOLS_REF.read_text(encoding="utf-8")
        print(f"Loaded leeroopedia_tools_reference.md ({len(kb_tools_text)} chars)")
    else:
        print("WARNING: leeroopedia_tools_reference.md not found, "
              "with_kb agent won't have KB tool docs")

    # Build agent prompts with strong execution preamble.
    # The preamble is critical — Claude Code often writes files then stops.
    execution_preamble = (
        "IMPORTANT — READ THIS CAREFULLY BEFORE STARTING:\n\n"
        "You must implement AND EXECUTE the following customer support "
        "triage API proposal. This is NOT a code-writing exercise — you "
        "must build a working service and run it.\n\n"
        "MANDATORY WORKFLOW:\n"
        "1. Write the code for each component.\n"
        "2. IMMEDIATELY run it using Bash after writing it.\n"
        "3. If it fails, debug and fix it. Re-run until it succeeds.\n"
        "4. Start the FastAPI service in the background.\n"
        "5. Load test_corpus.json (already in your working directory).\n"
        "6. Send ALL 200 test tickets through the running service.\n"
        "7. Print the final results: routing accuracy and "
        "completion rate.\n\n"
        "Do NOT just write scripts and stop. Do NOT say 'you can run this "
        "later'. Do NOT summarize what the code does instead of running it. "
        "YOU must run every script yourself using Bash.\n\n"
        "You have up to 200 tool-call turns — use as many as you need. "
        "If something fails, fix it and try again. Do NOT give up.\n\n"
        "The task is ONLY complete when you have printed concrete test "
        "corpus numbers. Writing code without executing it scores ZERO.\n\n"
        "---\n\n"
    )

    # LLMs pay more attention to the end, so we repeat the key instruction.
    completion_reminder = (
        "\n\n---\n\n"
        "FINAL REMINDER: Do NOT stop after writing code. You MUST:\n"
        "1. Install all dependencies (pip install).\n"
        "2. Start the FastAPI service.\n"
        "3. Load test_corpus.json from your working directory.\n"
        "4. Run ALL 200 test tickets through it.\n"
        "5. Save the per-ticket results to a JSON file (e.g. test_results.json).\n"
        "6. Print the results: how many tickets processed, routing "
        "breakdown, a sample resolution, and "
        "whether the service ran without crashing.\n"
        "7. VERIFY that the category field in each result is one of the "
        "27 fine-grained intents (e.g. check_invoice, get_invoice), NOT "
        "a broad group name (e.g. payments). If you grouped specialists, "
        "make sure the router classified intent flows through to the "
        "final structured output.\n"
        "Writing code without running it is ZERO points. "
        "You have 200 turns — use them."
    )

    baseline_prompt = (
        f"{execution_preamble}"
        f"{proposal_text}"
        f"{completion_reminder}"
    )

    # with_kb prompt: same proposal + KB tools reference appended
    kb_prompt = f"{execution_preamble}{proposal_text}"
    if kb_tools_text:
        kb_prompt += f"\n\n---\n\n{kb_tools_text}"
    kb_prompt += completion_reminder

    # Create workspace directories
    task_ws = WORKSPACES_DIR / "proposal_benchmark"
    baseline_ws = task_ws / "baseline"
    kb_ws = task_ws / "with_kb"
    baseline_ws.mkdir(parents=True, exist_ok=True)
    kb_ws.mkdir(parents=True, exist_ok=True)

    # --- Phase 1: Baseline (no MCP) ---
    print(f"\n{'='*60}")
    print("Phase 1: Baseline (no MCP)")
    print(f"{'='*60}")
    baseline_metrics = _run_one_phase(
        label="baseline",
        task_text=baseline_prompt,
        model=AGENT_MODEL,
        workspace=str(baseline_ws),
        task_ws=task_ws,
        corpus_dir=corpus_parent,
        mcp_config=None,
    )

    # --- Phase 2: With KB (Leeroopedia MCP) ---
    print(f"\n{'='*60}")
    print("Phase 2: With KB (Leeroopedia MCP)")
    print(f"{'='*60}")
    kb_metrics = _run_one_phase(
        label="with_kb",
        task_text=kb_prompt,
        model=AGENT_MODEL,
        workspace=str(kb_ws),
        task_ws=task_ws,
        corpus_dir=corpus_parent,
        mcp_config=MCP_CONFIG_PATH,
    )

    # Build and save result dict
    result = {
        "task": "Customer Support Triage API (proposal.md)",
        "proposal_file": str(PROPOSAL_FILE),
        "baseline": {
            "workspace_path": str(baseline_ws),
            "response": baseline_metrics["response"][:2000],
            "time_seconds": baseline_metrics["time_seconds"],
            "cost_usd": baseline_metrics["cost_usd"],
            "tool_call_count": baseline_metrics["tool_call_count"],
            "input_tokens": baseline_metrics["input_tokens"],
            "output_tokens": baseline_metrics["output_tokens"],
        },
        "with_kb": {
            "workspace_path": str(kb_ws),
            "response": kb_metrics["response"][:2000],
            "time_seconds": kb_metrics["time_seconds"],
            "cost_usd": kb_metrics["cost_usd"],
            "tool_call_count": kb_metrics["tool_call_count"],
            "input_tokens": kb_metrics["input_tokens"],
            "output_tokens": kb_metrics["output_tokens"],
        },
    }

    _save_results([result])

    # Print summary
    print(f"\n{'='*60}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*60}")
    print(
        f"\n  Baseline: {baseline_metrics['time_seconds']}s, "
        f"${baseline_metrics['cost_usd']:.4f}, "
        f"{baseline_metrics['tool_call_count']} tool calls"
    )
    print(
        f"  With KB:  {kb_metrics['time_seconds']}s, "
        f"${kb_metrics['cost_usd']:.4f}, "
        f"{kb_metrics['tool_call_count']} tool calls"
    )
    print(f"\n  Results saved to: {RESULTS_FILE}")
    print(f"{'='*60}")

    return result


# =============================================================================
# Main entry point
# =============================================================================

def _parse_args():
    """Parse CLI arguments. --bedrock switches to Bedrock mode."""
    parser = argparse.ArgumentParser(
        description="Leeroopedia Customer Support Benchmark Runner"
    )
    parser.add_argument(
        "--bedrock",
        action="store_true",
        default=False,
        help="Use AWS Bedrock instead of Anthropic API (default: Anthropic API)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # Set global mode based on CLI flag
    USE_BEDROCK = args.bedrock
    AGENT_MODEL = BEDROCK_MODEL if USE_BEDROCK else ANTHROPIC_MODEL

    mode_label = "Bedrock" if USE_BEDROCK else "Anthropic API"
    print(f"Mode: {mode_label}  Model: {AGENT_MODEL}", flush=True)

    run_proposal_benchmark()
