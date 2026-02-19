"""
Leeroopedia LLM Training Benchmark Runner

Reads proposal.md and runs it through Claude Code twice:
  1. Baseline: no knowledge base
  2. With KB: connected to Leeroopedia MCP

Usage:
  python run_benchmark.py               # run with Anthropic API key (default)
  python run_benchmark.py --bedrock     # run with AWS Bedrock
  python run_benchmark.py --test        # run a hello-world smoke test
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
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

# --- Paths (all relative to leeroopedia_llm_training/) ---
BENCH_DIR = Path(__file__).resolve().parent
load_dotenv(BENCH_DIR / ".env")

PROPOSAL_FILE = BENCH_DIR / "proposal.md"
EVAL_HELPER_SCRIPT = BENCH_DIR / "evaluate_model.py"
MCP_CONFIG_PATH = BENCH_DIR / "leeroopedia_mcp_config.json"
WORKSPACES_DIR = BENCH_DIR / "workspaces"
RESULTS_FILE = BENCH_DIR / "benchmark_results.json"

# --- Model config ---
# Anthropic API model ID (default)
ANTHROPIC_MODEL = "claude-opus-4-5-20251101"
# Bedrock cross-region inference profile ID (used with --bedrock)
BEDROCK_MODEL = "us.anthropic.claude-opus-4-5-20251101-v1:0"

# Resolved at runtime based on --bedrock flag
USE_BEDROCK = False
AGENT_MODEL = ANTHROPIC_MODEL
# No timeout for agent runs — LLM fine-tuning tasks can take hours
AGENT_TIMEOUT = None

# --- Execution config ---
# High max-turns so Claude Code keeps going through write + execute + debug cycles.
# LLM fine-tuning pipelines need many tool calls (write files, run training,
# check results, fix errors, run evals). 200 turns gives plenty of headroom.
AGENT_MAX_TURNS = 200

# Sandbox + execution enforcement: appended to the agent's system prompt
SANDBOX_SYSTEM_PROMPT = (
    "CRITICAL RULES:\n"
    "1. You must ONLY read, write, and explore files within your current working directory. "
    "Do NOT access files or directories outside your current working directory.\n"
    "2. You MUST actually execute every script you write using Bash. Do NOT just write code "
    "and stop. After writing a script, run it immediately and check the output. "
    "If it fails, fix and re-run until it succeeds.\n"
    "3. Do NOT stop until the full pipeline has completed and you have printed the final "
    "benchmark results. Writing code without executing it is NOT acceptable.\n"
    "4. For any command expected to take more than 2 minutes (training, evaluation, etc.), "
    "do NOT run it synchronously. Instead:\n"
    "   a) Run it in the background with output to a log file: `python script.py > output.log 2>&1 &`\n"
    "   b) Poll the log every 60 seconds with `tail -20 output.log` to monitor progress.\n"
    "   c) Check the process is still alive with `ps aux | grep script.py`.\n"
    "   d) If no new output appears for 5+ minutes, investigate — check for errors, OOM, or hangs.\n"
    "   e) Only proceed to the next step once the log confirms the command completed successfully.\n"
    "   Never fire-and-forget a long command. Always actively monitor its log output."
)


# =============================================================================
# Claude Code CLI runner
# =============================================================================

def _build_env() -> Dict[str, str]:
    """Build environment for Claude Code CLI.

    Two modes:
      - Anthropic API (default): uses ANTHROPIC_API_KEY from .env.
      - Bedrock (--bedrock flag): sets CLAUDE_CODE_USE_BEDROCK=1 and
        passes AWS_BEARER_TOKEN_BEDROCK.
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

    return env


def _print_live_event(event: Dict[str, Any]):
    """Print detailed live output of each Claude stream-json event."""
    etype = event.get("type", "")

    if etype == "system" and event.get("subtype") == "init":
        model = event.get("model", "?")
        tools = event.get("tools", [])
        mcp = event.get("mcp_servers", [])
        mcp_names = [s.get("name", "?") for s in mcp] if mcp else []
        print(f"    [init] model={model}  tools={len(tools)}  mcp={mcp_names or 'none'}", flush=True)

    elif etype == "assistant":
        for block in event.get("message", {}).get("content", []):
            btype = block.get("type", "")
            if btype == "text":
                text = block.get("text", "").strip()
                if text:
                    print(f"    [assistant]", flush=True)
                    for line in text.split("\n"):
                        print(f"      {line}", flush=True)
            elif btype == "tool_use":
                name = block.get("name", "?")
                inp = block.get("input", {})
                print(f"    [tool_call] {name}", flush=True)
                # Print full input arguments, indented
                inp_str = json.dumps(inp, indent=2)
                for line in inp_str.split("\n"):
                    print(f"      {line}", flush=True)

    elif etype == "user":
        # Tool results come back as "user" events
        content = event.get("message", {}).get("content", [])
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    is_err = block.get("is_error", False)
                    status = "ERROR" if is_err else "OK"
                    rc = block.get("content", "")
                    # Extract text from content (can be string or list)
                    if isinstance(rc, list):
                        rc = "\n".join(
                            i.get("text", "") for i in rc
                            if isinstance(i, dict) and i.get("type") == "text"
                        )
                    rc = str(rc)
                    # Show up to 500 chars of the result
                    preview = rc[:500]
                    truncated = "..." if len(rc) > 500 else ""
                    print(f"    [tool_result] {status}", flush=True)
                    for line in preview.split("\n"):
                        print(f"      {line}", flush=True)
                    if truncated:
                        print(f"      ... ({len(rc)} chars total)", flush=True)

    elif etype == "result":
        cost = event.get("total_cost_usd", 0)
        dur = event.get("duration_ms", 0)
        err = event.get("is_error", False)
        status = "ERROR" if err else "SUCCESS"
        print(f"    [result] {status}  {dur/1000:.1f}s  ${cost:.4f}", flush=True)


def run_claude_cli(
    prompt: str,
    model: str,
    workspace: str,
    mcp_config: Optional[Path] = None,
    append_system_prompt: Optional[str] = None,
    timeout: int = AGENT_TIMEOUT,
    max_turns: int = AGENT_MAX_TURNS,
) -> Dict[str, Any]:
    """
    Run Claude Code CLI and stream output.

    No timeout by default — LLM fine-tuning tasks can run for hours.
    max_turns controls how many tool-call turns Claude can take before stopping.

    Returns dict with: response, success, error, cost_usd, time_seconds,
                       raw_log_lines, tool_call_count, input_tokens, output_tokens
    """
    # NOTE: Prompt is piped via stdin because the positional prompt arg
    # hangs in non-TTY environments. Using --print + stdin works reliably.
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

    # Add MCP config for with_kb phase
    if mcp_config and mcp_config.exists():
        cmd.extend(["--mcp-config", str(mcp_config)])
        cmd.append("--strict-mcp-config")

    # Append sandbox system prompt
    if append_system_prompt:
        cmd.extend(["--append-system-prompt", append_system_prompt])

    env = _build_env()

    # Accumulators
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
        cmd, cwd=workspace,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        stdin=subprocess.PIPE,
        text=True, env=env,
    )

    # Write the prompt to stdin, then close to signal EOF
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

            # --- Live output to terminal ---
            _print_live_event(event)

            # Collect assistant text
            if etype == "assistant":
                for block in event.get("message", {}).get("content", []):
                    if block.get("type") == "text":
                        assistant_texts.append(block.get("text", ""))
                    elif block.get("type") == "tool_use":
                        tool_call_count += 1

            # Parse the final result event
            if etype == "result":
                result_text = event.get("result", "")
                total_cost = event.get("total_cost_usd", 0.0)
                is_error = event.get("is_error", False)
                usage = event.get("usage", {})
                input_tokens = usage.get("input_tokens", 0)
                output_tokens = usage.get("output_tokens", 0)

        # timeout=None means wait indefinitely (no timeout)
        proc.wait(timeout=timeout)
        stderr_thread.join(timeout=5)

    except subprocess.TimeoutExpired:
        # Safety net: only triggers if an explicit timeout was passed
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
        "error": "" if success else (result_text if is_error else f"exit code {proc.returncode}"),
        "cost_usd": total_cost,
        "time_seconds": round(elapsed, 2),
        "raw_log_lines": raw_lines,
        "tool_call_count": tool_call_count,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


# =============================================================================
# File helpers
# =============================================================================

def save_response_report(filepath: Path, response_text: str):
    """Save the agent's response as a markdown report."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text(response_text, encoding="utf-8")


def save_claude_log(filepath: Path, raw_log_lines: List[str]):
    """Save raw stream-json log and a human-readable .txt version."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text("\n".join(raw_log_lines), encoding="utf-8")

    # Human-readable version
    readable_path = filepath.with_suffix(".txt")
    readable_lines = _format_readable_log(raw_log_lines)
    readable_path.write_text("\n".join(readable_lines), encoding="utf-8")


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

    # Pass 2: format events
    lines = []
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
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        tool_use_id = block.get("tool_use_id", "")
                        is_err = block.get("is_error", False)
                        origin = tool_id_to_name.get(tool_use_id, "?")
                        status = "ERROR" if is_err else "OK"
                        rc = block.get("content", "")
                        if isinstance(rc, list):
                            rc = "\n".join(
                                i.get("text", "") for i in rc
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
            lines.append(f"RESULT: {'ERROR' if event.get('is_error') else 'SUCCESS'}")
            lines.append(f"  Duration: {duration/1000:.1f}s  Cost: ${cost:.4f}")
            lines.append("=" * 70)

    return lines


def load_results() -> List[Dict]:
    """Load existing benchmark results for resuming."""
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return []


def save_results(results: List[Dict]):
    """Save benchmark results to disk."""
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)


# =============================================================================
# Single-phase runner (baseline or with_kb)
# =============================================================================

def _run_one_phase(
    label: str,
    task_text: str,
    model: str,
    workspace: str,
    task_ws: Path,
    mcp_config: Optional[Path] = None,
    timeout: int = AGENT_TIMEOUT,
) -> Dict[str, Any]:
    """
    Run one agent phase in an isolated /tmp sandbox.

    1. Create empty temp dir under /tmp
    2. Run agent with cwd = that temp dir
    3. Copy generated files back to real workspace
    4. Clean up temp dir
    """
    print(f"\n  --- {label} Starting ---", flush=True)
    sandbox_dir = tempfile.mkdtemp(prefix=f"bench_{label}_")
    print(f"  Sandbox: {sandbox_dir}", flush=True)

    try:
        # Copy the pre-built evaluation helper into the sandbox
        # so the agent can use it directly without writing its own
        if EVAL_HELPER_SCRIPT.exists():
            shutil.copy2(EVAL_HELPER_SCRIPT, Path(sandbox_dir) / "evaluate_model.py")
            print(f"  Copied evaluate_model.py to sandbox", flush=True)

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

    # Save logs
    prefix = label.lower().replace(" ", "_")
    save_response_report(task_ws / f"{prefix}_response.md", metrics["response"])
    save_claude_log(task_ws / f"{prefix}_claude.log", metrics["raw_log_lines"])

    print(
        f"  --- {label} Done: {metrics['time_seconds']}s, "
        f"${metrics['cost_usd']:.4f}, {metrics['tool_call_count']} tool calls, "
        f"{metrics['input_tokens']}+{metrics['output_tokens']} tokens ---",
        flush=True,
    )
    return metrics


# =============================================================================
# Run the proposal benchmark (baseline -> with_kb)
# =============================================================================

def run_proposal_benchmark():
    """
    Main pipeline: run baseline & with_kb using the same proposal.

    - Baseline uses proposal.md without MCP
    - With KB uses proposal.md with Leeroopedia MCP connected
    """
    # Read proposal file
    if not PROPOSAL_FILE.exists():
        print(f"ERROR: {PROPOSAL_FILE} not found")
        sys.exit(1)

    proposal = PROPOSAL_FILE.read_text(encoding="utf-8")
    print(f"Loaded proposal.md ({len(proposal)} chars)")

    # Evaluation criteria based on the four tasks in the proposal
    eval_criteria = [
        "Task 1 SFT: Implements SFT fine-tuning with LoRA on ultrachat_200k using TRL with distributed training",
        "Task 1 DPO: Implements DPO training on argilla/ultrafeedback-binarized-preferences-cleaned with precomputed ref logprobs",
        "Task 2 Merge: Correctly merges LoRA adapters into base model, saved to ./merged_model/",
        "Task 3 Serve: Deploys merged model via vLLM with OpenAI-compatible chat API and reports throughput at batch sizes 1 and 4",
        "Task 4 Eval: Reports IFEval strict-prompt accuracy for the trained model using evaluate_model.py",
    ]

    # The preamble is critical — it tells Claude to EXECUTE, not just write code.
    criteria_list = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(eval_criteria))

    execution_preamble = (
        "You must implement AND EXECUTE the following ML pipeline proposal. "
        "Do NOT just write scripts — you must actually run them using Bash and "
        "report the real output. Write each script, then immediately execute it. "
        "If a step fails, debug and fix it until it succeeds. "
        "IMPORTANT: The file evaluate_model.py is already in your workspace. "
        "Use it exactly as shown in the proposal for Task 4 evaluation. "
        "Do NOT stop until the full pipeline has run end-to-end and you have "
        "printed the final benchmark numbers (IFEval score).\n\n"
    )

    # Both phases get the same prompt (same proposal.md)
    agent_prompt = (
        f"{execution_preamble}"
        f"{proposal}\n\n"
        f"SUCCESS CRITERIA (your implementation will be evaluated against these):\n"
        f"{criteria_list}"
    )

    # Create workspace directories under leeroopedia_llm_training/workspaces/
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
        task_text=agent_prompt,
        model=AGENT_MODEL,
        workspace=str(baseline_ws),
        task_ws=task_ws,
        mcp_config=None,
    )

    # --- Phase 2: With KB (Leeroopedia MCP) ---
    print(f"\n{'='*60}")
    print("Phase 2: With KB (Leeroopedia MCP)")
    print(f"{'='*60}")
    kb_metrics = _run_one_phase(
        label="with_kb",
        task_text=agent_prompt,
        model=AGENT_MODEL,
        workspace=str(kb_ws),
        task_ws=task_ws,
        mcp_config=MCP_CONFIG_PATH,
    )

    # Build result dict
    result = {
        "task": "proposal.md - LLM Post-Training & Deployment Pipeline",
        "proposal_file": str(PROPOSAL_FILE),
        "evaluation_criteria": eval_criteria,
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

    save_results([result])

    # Print summary
    print(f"\n{'='*60}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*60}")
    print(f"  Baseline:")
    print(f"    time: {baseline_metrics['time_seconds']}s, cost: ${baseline_metrics['cost_usd']:.4f}")
    print(f"    tool calls: {baseline_metrics['tool_call_count']}, "
          f"tokens: {baseline_metrics['input_tokens']}+{baseline_metrics['output_tokens']}")
    print(f"  With KB:")
    print(f"    time: {kb_metrics['time_seconds']}s, cost: ${kb_metrics['cost_usd']:.4f}")
    print(f"    tool calls: {kb_metrics['tool_call_count']}, "
          f"tokens: {kb_metrics['input_tokens']}+{kb_metrics['output_tokens']}")
    print(f"  Results saved to: {RESULTS_FILE}")
    print(f"{'='*60}")

    return result


# =============================================================================
# Hello world smoke test
# =============================================================================

def run_hello_world_test():
    """Verify the full pipeline works with a trivial task."""
    print("=" * 60)
    print("HELLO WORLD TEST - verifying full pipeline")
    print("=" * 60)

    task = (
        "Create a Python script called hello.py that prints 'Hello, World!' "
        "and includes a function called add(a, b) that returns the sum."
    )
    eval_criteria = [
        "The file hello.py exists in the workspace",
        "The script prints 'Hello, World!' when run",
        "There is a function named 'add' that returns the sum of two arguments",
    ]

    criteria_list = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(eval_criteria))
    agent_prompt = f"{task}\n\nSUCCESS CRITERIA:\n{criteria_list}"

    task_ws = WORKSPACES_DIR / "_test_hello_world"
    baseline_ws = task_ws / "baseline"
    kb_ws = task_ws / "with_kb"
    baseline_ws.mkdir(parents=True, exist_ok=True)
    kb_ws.mkdir(parents=True, exist_ok=True)

    # Phase 1: Baseline
    baseline_metrics = _run_one_phase(
        label="baseline", task_text=agent_prompt, model=AGENT_MODEL,
        workspace=str(baseline_ws), task_ws=task_ws, mcp_config=None,
    )

    # Phase 2: With KB
    kb_metrics = _run_one_phase(
        label="with_kb", task_text=agent_prompt, model=AGENT_MODEL,
        workspace=str(kb_ws), task_ws=task_ws, mcp_config=MCP_CONFIG_PATH,
    )

    result = {
        "task": "hello_world_test",
        "evaluation_criteria": eval_criteria,
        "baseline": {
            "time_seconds": baseline_metrics["time_seconds"],
            "cost_usd": baseline_metrics["cost_usd"],
            "tool_call_count": baseline_metrics["tool_call_count"],
        },
        "with_kb": {
            "time_seconds": kb_metrics["time_seconds"],
            "cost_usd": kb_metrics["cost_usd"],
            "tool_call_count": kb_metrics["tool_call_count"],
        },
    }
    save_results([result])

    print(f"\n{'='*60}")
    print("TEST COMPLETE")
    print(f"  Baseline: {baseline_metrics['time_seconds']}s, ${baseline_metrics['cost_usd']:.4f}")
    print(f"  With KB:  {kb_metrics['time_seconds']}s, ${kb_metrics['cost_usd']:.4f}")
    print(f"  Results: {RESULTS_FILE}")
    print(f"{'='*60}")

    return result


# =============================================================================
# Main entry point
# =============================================================================

def _parse_args():
    """Parse CLI arguments. --bedrock switches to Bedrock mode."""
    parser = argparse.ArgumentParser(
        description="Leeroopedia LLM Training Benchmark Runner"
    )
    parser.add_argument(
        "--bedrock",
        action="store_true",
        default=False,
        help="Use AWS Bedrock instead of Anthropic API (default: Anthropic API)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        default=False,
        help="Run a hello-world smoke test instead of the full benchmark",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # Set global mode based on CLI flag
    USE_BEDROCK = args.bedrock
    AGENT_MODEL = BEDROCK_MODEL if USE_BEDROCK else ANTHROPIC_MODEL

    mode_label = "Bedrock" if USE_BEDROCK else "Anthropic API"
    print(f"Mode: {mode_label}  Model: {AGENT_MODEL}", flush=True)

    if args.test:
        run_hello_world_test()
    else:
        run_proposal_benchmark()
