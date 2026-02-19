"""
KernelBench Benchmark Runner

Runs the GPU kernel optimization task through Claude Code twice:
  1. Baseline: no knowledge base (empty MCP config)
  2. With KB:  Leeroopedia MCP enabled

Both runs use the same proposal (kernelbench_proposal.md). The only
difference is whether Leeroopedia MCP tools are available.

Usage:
  python run_benchmark.py                # run with Anthropic API key (default)
  python run_benchmark.py --bedrock      # run with AWS Bedrock
  python run_benchmark.py --baseline     # run baseline only
  python run_benchmark.py --with-kb      # run with_kb only
  python run_benchmark.py --generate     # run fixture generation first
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

# --- Paths ---
BENCH_DIR = Path(__file__).resolve().parent          # .../kernelbench/
PARENT_DIR = BENCH_DIR.parent                        # parent of kernelbench/
# MCP configs and .env live alongside run_kernelbench.py inside kernelbench/
MCP_CONFIG_PATH = BENCH_DIR / "leeroopedia_mcp_config.json"
# Empty MCP config ensures baseline runs have zero MCP servers,
# even if the user has global MCP servers configured.
EMPTY_MCP_CONFIG_PATH = BENCH_DIR / "empty_mcp_config.json"
PROPOSAL_FILE = BENCH_DIR / "kernelbench_proposal.md"
FIXTURES_DIR = BENCH_DIR / "kernelbench_fixtures"
TASK_WORKSPACE = BENCH_DIR / "workspaces"
RESULTS_FILE = TASK_WORKSPACE / "benchmark_results.json"

# Load credentials from .env in the kernelbench directory (fallback to parent)
load_dotenv(BENCH_DIR / ".env")
load_dotenv(PARENT_DIR / ".env")

# --- Config ---
# Anthropic API model ID (default)
ANTHROPIC_MODEL = "claude-opus-4-5-20251101"
# Bedrock model ID (used with --bedrock)
BEDROCK_MODEL = os.getenv("ANTHROPIC_MODEL", "us.anthropic.claude-opus-4-6-v1")

# Resolved at runtime based on --bedrock flag
USE_BEDROCK = False
AGENT_MODEL = ANTHROPIC_MODEL
AGENT_TIMEOUT = 2400  # 40 min per agent (kernel writing + compilation)

SANDBOX_SYSTEM_PROMPT = (
    "CRITICAL RULE: You must ONLY read, write, and explore files within your "
    "current working directory. Do NOT use Bash, Read, Glob, Grep, or any tool "
    "to access files or directories outside your current working directory. "
    "Do NOT navigate to parent directories or any other path on the filesystem. "
    "If you need information about the project or framework, rely solely on your "
    "training knowledge or any MCP tools available to you."
)


# =============================================================================
# Claude Code CLI runner (same pattern as run_demo_task2.py)
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
        env["AWS_REGION"] = os.getenv("AWS_REGION", "us-east-1")
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


def run_claude_cli(
    prompt: str, model: str, workspace: str,
    mcp_config: Optional[Path] = None,
    append_system_prompt: Optional[str] = None,
    timeout: int = AGENT_TIMEOUT,
) -> Dict[str, Any]:
    """Run Claude Code CLI and stream output. Returns metrics dict."""
    cmd = [
        "claude", "-p", prompt, "--model", model,
        "--output-format", "stream-json", "--verbose",
        "--dangerously-skip-permissions",
    ]
    # Always use --strict-mcp-config to block global MCP servers.
    # For with_kb: loads Leeroopedia MCP. For baseline: loads empty config.
    effective_mcp = mcp_config if (mcp_config and mcp_config.exists()) else EMPTY_MCP_CONFIG_PATH
    cmd.extend(["--mcp-config", str(effective_mcp)])
    cmd.append("--strict-mcp-config")
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
        cmd, cwd=workspace, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, env=env,
    )

    stderr_lines: List[str] = []
    def _drain():
        for line in proc.stderr:
            stderr_lines.append(line)
    t = threading.Thread(target=_drain, daemon=True)
    t.start()

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

            # Live logging
            if etype == "system" and event.get("subtype") == "init":
                m = event.get("model", "?")
                tools = event.get("tools", [])
                mcp_tools = [x for x in tools if x.startswith("mcp__")]
                print(f"    [init] model={m}  tools={len(tools)} ({len(mcp_tools)} MCP)", flush=True)
            elif etype == "assistant":
                for block in event.get("message", {}).get("content", []):
                    btype = block.get("type")
                    if btype == "text":
                        txt = block.get("text", "").strip()
                        assistant_texts.append(txt)
                        preview = txt[:200].replace("\n", " ")
                        if preview:
                            print(f"    [text] {preview}{'...' if len(txt) > 200 else ''}", flush=True)
                    elif btype == "tool_use":
                        tool_call_count += 1
                        tname = block.get("name", "?")
                        tinput = block.get("input", {})
                        if tname == "Bash":
                            print(f"    [tool] {tname}: $ {tinput.get('command', '')[:120]}", flush=True)
                        elif tname.startswith("mcp__"):
                            args_s = " ".join(f"{k}={str(v)[:60]}" for k, v in tinput.items())
                            print(f"    [mcp]  {tname}: {args_s[:150]}", flush=True)
                        else:
                            print(f"    [tool] {tname}", flush=True)
            elif etype == "result":
                result_text = event.get("result", "")
                total_cost = event.get("total_cost_usd", 0.0)
                is_error = event.get("is_error", False)
                usage = event.get("usage", {})
                input_tokens = usage.get("input_tokens", 0)
                output_tokens = usage.get("output_tokens", 0)
        proc.wait(timeout=timeout)
        t.join(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        t.join(timeout=5)
        is_error = True

    elapsed = time.time() - start

    # Fallback: sum per-turn tokens
    if input_tokens == 0 or output_tokens == 0:
        ci, co = 0, 0
        for raw in raw_lines:
            try:
                ev = json.loads(raw)
                if ev.get("type") == "assistant":
                    u = ev.get("message", {}).get("usage", {})
                    ci += u.get("input_tokens", 0)
                    co += u.get("output_tokens", 0)
            except json.JSONDecodeError:
                continue
        input_tokens = input_tokens or ci
        output_tokens = output_tokens or co

    response = result_text or "\n".join(assistant_texts)
    success = proc.returncode == 0 and not is_error
    return {
        "response": response, "success": success,
        "error": "" if success else (result_text if is_error else f"exit {proc.returncode}"),
        "cost_usd": total_cost, "time_seconds": round(elapsed, 2),
        "raw_log_lines": raw_lines, "tool_call_count": tool_call_count,
        "input_tokens": input_tokens, "output_tokens": output_tokens,
    }


# =============================================================================
# File helpers
# =============================================================================

def save_text(filepath: Path, text: str):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text(text, encoding="utf-8")

def save_log(filepath: Path, raw_lines: List[str]):
    """Save raw .log and human-readable .txt."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text("\n".join(raw_lines), encoding="utf-8")
    readable = filepath.with_suffix(".txt")
    tool_map: Dict[str, str] = {}
    parsed = []
    for raw in raw_lines:
        if not raw.strip():
            continue
        try:
            ev = json.loads(raw)
        except json.JSONDecodeError:
            continue
        parsed.append(ev)
        if ev.get("type") == "assistant":
            for b in ev.get("message", {}).get("content", []):
                if b.get("type") == "tool_use" and b.get("id"):
                    tool_map[b["id"]] = b.get("name", "?")
    out = []
    for ev in parsed:
        et = ev.get("type", "")
        if et == "system" and ev.get("subtype") == "init":
            out.append(f"SESSION INIT -- Model: {ev.get('model','?')}")
            out.append("")
        elif et == "assistant":
            for b in ev.get("message", {}).get("content", []):
                if b.get("type") == "text":
                    out.append("[TEXT] " + b.get("text", "").strip()[:500])
                    out.append("")
                elif b.get("type") == "tool_use":
                    out.append(f"[TOOL] {b.get('name','?')}")
                    out.append("")
        elif et == "result":
            out.append(f"RESULT: cost=${ev.get('total_cost_usd',0):.4f}")
    readable.write_text("\n".join(out), encoding="utf-8")


# =============================================================================
# Agent phase runner
# =============================================================================

def run_agent_phase(label: str, proposal_file: Path, mcp_config: Optional[Path] = None):
    """Run one agent phase in an isolated /tmp sandbox with fixtures."""
    prompt = proposal_file.read_text(encoding="utf-8")
    dest_ws = TASK_WORKSPACE / label
    dest_ws.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  PHASE: {label.upper()} | Proposal: {proposal_file.name}")
    print(f"  MCP: {'YES' if mcp_config else 'NO'}")
    print(f"{'='*60}", flush=True)

    # Create sandbox and copy fixtures into it
    sandbox = tempfile.mkdtemp(prefix=f"kernelbench_{label}_")
    print(f"  Sandbox: {sandbox}", flush=True)

    # Set up fixtures in sandbox using symlinks for large data dirs
    # to avoid copying tens of GB of tensor files.
    sandbox_fixtures = Path(sandbox) / "fixtures"
    sandbox_fixtures.mkdir(parents=True, exist_ok=True)

    # Symlink the large data directories (inputs, expected_outputs, reference_timing, weights)
    for subdir in ["inputs", "expected_outputs", "reference_timing", "problems", "weights"]:
        src = FIXTURES_DIR / subdir
        if src.exists():
            os.symlink(src, sandbox_fixtures / subdir)

    # Copy small files directly (manifest.json, evaluate.py)
    for fname in ["manifest.json", "evaluate.py"]:
        src = FIXTURES_DIR / fname
        if src.exists():
            shutil.copy2(src, sandbox_fixtures / fname)

    file_count = len(list(sandbox_fixtures.rglob("*")))
    print(f"  Prepared fixtures ({file_count} entries, large dirs symlinked)", flush=True)

    try:
        metrics = run_claude_cli(
            prompt=prompt, model=AGENT_MODEL, workspace=sandbox,
            mcp_config=mcp_config, append_system_prompt=SANDBOX_SYSTEM_PROMPT,
            timeout=AGENT_TIMEOUT,
        )
        # Copy results from sandbox to workspace
        for item in Path(sandbox).iterdir():
            if item.name.startswith(".claude"):
                continue
            dest = dest_ws / item.name
            if item.is_dir():
                shutil.copytree(item, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dest)
    finally:
        shutil.rmtree(sandbox, ignore_errors=True)

    save_text(TASK_WORKSPACE / f"{label}_response.md", metrics["response"])
    save_log(TASK_WORKSPACE / f"{label}_claude.log", metrics["raw_log_lines"])

    print(
        f"  Done: {metrics['time_seconds']}s, ${metrics['cost_usd']:.4f}, "
        f"{metrics['tool_call_count']} tools, "
        f"{metrics['input_tokens']}+{metrics['output_tokens']} tokens",
        flush=True,
    )
    return metrics


# =============================================================================
# Fallback: re-run evaluate.py if agent didn't
# =============================================================================

def run_evaluate_fallback(workspace: Path) -> Optional[Dict]:
    """If agent didn't produce results.json, try running evaluate.py."""
    results_json = workspace / "results.json"
    if results_json.exists():
        try:
            with open(results_json) as f:
                return json.load(f)
        except json.JSONDecodeError:
            pass

    # Try to find solutions and run evaluate.py
    solutions_dir = workspace / "solutions"
    evaluate_py = workspace / "fixtures" / "evaluate.py"
    if not solutions_dir.exists() or not evaluate_py.exists():
        return None

    print(f"  Running evaluate.py fallback for {workspace.name}...", flush=True)
    try:
        result = subprocess.run(
            ["python", str(evaluate_py), "--solutions-dir", str(solutions_dir),
             "--output", str(results_json)],
            cwd=str(workspace), capture_output=True, text=True, timeout=600,
        )
        if result.returncode == 0 and results_json.exists():
            with open(results_json) as f:
                return json.load(f)
        else:
            print(f"  evaluate.py failed: {result.stderr[:200]}", flush=True)
    except Exception as e:
        print(f"  evaluate.py error: {e}", flush=True)
    return None


# =============================================================================
# Main
# =============================================================================

def _parse_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="KernelBench Benchmark Runner"
    )
    parser.add_argument(
        "--bedrock",
        action="store_true",
        default=False,
        help="Use AWS Bedrock instead of Anthropic API (default: Anthropic API)",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        default=False,
        help="Run fixture generation before the benchmark",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        default=False,
        help="Run baseline phase only",
    )
    parser.add_argument(
        "--with-kb",
        action="store_true",
        default=False,
        help="Run with-KB phase only",
    )
    return parser.parse_args()


def main():
    global USE_BEDROCK, AGENT_MODEL

    args = _parse_args()

    # Set global mode based on CLI flag
    USE_BEDROCK = args.bedrock
    AGENT_MODEL = BEDROCK_MODEL if USE_BEDROCK else ANTHROPIC_MODEL

    mode_label = "Bedrock" if USE_BEDROCK else "Anthropic API"
    print(f"Mode: {mode_label}  Model: {AGENT_MODEL}", flush=True)

    # Handle --generate: run fixture generation first
    if args.generate:
        print("Generating fixtures...")
        subprocess.run(
            [sys.executable, str(BENCH_DIR / "generate_kernelbench_fixtures.py"),
             "--fixtures-dir", str(FIXTURES_DIR)],
            check=True,
        )
        if not args.baseline and not args.with_kb:
            print("Fixtures generated. Run again without --generate to run agents.")
            return

    # Check fixtures exist
    if not FIXTURES_DIR.exists() or not (FIXTURES_DIR / "manifest.json").exists():
        print(f"ERROR: Fixtures not found at {FIXTURES_DIR}")
        print(f"Run: python generate_kernelbench_fixtures.py")
        print(f"  or: python run_benchmark.py --generate")
        sys.exit(1)

    # Determine which phases to run
    run_bl = args.baseline or (not args.baseline and not args.with_kb)
    run_kb = args.with_kb or (not args.baseline and not args.with_kb)

    TASK_WORKSPACE.mkdir(parents=True, exist_ok=True)

    results = {}
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            results = json.load(f)

    if run_bl:
        m = run_agent_phase("baseline", PROPOSAL_FILE, mcp_config=None)
        results.update(baseline_time=m["time_seconds"], baseline_cost=m["cost_usd"],
                       baseline_tool_calls=m["tool_call_count"], baseline_success=m["success"])
        eval_data = run_evaluate_fallback(TASK_WORKSPACE / "baseline")
        if eval_data:
            results["baseline_eval"] = eval_data
        save_text(RESULTS_FILE, json.dumps(results, indent=2))

    if run_kb:
        m = run_agent_phase("with_kb", PROPOSAL_FILE, mcp_config=MCP_CONFIG_PATH)
        results.update(with_kb_time=m["time_seconds"], with_kb_cost=m["cost_usd"],
                       with_kb_tool_calls=m["tool_call_count"], with_kb_success=m["success"])
        eval_data = run_evaluate_fallback(TASK_WORKSPACE / "with_kb")
        if eval_data:
            results["with_kb_eval"] = eval_data
        save_text(RESULTS_FILE, json.dumps(results, indent=2))

    print(f"\n  Results saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
