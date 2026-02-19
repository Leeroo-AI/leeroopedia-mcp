"""
Demo Task 2 Benchmark Runner: Self-Evolving RAG

Runs the self-evolving RAG task through Claude Code twice:
  1. With KB:  same prompt + Leeroopedia MCP tools available
  2. Baseline: same prompt, no knowledge base tools

Usage:
  python run_benchmark.py                # run with Anthropic API key (default)
  python run_benchmark.py --bedrock      # run with AWS Bedrock
  python run_benchmark.py --baseline     # run baseline only
  python run_benchmark.py --with-kb      # run with_kb only
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
BENCH_DIR = Path(__file__).resolve().parent
EVAL_HARNESS_PATH = BENCH_DIR / "eval_harness.py"
EVAL_QUERIES_PATH = BENCH_DIR / "eval_queries.json"
MCP_CONFIG_PATH = BENCH_DIR / "leeroopedia_mcp_config.json"
# Empty MCP config ensures baseline runs have zero MCP servers,
# even if the user has global MCP servers configured.
EMPTY_MCP_CONFIG_PATH = BENCH_DIR / "empty_mcp_config.json"
PROPOSAL_FILE = BENCH_DIR / "proposal.md"
TASK_WORKSPACE = BENCH_DIR / "workspaces" / "task2_self_evolving_rag"
RESULTS_FILE = TASK_WORKSPACE / "benchmark_results.json"

# Load credentials from the benchmark .env
load_dotenv(BENCH_DIR / ".env")

# --- Config ---
# Anthropic API model ID (default)
ANTHROPIC_MODEL_ID = "claude-opus-4-5-20251101"
# Bedrock model ID (used with --bedrock)
BEDROCK_MODEL = os.getenv("ANTHROPIC_MODEL", "us.anthropic.claude-opus-4-6-v1")

# Resolved at runtime based on --bedrock flag
USE_BEDROCK = False
AGENT_MODEL = ANTHROPIC_MODEL_ID
AGENT_TIMEOUT = 1800  # 30 min per agent run

SANDBOX_SYSTEM_PROMPT = (
    "CRITICAL RULE: You must ONLY read, write, and explore files within your "
    "current working directory. Do NOT use Bash, Read, Glob, Grep, or any tool "
    "to access files or directories outside your current working directory. "
    "Do NOT navigate to parent directories or any other path on the filesystem. "
    "If you need information about the project or framework, rely solely on your "
    "training knowledge or any MCP tools available to you."
)


# =============================================================================
# Claude Code CLI runner
# =============================================================================

def _build_env() -> Dict[str, str]:
    """Build environment for Claude Code CLI.

    Two modes:
      - Anthropic API (default): uses ANTHROPIC_API_KEY from .env.
      - Bedrock (--bedrock flag): sets CLAUDE_CODE_USE_BEDROCK=1 and
        passes AWS credentials.
    """
    env = os.environ.copy()

    if USE_BEDROCK:
        # Bedrock mode: authenticate via AWS bearer token
        env["CLAUDE_CODE_USE_BEDROCK"] = "1"
        env["AWS_REGION"] = os.getenv("AWS_REGION", "us-east-1")
    else:
        # Anthropic API mode (default): authenticate via API key
        env.pop("CLAUDE_CODE_USE_BEDROCK", None)
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            print("ERROR: ANTHROPIC_API_KEY not set in .env. "
                  "Use --bedrock flag for Bedrock mode.", flush=True)
            sys.exit(1)
        env["ANTHROPIC_API_KEY"] = api_key

    # Pass through keys so the generated service can use them
    for key in ("OPENAI_API_KEY", "HF_TOKEN"):
        val = os.getenv(key)
        if val:
            env[key] = val
    return env


def run_claude_cli(
    prompt: str,
    model: str,
    workspace: str,
    mcp_config: Optional[Path] = None,
    append_system_prompt: Optional[str] = None,
    timeout: int = AGENT_TIMEOUT,
) -> Dict[str, Any]:
    """Run Claude Code CLI and stream output. Returns metrics dict."""
    cmd = [
        "claude", "-p", prompt,
        "--model", model,
        "--output-format", "stream-json",
        "--verbose",
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
    tool_id_to_name: Dict[str, str] = {}  # track tool_use_id -> tool name for MCP detection
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

            # --- Live log: print events to terminal as they stream ---
            if etype == "system" and event.get("subtype") == "init":
                m = event.get("model", "?")
                tools = event.get("tools", [])
                mcp = [t for t in tools if t.startswith("mcp__")]
                print(f"    [init] model={m}  tools={len(tools)} ({len(mcp)} MCP)", flush=True)

            elif etype == "assistant":
                for block in event.get("message", {}).get("content", []):
                    if block.get("type") == "text":
                        text = block.get("text", "").strip()
                        assistant_texts.append(text)
                        # Show first 200 chars of each text block
                        preview = text[:200].replace("\n", " ")
                        if preview:
                            print(f"    [text] {preview}{'...' if len(text) > 200 else ''}", flush=True)
                    elif block.get("type") == "tool_use":
                        tool_call_count += 1
                        tname = block.get("name", "?")
                        tinput = block.get("input", {})
                        # Track tool_use_id for matching results to MCP tools
                        tid = block.get("id", "")
                        if tid:
                            tool_id_to_name[tid] = tname
                        # Show tool call — full output for MCP, compact for others
                        if tname == "Bash":
                            cmd_str = tinput.get("command", "")[:120]
                            print(f"    [tool] {tname}: $ {cmd_str}", flush=True)
                        elif tname in ("Write", "Edit", "Read"):
                            fpath = tinput.get("file_path", tinput.get("path", "?"))
                            print(f"    [tool] {tname}: {fpath}", flush=True)
                        elif tname.startswith("mcp__"):
                            # Full untruncated MCP tool input
                            print(f"    [mcp]  {tname}:", flush=True)
                            for k, v in tinput.items():
                                print(f"           {k}={v}", flush=True)
                        else:
                            print(f"    [tool] {tname}", flush=True)

            elif etype == "user":
                # Tool results — show full output for MCP, brief for others
                content = event.get("message", {}).get("content", [])
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "tool_result":
                            is_err = block.get("is_error", False)
                            tid = block.get("tool_use_id", "")
                            rc = block.get("content", "")
                            if isinstance(rc, list):
                                rc = "\n".join(i.get("text", "") for i in rc if isinstance(i, dict))
                            else:
                                rc = str(rc)
                            status = "ERR" if is_err else "ok"
                            # Check if this result came from an MCP tool
                            origin = tool_id_to_name.get(tid, "")
                            if origin.startswith("mcp__"):
                                # Full untruncated MCP result
                                print(f"    [result] {status} ({origin}):", flush=True)
                                for rline in rc.split("\n"):
                                    print(f"             {rline}", flush=True)
                            else:
                                preview = rc.replace(chr(10), " ")[:120]
                                print(f"    [result] {status}: {preview}", flush=True)

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
    """Save raw .log and a human-readable .txt version."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text("\n".join(raw_lines), encoding="utf-8")

    # Build human-readable version
    readable_path = filepath.with_suffix(".txt")
    tool_id_to_name: Dict[str, str] = {}
    parsed: List[dict] = []
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
                    tool_id_to_name[b["id"]] = b.get("name", "?")

    lines = []
    LIMIT = 1500
    for ev in parsed:
        etype = ev.get("type", "")

        if etype == "system" and ev.get("subtype") == "init":
            model = ev.get("model", "?")
            tools = ev.get("tools", [])
            mcp = [t for t in tools if t.startswith("mcp__")]
            lines.append("=" * 70)
            lines.append(f"SESSION INIT — Model: {model}")
            lines.append(f"  Tools: {len(tools)} total, {len(mcp)} MCP")
            if mcp:
                lines.append(f"  MCP tools: {', '.join(mcp)}")
            lines.append("=" * 70)
            lines.append("")

        elif etype == "assistant":
            for b in ev.get("message", {}).get("content", []):
                bt = b.get("type", "")
                if bt == "text":
                    text = b.get("text", "").strip()
                    if text:
                        lines.append("[ASSISTANT TEXT]")
                        for tl in text.split("\n"):
                            lines.append(f"  {tl}")
                        lines.append("")
                elif bt == "tool_use":
                    name = b.get("name", "?")
                    inp = b.get("input", {})
                    lines.append(f"[TOOL CALL] {name}")
                    if name == "Bash":
                        lines.append(f"  $ {inp.get('command', '')[:300]}")
                    elif name in ("Write", "Read", "Edit"):
                        fp = inp.get("file_path", inp.get("path", "?"))
                        lines.append(f"  File: {fp}")
                        if name == "Write":
                            lines.append(f"  Content: ({len(inp.get('content', ''))} chars)")
                    elif name.startswith("mcp__"):
                        # Full untruncated MCP input in log file
                        for k, v in inp.items():
                            lines.append(f"  {k}: {v}")
                    else:
                        s = json.dumps(inp, indent=2)[:500]
                        for il in s.split("\n"):
                            lines.append(f"  {il}")
                    lines.append("")

        elif etype == "user":
            content = ev.get("message", {}).get("content", [])
            if isinstance(content, list):
                for b in content:
                    if isinstance(b, dict) and b.get("type") == "tool_result":
                        tid = b.get("tool_use_id", "")
                        is_err = b.get("is_error", False)
                        origin = tool_id_to_name.get(tid, "?")
                        rc = b.get("content", "")
                        if isinstance(rc, list):
                            rc = "\n".join(i.get("text", "") for i in rc if isinstance(i, dict))
                        else:
                            rc = str(rc)
                        status = "ERROR" if is_err else "OK"
                        lines.append(f"[TOOL RESULT] {origin} -> {status}")
                        # Full untruncated output for MCP results,
                        # truncated for non-MCP to keep log manageable
                        if origin.startswith("mcp__"):
                            for rl in rc.split("\n"):
                                lines.append(f"    {rl}")
                        else:
                            preview = rc[:LIMIT]
                            for rl in preview.split("\n"):
                                lines.append(f"    {rl}")
                            if len(rc) > LIMIT:
                                lines.append(f"    ... ({len(rc)} chars total)")
                        lines.append("")

        elif etype == "result":
            cost = ev.get("total_cost_usd", 0)
            is_err = ev.get("is_error", False)
            duration = ev.get("duration_ms", 0)
            lines.append("=" * 70)
            lines.append(f"RESULT: {'ERROR' if is_err else 'SUCCESS'}")
            lines.append(f"  Duration: {duration/1000:.1f}s | Cost: ${cost:.4f}")
            lines.append("=" * 70)

    readable_path.write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# Evaluation data preparation (runs BEFORE any agent)
# =============================================================================

def prepare_eval_data() -> Path:
    """
    Pre-generate eval_queries.json from the RAGBench techqa dataset.

    This ensures both agents (baseline and with_kb) evaluate against the
    exact same query sets with the exact same ground-truth document texts.
    The file is dropped into each agent's sandbox so they must use it.

    Returns the path to the generated eval_queries.json.
    """
    if EVAL_QUERIES_PATH.exists():
        print(f"  eval_queries.json already exists, reusing: {EVAL_QUERIES_PATH}")
        data = json.loads(EVAL_QUERIES_PATH.read_text(encoding="utf-8"))
        print(f"  Evolution queries: {data['evolution_query_count']}, "
              f"Held-out queries: {data['held_out_query_count']}, "
              f"Corpus docs: {data['corpus_doc_count']}")
        return EVAL_QUERIES_PATH

    print("  Generating eval_queries.json from RAGBench techqa dataset...")
    from datasets import load_dataset

    ds = load_dataset("rungalileo/ragbench", "techqa")

    # Extract unique documents from all splits — deterministic ordering by
    # content hash so the corpus is identical regardless of iteration order.
    doc_text_set = set()
    for split_name in ds.keys():
        for example in ds[split_name]:
            for doc in example.get("documents", []):
                if doc:
                    doc_text_set.add(doc)

    corpus_doc_count = len(doc_text_set)
    print(f"  Unique corpus documents: {corpus_doc_count}")

    # Build evolution queries from train split
    evolution_queries = []
    for i, example in enumerate(ds["train"]):
        evolution_queries.append({
            "query_id": f"evo_{i:04d}",
            "question": example["question"],
            "ground_truth_doc_texts": example.get("documents", []),
            "reference_answer": example.get("response", ""),
        })

    # Build held-out queries from test split (ALL examples, no subsampling)
    held_out_queries = []
    for i, example in enumerate(ds["test"]):
        held_out_queries.append({
            "query_id": f"held_{i:04d}",
            "question": example["question"],
            "ground_truth_doc_texts": example.get("documents", []),
            "reference_answer": example.get("response", ""),
        })

    data = {
        "evolution_queries": evolution_queries,
        "held_out_queries": held_out_queries,
        "corpus_doc_count": corpus_doc_count,
        "evolution_query_count": len(evolution_queries),
        "held_out_query_count": len(held_out_queries),
    }

    EVAL_QUERIES_PATH.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    print(f"  Wrote eval_queries.json: {len(evolution_queries)} evolution, "
          f"{len(held_out_queries)} held-out queries")
    return EVAL_QUERIES_PATH


# =============================================================================
# Agent phase runner
# =============================================================================

def run_agent_phase(label: str, proposal_file: Path, mcp_config: Optional[Path] = None):
    """Run one agent phase in an isolated /tmp sandbox."""
    prompt = proposal_file.read_text(encoding="utf-8")
    dest_ws = TASK_WORKSPACE / label
    dest_ws.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  PHASE: {label.upper()} | Proposal: {proposal_file.name}")
    print(f"  MCP: {'YES' if mcp_config else 'NO'}")
    print(f"{'='*60}", flush=True)

    sandbox = tempfile.mkdtemp(prefix=f"demo2_{label}_")
    print(f"  Sandbox: {sandbox}", flush=True)

    # Copy eval harness + queries into sandbox so agents use standardized metrics
    shutil.copy2(EVAL_HARNESS_PATH, sandbox)
    shutil.copy2(EVAL_QUERIES_PATH, sandbox)

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
# Summary
# =============================================================================

def print_summary(results: Dict):
    """Print a summary of agent run metrics."""
    print(f"\n{'='*60}")
    print(f"  DEMO TASK 2 RESULTS: Self-Evolving RAG")
    print(f"{'='*60}")
    for label, key in [("BASELINE", "baseline"), ("WITH KB", "with_kb")]:
        t = results.get(f"{key}_time", "?")
        c = results.get(f"{key}_cost", 0)
        tc = results.get(f"{key}_tool_calls", "?")
        ok = results.get(f"{key}_success", "?")
        print(f"\n  {label}:")
        print(f"    Success:    {ok}")
        print(f"    Time:       {t}s")
        print(f"    Cost:       ${c:.4f}")
        print(f"    Tool calls: {tc}")
    print(f"\n  Results: {RESULTS_FILE}")
    print(f"{'='*60}")


# =============================================================================
# Main
# =============================================================================

def _parse_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Self-Evolving RAG Benchmark Runner"
    )
    parser.add_argument(
        "--bedrock",
        action="store_true",
        default=False,
        help="Use AWS Bedrock instead of Anthropic API (default: Anthropic API)",
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
    AGENT_MODEL = BEDROCK_MODEL if USE_BEDROCK else ANTHROPIC_MODEL_ID

    mode_label = "Bedrock" if USE_BEDROCK else "Anthropic API"
    print(f"Mode: {mode_label}  Model: {AGENT_MODEL}", flush=True)

    # Determine which phases to run
    run_bl = args.baseline or (not args.baseline and not args.with_kb)
    run_kb = args.with_kb or (not args.baseline and not args.with_kb)

    TASK_WORKSPACE.mkdir(parents=True, exist_ok=True)

    # Pre-generate standardized evaluation data before any agent runs
    print("\n  Preparing evaluation data...")
    prepare_eval_data()

    results = {}
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            results = json.load(f)

    # Run with_kb first so the KB-assisted agent isn't influenced by baseline artifacts
    if run_kb:
        m = run_agent_phase("with_kb", PROPOSAL_FILE, mcp_config=MCP_CONFIG_PATH)
        results.update(with_kb_time=m["time_seconds"], with_kb_cost=m["cost_usd"],
                       with_kb_tool_calls=m["tool_call_count"], with_kb_success=m["success"])
        save_text(RESULTS_FILE, json.dumps(results, indent=2))

    if run_bl:
        m = run_agent_phase("baseline", PROPOSAL_FILE, mcp_config=None)
        results.update(baseline_time=m["time_seconds"], baseline_cost=m["cost_usd"],
                       baseline_tool_calls=m["tool_call_count"], baseline_success=m["success"])
        save_text(RESULTS_FILE, json.dumps(results, indent=2))

    print_summary(results)


if __name__ == "__main__":
    main()
