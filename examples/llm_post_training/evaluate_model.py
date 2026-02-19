#!/usr/bin/env python3
"""
evaluate_model.py — Pre-built evaluation helper for the LLM pipeline benchmark.

Runs IFEval evaluation using lm-evaluation-harness.
This script is provided to ensure consistent, correct evaluation across runs.

Usage:
    python evaluate_model.py --model ./merged_model --label trained
    python evaluate_model.py --model Qwen/Qwen2.5-1.5B --label base
    python evaluate_model.py --model Qwen/Qwen2.5-1.5B-Instruct --label instruct

Each run saves JSON results to ./eval_results/{label}_ifeval/ and a summary
to ./eval_results/{label}_summary.json.
"""

import argparse
import json
import os
import subprocess
import sys
import time


# ---- Benchmark task names (DO NOT CHANGE) ----
# These are the exact lm-evaluation-harness task identifiers.
IFEVAL_TASK = "ifeval"

# ---- Default settings ----
BATCH_SIZE = 8
DTYPE = "bfloat16"


def run_lm_eval(model_path, task_name, output_dir, batch_size=BATCH_SIZE):
    """
    Run a single lm-evaluation-harness benchmark.

    Args:
        model_path: HuggingFace model ID or local path to model directory.
        task_name: lm-eval task name (e.g. 'ifeval').
        output_dir: Directory where lm-eval writes its result JSON files.
        batch_size: Evaluation batch size.

    Returns:
        True if the evaluation succeeded, False otherwise.
    """
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_path},trust_remote_code=true,dtype={DTYPE}",
        "--tasks", task_name,
        "--batch_size", str(batch_size),
        "--output_path", output_dir,
    ]

    print(f"\n{'='*60}")
    print(f"Running: {task_name}")
    print(f"Model:   {model_path}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    sys.stdout.flush()

    start = time.time()
    result = subprocess.run(cmd, capture_output=False, text=True)
    elapsed = time.time() - start

    print(f"  Completed in {elapsed:.0f}s (exit code {result.returncode})")
    return result.returncode == 0


def find_results_json(output_dir):
    """
    Find and load the results JSON file written by lm-eval-harness.

    lm-eval writes results to: output_dir/<model_name_sanitized>/results_<timestamp>.json
    This function walks the output_dir to find it.
    """
    for root, dirs, files in os.walk(output_dir):
        for f in files:
            if f.startswith("results_") and f.endswith(".json"):
                filepath = os.path.join(root, f)
                with open(filepath) as fh:
                    return json.load(fh)
    return None


def extract_ifeval_score(results_json):
    """Extract prompt-level strict accuracy from IFEval results."""
    if not results_json or "results" not in results_json:
        return None
    ifeval = results_json["results"].get("ifeval", {})
    return ifeval.get("prompt_level_strict_acc,none")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a model on the IFEval benchmark."
    )
    parser.add_argument(
        "--model", required=True,
        help="HuggingFace model ID or local path (e.g. ./merged_model)"
    )
    parser.add_argument(
        "--label", required=True,
        help="Label for this evaluation run (e.g. trained, base, instruct)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE,
        help=f"Batch size for evaluation (default: {BATCH_SIZE})"
    )
    parser.add_argument(
        "--output-dir", default="./eval_results",
        help="Base directory for evaluation results (default: ./eval_results)"
    )
    args = parser.parse_args()

    # Output directory for IFEval benchmark
    ifeval_dir = os.path.join(args.output_dir, f"{args.label}_ifeval")

    print(f"\n{'#'*60}")
    print(f"# Evaluating: {args.label}")
    print(f"# Model: {args.model}")
    print(f"# Output: {args.output_dir}")
    print(f"{'#'*60}")

    # ---- Run IFEval ----
    print("\n[1/1] IFEval evaluation...")
    ifeval_ok = run_lm_eval(args.model, IFEVAL_TASK, ifeval_dir, args.batch_size)

    # ---- Extract and print results ----
    print(f"\n{'='*60}")
    print(f"RESULTS: {args.label}")
    print(f"{'='*60}")

    ifeval_score = None

    if ifeval_ok:
        ifeval_json = find_results_json(ifeval_dir)
        ifeval_score = extract_ifeval_score(ifeval_json)
        if ifeval_score is not None:
            print(f"  IFEval (prompt_level_strict_acc): {ifeval_score:.4f} ({ifeval_score*100:.2f}%)")
        else:
            print("  IFEval: Could not extract score from results")
    else:
        print("  IFEval: FAILED")

    # Save a summary JSON for easy consumption
    summary = {
        "label": args.label,
        "model": args.model,
        "ifeval_prompt_strict_acc": ifeval_score,
        "ifeval_success": ifeval_ok,
    }
    summary_path = os.path.join(args.output_dir, f"{args.label}_summary.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved to: {summary_path}")
    print(f"{'='*60}\n")

    # Return non-zero exit code if benchmark failed
    if not ifeval_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
