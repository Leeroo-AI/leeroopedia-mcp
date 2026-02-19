"""
KernelBench Fixture Generator

Downloads 10 selected KernelBench problems from GitHub, pre-generates
deterministic random inputs, runs the reference Model to get expected
outputs, measures reference timing, and creates evaluate.py.

Must run on a machine with an NVIDIA GPU before any agent runs.

Usage:
  python generate_kernelbench_fixtures.py
  python generate_kernelbench_fixtures.py --fixtures-dir ./custom_fixtures
"""

import gc
import importlib.util
import json
import os
import shutil
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Configuration: the 10 selected KernelBench problems
# ---------------------------------------------------------------------------

PROBLEMS = [
    {
        "id": "L1_36",
        "name": "RMSNorm",
        "level": 1,
        "github_path": "KernelBench/level1/36_RMSNorm_.py",
        "description": "RMS Normalization on (112, 64, 512, 512) tensor",
    },
    {
        "id": "L1_82",
        "name": "DepthwiseConv2D",
        "level": 1,
        "github_path": "KernelBench/level1/82_conv_depthwise_2D_square_input_square_kernel.py",
        "description": "Depthwise 2D convolution (16, 64, 512, 512) with 3x3 kernel",
    },
    {
        "id": "L1_97",
        "name": "ScaledDotProductAttention",
        "level": 1,
        "github_path": "KernelBench/level1/97_ScaledDotProductAttention.py",
        "description": "Scaled dot-product attention (32, 32, 512, 1024)",
    },
    {
        "id": "L2_34",
        "name": "ConvTranspose3d_LayerNorm_GELU_Scaling",
        "level": 2,
        "github_path": "KernelBench/level2/34_ConvTranspose3d_LayerNorm_GELU_Scaling.py",
        "description": "3D transposed conv + LayerNorm + GELU + scaling",
    },
    {
        "id": "L2_37",
        "name": "Matmul_Swish_Sum_GroupNorm",
        "level": 2,
        "github_path": "KernelBench/level2/37_Matmul_Swish_Sum_GroupNorm.py",
        "description": "Linear + Swish + bias add + GroupNorm (batch=32768)",
    },
    {
        "id": "L2_84",
        "name": "Gemm_BatchNorm_Scaling_Softmax",
        "level": 2,
        "github_path": "KernelBench/level2/84_Gemm_BatchNorm_Scaling_Softmax.py",
        "description": "Linear + BatchNorm + scaling + Softmax (8192x8192)",
    },
    {
        "id": "L2_99",
        "name": "Matmul_GELU_Softmax",
        "level": 2,
        "github_path": "KernelBench/level2/99_Matmul_GELU_Softmax.py",
        "description": "Linear + GELU + Softmax (8192x8192)",
    },
    {
        "id": "L3_43",
        "name": "MinGPTCausalAttention",
        "level": 3,
        "github_path": "KernelBench/level3/43_MinGPTCausalAttention.py",
        "description": "MinGPT causal self-attention (128, 512, 768, 8 heads)",
    },
    {
        "id": "L3_44",
        "name": "MiniGPTBlock",
        "level": 3,
        "github_path": "KernelBench/level3/44_MiniGPTBlock.py",
        "description": "Full MiniGPT transformer block (LN + Attn + LN + MLP)",
    },
    {
        "id": "L3_20",
        "name": "MobileNetV2",
        "level": 3,
        "github_path": "KernelBench/level3/20_MobileNetV2.py",
        "description": "Full MobileNetV2 architecture with inverted residuals",
    },
]

GITHUB_RAW_BASE = (
    "https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/main/"
)

# Number of random input sets to generate per problem
NUM_INPUT_SETS = 5

# Timing: warmup runs + timed runs
WARMUP_RUNS = 3
TIMED_RUNS = 10

# Correctness tolerance (matches KernelBench defaults)
ATOL = 1e-2
RTOL = 1e-2

BENCH_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def download_file(url: str, dest: Path):
    """Download a file from a URL to a local path."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {url} ...", flush=True)
    urllib.request.urlretrieve(url, dest)


def load_module_from_file(filepath: Path, module_name: str):
    """Dynamically import a Python file as a module."""
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def measure_timing_ms(model, inputs: List, device: str = "cuda") -> Dict[str, float]:
    """
    Measure forward pass timing using CUDA events.
    Returns dict with mean_ms and std_ms.
    """
    import torch

    model.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(WARMUP_RUNS):
            _ = model(*[x.to(device) if hasattr(x, "to") else x for x in inputs])
            torch.cuda.synchronize()

        # Timed runs
        times = []
        for _ in range(TIMED_RUNS):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            torch.cuda.synchronize()
            start_event.record()
            _ = model(*[x.to(device) if hasattr(x, "to") else x for x in inputs])
            end_event.record()
            torch.cuda.synchronize()

            times.append(start_event.elapsed_time(end_event))

    mean_ms = sum(times) / len(times)
    variance = sum((t - mean_ms) ** 2 for t in times) / len(times)
    std_ms = variance ** 0.5
    return {"mean_ms": round(mean_ms, 4), "std_ms": round(std_ms, 4)}


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------

def _log_memory(label: str):
    """Print current CPU and GPU memory usage for debugging."""
    import torch

    # GPU memory
    gpu_alloc = torch.cuda.memory_allocated() / (1024 ** 3)
    gpu_reserved = torch.cuda.memory_reserved() / (1024 ** 3)

    # CPU memory via /proc/self/status (Linux only)
    cpu_gb = 0.0
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    cpu_gb = int(line.split()[1]) / (1024 ** 2)  # kB -> GB
                    break
    except Exception:
        pass

    print(
        f"  [{label}] CPU RSS: {cpu_gb:.1f} GB | "
        f"GPU alloc: {gpu_alloc:.1f} GB | GPU reserved: {gpu_reserved:.1f} GB",
        flush=True,
    )


def _cleanup_memory():
    """Aggressively free CPU and GPU memory."""
    import torch
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()


# ---------------------------------------------------------------------------
# Main fixture generation
# ---------------------------------------------------------------------------

def generate_fixtures(fixtures_dir: Path):
    """Generate all fixtures for the 10 KernelBench problems.

    Memory-safe: processes one input set at a time, saves to disk
    immediately, and frees all tensors between problems.
    """
    import torch

    print(f"Generating KernelBench fixtures in {fixtures_dir}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("ERROR: CUDA is required to generate fixtures.")
        sys.exit(1)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    print()

    # Create directory structure
    problems_dir = fixtures_dir / "problems"
    inputs_dir = fixtures_dir / "inputs"
    outputs_dir = fixtures_dir / "expected_outputs"
    timing_dir = fixtures_dir / "reference_timing"
    for d in [problems_dir, inputs_dir, outputs_dir, timing_dir]:
        d.mkdir(parents=True, exist_ok=True)

    manifest = []
    failed_problems = []

    for prob_idx, prob in enumerate(PROBLEMS):
        pid = prob["id"]
        pname = prob["name"]
        filename = f"{pid}_{pname}.py"
        print(f"=== [{prob_idx+1}/{len(PROBLEMS)}] {pid}: {pname} ===")
        _log_memory("before")

        try:
            _generate_one_problem(
                prob, filename, problems_dir, inputs_dir,
                outputs_dir, timing_dir, manifest, fixtures_dir,
            )
        except Exception as e:
            print(f"  ERROR: {pid} failed -- {e}", flush=True)
            failed_problems.append({"id": pid, "name": pname, "error": str(e)})

        # Aggressive cleanup between problems to prevent OOM
        _cleanup_memory()
        _log_memory("after cleanup")
        print()

    # Save manifest (only includes successful problems)
    manifest_path = fixtures_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest saved to {manifest_path}")

    if failed_problems:
        print(f"\nWARNING: {len(failed_problems)} problem(s) failed:")
        for fp in failed_problems:
            print(f"  - {fp['id']}: {fp['error']}")

    # Generate evaluate.py
    generate_evaluate_script(fixtures_dir)
    print(f"\nFixtures generated successfully in {fixtures_dir}")


def _generate_one_problem(
    prob: Dict, filename: str,
    problems_dir: Path, inputs_dir: Path,
    outputs_dir: Path, timing_dir: Path,
    manifest: List, fixtures_dir: Path,
):
    """Generate fixtures for a single problem.

    Memory-safe: saves each input/output set as a separate file
    (e.g., L1_36_RMSNorm_input_0.pt) so we never hold more than
    one set in memory at once. The evaluate script loads them
    individually too.
    """
    import torch

    pid = prob["id"]
    pname = prob["name"]
    prefix = f"{pid}_{pname}"

    # Step 1: Download the problem file
    url = GITHUB_RAW_BASE + prob["github_path"]
    dest = problems_dir / filename
    if not dest.exists():
        download_file(url, dest)
    else:
        print(f"  Already downloaded: {dest.name}")

    # Step 2: Import the problem module
    print(f"  Loading module ...", flush=True)
    mod = load_module_from_file(dest, f"problem_{pid}")

    # Step 3: Instantiate the reference Model
    init_inputs = mod.get_init_inputs()
    model = mod.Model(*init_inputs).cuda().eval()

    # Save init_inputs right away (small)
    torch.save(init_inputs, inputs_dir / f"{prefix}_init_inputs.pt")

    # Save model weights so the evaluator can load them into ModelNew.
    # This ensures ModelNew uses the exact same weights that generated
    # the expected outputs, regardless of random state differences.
    weights_dir = fixtures_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), weights_dir / f"{prefix}_weights.pt")
    print(f"  Saved model weights ({len(model.state_dict())} tensors)")

    # Step 4: Generate each input/output set one at a time.
    # Each set is saved to its own file — never combined.
    print(f"  Generating {NUM_INPUT_SETS} input sets (one file each) ...", flush=True)

    for i in range(NUM_INPUT_SETS):
        torch.manual_seed(i)
        torch.cuda.manual_seed(i)

        # Generate inputs on CPU
        inputs = mod.get_inputs()

        # Save CPU inputs to their own file immediately
        torch.save(inputs, inputs_dir / f"{prefix}_input_{i}.pt")

        # Move to GPU for forward pass
        cuda_inputs = [
            x.cuda() if hasattr(x, "cuda") else x for x in inputs
        ]

        # Free the CPU copies — saved to disk already
        del inputs
        gc.collect()

        # Run reference model
        with torch.no_grad():
            output = model(*cuda_inputs)

        # Save output CPU copy to its own file
        cpu_output = output.cpu() if isinstance(output, torch.Tensor) else output
        torch.save(cpu_output, outputs_dir / f"{prefix}_output_{i}.pt")

        # Free ALL GPU and CPU tensors for this input set
        del cuda_inputs, output, cpu_output
        torch.cuda.empty_cache()
        gc.collect()

        print(f"    Input set {i+1}/{NUM_INPUT_SETS} done", flush=True)

    print(f"  Saved {NUM_INPUT_SETS} input/output file pairs")

    # Step 5: Measure reference timing using the first input set.
    # Load just set 0 from disk — small memory footprint.
    print(f"  Measuring reference timing ...", flush=True)
    first_inputs = torch.load(
        inputs_dir / f"{prefix}_input_0.pt", weights_only=False
    )
    cuda_first = [
        x.cuda() if hasattr(x, "cuda") else x for x in first_inputs
    ]
    del first_inputs
    gc.collect()

    timing = measure_timing_ms(model, cuda_first)
    timing_path = timing_dir / f"{prefix}_timing.json"
    with open(timing_path, "w") as f:
        json.dump(timing, f, indent=2)
    print(f"  Reference timing: {timing['mean_ms']:.2f} +/- {timing['std_ms']:.2f} ms")

    # Free timing resources
    del cuda_first

    # Step 6: Add to manifest
    manifest.append({
        "id": pid,
        "name": pname,
        "level": prob["level"],
        "file": filename,
        "description": prob["description"],
        "reference_timing_ms": timing["mean_ms"],
        "num_input_sets": NUM_INPUT_SETS,
    })

    # Step 7: Free ALL memory for this problem
    del model, init_inputs, mod
    torch.cuda.empty_cache()
    gc.collect()


# ---------------------------------------------------------------------------
# Generate the evaluate.py script
# ---------------------------------------------------------------------------

def generate_evaluate_script(fixtures_dir: Path):
    """Create the evaluate.py script that agents will run."""
    script = '''\
"""
KernelBench Evaluation Script

Evaluates agent-generated CUDA/Triton kernel solutions against
pre-generated inputs and expected outputs. Deterministic -- no
randomness at evaluation time.

Usage:
  python evaluate.py --solutions-dir ./solutions/

The script loads each solution's ModelNew class, runs it on pre-generated
inputs, compares outputs against expected values, and measures speedup
vs the reference PyTorch timing.

Output: prints a results table and writes results.json.
"""

import argparse
import importlib.util
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.nn as nn


# --- Configuration ---
ATOL = 1e-2
RTOL = 1e-2
NUM_CORRECTNESS_CHECKS = 5   # matches the number of pre-generated input sets
WARMUP_RUNS = 3
TIMED_RUNS = 10


def load_module(filepath: Path, module_name: str):
    """Dynamically import a Python file as a module."""
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_correctness(model_new, inputs_list, expected_outputs, device="cuda"):
    """
    Check if ModelNew produces correct outputs on all input sets.
    Returns (is_correct, error_message).
    """
    model_new.eval()
    with torch.no_grad():
        for i, (inputs, expected) in enumerate(zip(inputs_list, expected_outputs)):
            cuda_inputs = [
                x.to(device) if hasattr(x, "to") else x for x in inputs
            ]
            try:
                output = model_new(*cuda_inputs)
            except Exception as e:
                return False, f"Runtime error on input set {i}: {e}"

            # Move to CPU for comparison
            if isinstance(output, torch.Tensor):
                output_cpu = output.cpu()
            else:
                output_cpu = output

            expected_cpu = expected.cpu() if isinstance(expected, torch.Tensor) else expected

            if isinstance(output_cpu, torch.Tensor) and isinstance(expected_cpu, torch.Tensor):
                if output_cpu.shape != expected_cpu.shape:
                    return False, (
                        f"Shape mismatch on input set {i}: "
                        f"got {output_cpu.shape}, expected {expected_cpu.shape}"
                    )
                if not torch.allclose(output_cpu, expected_cpu, atol=ATOL, rtol=RTOL):
                    max_diff = (output_cpu - expected_cpu).abs().max().item()
                    return False, (
                        f"Value mismatch on input set {i}: max_diff={max_diff:.6f}"
                    )
            else:
                # Non-tensor output -- try direct comparison
                if output_cpu != expected_cpu:
                    return False, f"Output mismatch on input set {i}"
    return True, ""


def measure_timing(model_new, inputs, device="cuda"):
    """Measure forward pass timing using CUDA events. Returns mean_ms."""
    model_new.eval()
    cuda_inputs = [x.to(device) if hasattr(x, "to") else x for x in inputs]

    with torch.no_grad():
        # Warmup
        for _ in range(WARMUP_RUNS):
            _ = model_new(*cuda_inputs)
            torch.cuda.synchronize()

        # Timed runs
        times = []
        for _ in range(TIMED_RUNS):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            torch.cuda.synchronize()
            start.record()
            _ = model_new(*cuda_inputs)
            end.record()
            torch.cuda.synchronize()

            times.append(start.elapsed_time(end))

    return sum(times) / len(times)


def main():
    parser = argparse.ArgumentParser(description="KernelBench evaluation")
    parser.add_argument(
        "--solutions-dir", type=str, required=True,
        help="Directory containing solution files (e.g., L1_36_RMSNorm.py)",
    )
    parser.add_argument(
        "--output", type=str, default="results.json",
        help="Path to write results JSON (default: results.json)",
    )
    args = parser.parse_args()

    solutions_dir = Path(args.solutions_dir)
    fixtures_dir = Path(__file__).resolve().parent

    # Load manifest
    manifest_path = fixtures_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"ERROR: manifest.json not found at {manifest_path}")
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("WARNING: CUDA not available. Timing will be meaningless.")

    results = {}
    total_correct = 0
    total_faster = 0

    print()
    print("=" * 75)
    print("  KernelBench Evaluation")
    print("=" * 75)
    print(f"  Solutions dir: {solutions_dir}")
    print(f"  Device: {device}")
    print(f"  Problems: {len(manifest)}")
    print("=" * 75)
    print()

    for prob in manifest:
        pid = prob["id"]
        pname = prob["name"]
        filename = prob["file"]
        ref_timing = prob["reference_timing_ms"]

        print(f"--- {pid}: {pname} ---")

        # Load pre-generated inputs and expected outputs (individual files per set)
        init_inputs_path = fixtures_dir / "inputs" / f"{pid}_{pname}_init_inputs.pt"
        num_sets = prob.get("num_input_sets", NUM_CORRECTNESS_CHECKS)

        # Check that at least the first input/output file exists
        first_input = fixtures_dir / "inputs" / f"{pid}_{pname}_input_0.pt"
        first_output = fixtures_dir / "expected_outputs" / f"{pid}_{pname}_output_0.pt"

        if not first_input.exists() or not first_output.exists():
            print(f"  SKIP: fixture files not found")
            results[pid] = {
                "name": pname, "compiled": False, "correct": False,
                "speedup": 0.0, "error": "fixture files missing",
            }
            continue

        init_inputs = torch.load(init_inputs_path, weights_only=False)

        # Find solution file
        solution_file = None
        # Try exact match first, then prefix match
        for candidate in [
            solutions_dir / filename,
            solutions_dir / f"{pid}_{pname}.py",
            solutions_dir / f"{pid}.py",
        ]:
            if candidate.exists():
                solution_file = candidate
                break

        # Also search for any file containing the problem ID
        if solution_file is None:
            for f in solutions_dir.glob("*.py"):
                if pid in f.name:
                    solution_file = f
                    break

        if solution_file is None:
            print(f"  NOT FOUND: no solution file for {pid}")
            results[pid] = {
                "name": pname, "compiled": False, "correct": False,
                "speedup": 0.0, "error": "solution file not found",
            }
            continue

        print(f"  Solution: {solution_file.name}")

        # Try to import and compile the solution
        try:
            mod = load_module(solution_file, f"solution_{pid}")
            model_new = mod.ModelNew(*init_inputs).to(device).eval()

            # Load the saved reference weights into ModelNew.
            # This ensures ModelNew uses the exact same weights that
            # generated the expected outputs, fixing weight mismatch.
            weights_path = fixtures_dir / "weights" / f"{pid}_{pname}_weights.pt"
            if weights_path.exists():
                state_dict = torch.load(weights_path, weights_only=False, map_location=device)
                model_new.load_state_dict(state_dict, strict=True)
                model_new.eval()
                print(f"  Weights: loaded ({len(state_dict)} tensors)")
            else:
                print(f"  Weights: none saved (weight-free model)")

            compiled = True
            print(f"  Compiled: YES")
        except Exception as e:
            tb = traceback.format_exc()
            print(f"  Compiled: NO -- {e}")
            results[pid] = {
                "name": pname, "compiled": False, "correct": False,
                "speedup": 0.0, "error": f"compile error: {e}",
                "traceback": tb,
            }
            continue

        # Check correctness — load each input/output set one at a time
        try:
            correct, err_msg = True, ""
            model_new.eval()
            with torch.no_grad():
                for i in range(num_sets):
                    in_path = fixtures_dir / "inputs" / f"{pid}_{pname}_input_{i}.pt"
                    out_path = fixtures_dir / "expected_outputs" / f"{pid}_{pname}_output_{i}.pt"
                    if not in_path.exists() or not out_path.exists():
                        continue
                    inputs = torch.load(in_path, weights_only=False)
                    expected = torch.load(out_path, weights_only=False)
                    cuda_inputs = [
                        x.to(device) if hasattr(x, "to") else x for x in inputs
                    ]
                    del inputs

                    try:
                        output = model_new(*cuda_inputs)
                    except Exception as e:
                        correct, err_msg = False, f"Runtime error on input set {i}: {e}"
                        del cuda_inputs
                        break

                    output_cpu = output.cpu() if isinstance(output, torch.Tensor) else output
                    expected_cpu = expected.cpu() if isinstance(expected, torch.Tensor) else expected
                    del cuda_inputs, output

                    if isinstance(output_cpu, torch.Tensor) and isinstance(expected_cpu, torch.Tensor):
                        if output_cpu.shape != expected_cpu.shape:
                            correct = False
                            err_msg = (
                                f"Shape mismatch on set {i}: "
                                f"got {output_cpu.shape}, expected {expected_cpu.shape}"
                            )
                            del output_cpu, expected_cpu, expected
                            break
                        if not torch.allclose(output_cpu, expected_cpu, atol=ATOL, rtol=RTOL):
                            max_diff = (output_cpu - expected_cpu).abs().max().item()
                            correct = False
                            err_msg = f"Value mismatch on set {i}: max_diff={max_diff:.6f}"
                            del output_cpu, expected_cpu, expected
                            break
                    del output_cpu, expected_cpu, expected
                    torch.cuda.empty_cache()
        except Exception as e:
            correct = False
            err_msg = f"correctness check crashed: {e}"

        print(f"  Correct:  {'YES' if correct else 'NO'}")
        if not correct:
            print(f"    Error: {err_msg}")
            results[pid] = {
                "name": pname, "compiled": True, "correct": False,
                "speedup": 0.0, "error": err_msg,
            }
            # Free GPU memory
            del model_new
            torch.cuda.empty_cache()
            continue

        total_correct += 1

        # Measure timing using FRESH random inputs (not from fixtures).
        # This prevents output-caching cheats: the model must run real
        # computation on inputs it has never seen before.
        try:
            # Load the problem module to access get_inputs()
            problem_file = fixtures_dir / "problems" / filename
            problem_mod = load_module(problem_file, f"timing_{pid}")

            # Use a fixed seed for reproducible timing across runs,
            # but different from the seeds used for correctness fixtures.
            torch.manual_seed(99999)
            torch.cuda.manual_seed(99999)
            fresh_inputs = problem_mod.get_inputs()

            new_timing = measure_timing(model_new, fresh_inputs, device)
            del fresh_inputs
            speedup = ref_timing / new_timing if new_timing > 0 else 0.0
        except Exception as e:
            new_timing = 0.0
            speedup = 0.0
            print(f"    Timing error: {e}")

        if speedup > 1.0:
            total_faster += 1

        print(f"  Timing:   {new_timing:.2f} ms (ref: {ref_timing:.2f} ms)")
        print(f"  Speedup:  {speedup:.2f}x")

        results[pid] = {
            "name": pname,
            "compiled": True,
            "correct": True,
            "kernel_ms": round(new_timing, 4),
            "reference_ms": ref_timing,
            "speedup": round(speedup, 4),
            "error": "",
        }

        # Free GPU memory
        del model_new
        torch.cuda.empty_cache()
        print()

    # --- Summary ---
    print()
    print("=" * 75)
    print("  RESULTS SUMMARY")
    print("=" * 75)
    print(f"  {'Problem':<45} {'Compiled':>9} {'Correct':>9} {'Speedup':>9}")
    print(f"  {'-'*45} {'-'*9} {'-'*9} {'-'*9}")
    for pid, r in results.items():
        comp = "YES" if r["compiled"] else "NO"
        corr = "YES" if r["correct"] else "NO"
        spd = f"{r['speedup']:.2f}x" if r["correct"] else "N/A"
        print(f"  {pid + ': ' + r['name']:<45} {comp:>9} {corr:>9} {spd:>9}")
    print(f"  {'-'*45} {'-'*9} {'-'*9} {'-'*9}")
    print(f"  Total correct: {total_correct}/{len(manifest)}")
    print(f"  Total faster than PyTorch: {total_faster}/{len(manifest)}")
    print("=" * 75)

    # Write results JSON
    output_path = Path(args.output)
    summary = {
        "total_problems": len(manifest),
        "total_correct": total_correct,
        "total_faster": total_faster,
        "problems": results,
    }
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\\nResults written to {output_path}")


if __name__ == "__main__":
    main()
'''
    eval_path = fixtures_dir / "evaluate.py"
    eval_path.write_text(script, encoding="utf-8")
    print(f"evaluate.py written to {eval_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate deterministic KernelBench fixtures"
    )
    parser.add_argument(
        "--fixtures-dir",
        type=str,
        default=str(BENCH_DIR / "kernelbench_fixtures"),
        help="Output directory for fixtures (default: benchmarks/.../kernelbench_fixtures)",
    )
    args = parser.parse_args()
    generate_fixtures(Path(args.fixtures_dir))
