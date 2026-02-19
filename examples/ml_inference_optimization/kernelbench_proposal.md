# KernelBench GPU Kernel Optimization — Proposal

## Overview

Write optimized CUDA C++ or Triton GPU kernels for 10 selected KernelBench problems. Your workspace contains a `fixtures/` directory with the original PyTorch reference code, pre-generated inputs, expected outputs, and an evaluation script.

Your goal: for each problem, write a `ModelNew` class that produces the **same outputs** as the reference `Model` but runs **as fast as possible**. Speedup > 1.0× is the minimum bar — not the target. Before writing any kernel, estimate the theoretical speedup ceiling (from roofline analysis, fusion opportunities, or precision gains) and aim for that ceiling.

---

## Environment

You are running inside the `kernelbench` conda environment on a machine with an NVIDIA GPU. CUDA toolkit, PyTorch (with CUDA support), Triton, and Ninja (for JIT compilation) are pre-installed. You are free to install additional Python packages with `pip install` if needed.

Verify GPU access:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## Task — GPU Kernel Optimization

### Phase 1 — Understand the Problems

Read each problem file in `fixtures/problems/`. Each file defines:

- A `Model(nn.Module)` class — the PyTorch reference implementation (correct but unoptimized).
- `get_inputs()` — returns a list of input tensors for one forward pass.
- `get_init_inputs()` — returns constructor arguments for `Model(...)`.

Your job is to write a `ModelNew(nn.Module)` class for each problem that produces the same outputs but runs as fast as possible using custom GPU kernels.

Study each problem carefully. Understand the shapes and the operations being performed.

### Phase 2 — Write Solutions

For each problem, create a solution file in `solutions/` with the **exact same filename** as the problem file. For example, if the problem is `L1_36_RMSNorm.py`, your solution must be `solutions/L1_36_RMSNorm.py`.

Each solution file must define a `ModelNew(nn.Module)` class with the same constructor signature as the original `Model`. The `ModelNew.forward()` method must accept the same inputs and produce equivalent outputs.

**Performance goal (primary):**

- Each solution MUST be faster than the PyTorch reference (speedup > 1.0×).
- 1.0× is the floor, not the target. Aim for the theoretical ceiling.
- Before writing any kernel, estimate the achievable speedup (roofline analysis, fusion opportunities, precision opportunities). Use that estimate as your per-problem target.
- If your solution achieves < 1.2× speedup, try at least one fundamentally different approach before accepting the result.

**Method constraint (minimal):**

- Each solution must include at least one custom GPU kernel (CUDA C++ via `torch.utils.cpp_extension.load_inline`, or Triton JIT). You have complete freedom to choose which operators to replace and which to leave as PyTorch.
- Must be a self-contained `.py` file (all kernel code inline).
- Must import `torch` and `torch.nn as nn` at minimum.

**The 10 problems:**

| # | Problem ID | Description |
|---|-----------|-------------|
| 1 | L1-36 | RMSNorm (batch=112, features=64, 512×512) |
| 2 | L1-82 | Depthwise Conv2D (batch=16, ch=64, 512×512, k=3) |
| 3 | L1-97 | Scaled Dot-Product Attention (batch=32, heads=32, seq=512, dim=1024) |
| 4 | L2-34 | ConvTranspose3d + LayerNorm + GELU + Scaling |
| 5 | L2-37 | Matmul + Swish + Sum + GroupNorm (batch=32768) |
| 6 | L2-84 | Gemm + BatchNorm + Scaling + Softmax (8192×8192) |
| 7 | L2-99 | Matmul + GELU + Softmax (8192×8192) |
| 8 | L3-43 | MinGPT Causal Self-Attention (128, 512, 768, 8 heads) |
| 9 | L3-44 | MiniGPT Transformer Block (LN + Attn + LN + MLP) |
| 10 | L3-20 | MobileNetV2 (inverted residuals) |

### Phase 3 — Evaluate

After writing all 10 solutions, run the pre-built evaluation script:

```bash
python fixtures/evaluate.py --solutions-dir ./solutions/
```

**Do not** pre-compile kernels, kill GPU processes, or add extra setup steps. Just run `evaluate.py` — it handles everything.

This script:

1. Loads your `ModelNew` for each problem.
2. Instantiates it with the pre-saved constructor arguments.
3. Loads the reference model weights into `ModelNew` via `load_state_dict(strict=True)` — your `ModelNew` must have **exactly the same parameter names** as the original `Model`.
4. Runs it on 5 pre-generated input sets.
5. Compares outputs against pre-saved expected outputs (atol=1e-2, rtol=1e-2).
6. Measures kernel timing using **freshly generated random inputs** (not the fixture inputs).
7. Computes speedup vs. the reference PyTorch timing.
8. Prints a results table and writes `results.json`.

**Important:** Correctness checks use pre-generated inputs and expected outputs with saved model weights — this ensures deterministic, reproducible results. Timing uses fresh random inputs to measure real kernel performance.

**Critical:** Because weights are loaded with `strict=True`, your `ModelNew` must define layers with the **exact same names** as the original `Model`. For example, if `Model` has `self.conv_transpose`, your `ModelNew` must also use `self.conv_transpose` (not `self.conv` or `self.deconv`). Read the original `Model` constructor carefully and match every `nn.Module` attribute name exactly.

---

## Core Technical Requirements

- Each solution must define `ModelNew(nn.Module)` with the **same constructor signature** as `Model`.
- `ModelNew` must use the **exact same layer/parameter names** as `Model` (e.g., if `Model` has `self.matmul`, so must `ModelNew`). The evaluator loads reference weights via `load_state_dict(strict=True)`.
- Each solution must include at least one custom CUDA C++ or Triton kernel. You have complete freedom to choose which operators to replace and which to leave as PyTorch.
- Correctness is checked against pre-generated expected outputs (atol=1e-2, rtol=1e-2).
- Solutions must be self-contained files (one per problem, all kernel code inline).

---

## Completion Requirement

Do NOT stop until all 10 problems have been attempted and `evaluate.py` has been run. Print the final results showing for each problem:

- **Compiled**: whether the solution compiled successfully
- **Correct**: whether outputs matched expected values within tolerance
- **Speedup**: the ratio of reference timing to your kernel timing

If a solution fails to compile or produces incorrect results, debug it. Try at least 2 fix attempts per problem before moving on.

### Iterative Improvement (mandatory)

After the first full evaluation, review the results:

1. Identify every problem with speedup < 1.2×.
2. Try again to improve it.
3. Re-evaluate.

Do NOT consider a problem "done" at 1.01× unless you have tried at least 2 different approaches and documented (in comments) why further improvement is impractical.

### Performance Expectations by Problem Type

Use these as rough targets — not hard limits:

- **Naive PyTorch reference (raw ops, no library calls):** Expect 1.5×–3.0×+.
- **Reference using optimized libraries (cuDNN, F.sdpa, cuBLAS):** Expect 1.1×–1.5×. If the library call dominates, small gains are acceptable — but you must try.
- **Multi-op pipelines (L2/L3 problems):** Expect 1.2×–2.0×.

After all improvements, run the full evaluation one final time and print the complete results table.
