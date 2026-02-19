# GPU Kernel Optimization Benchmark

This benchmark measures how a curated knowledge base (Leeroopedia MCP) improves an AI coding agent's ability to write optimized CUDA/Triton GPU kernels.

Both agents receive the **same task**: write custom GPU kernels for 10 [KernelBench](https://github.com/ScalingIntelligence/KernelBench) problems spanning element-wise ops (L1), fused operator chains (L2), and full transformer blocks (L3). Each solution must produce correct outputs and run faster than the PyTorch reference. The only difference is that one agent has access to Leeroopedia's kernel optimization docs via MCP tools. Both agents used 1 × NVIDIA L4 Tensor Core GPU machine.

## Results

| Problem | Baseline | + Leeroopedia | Delta |
|---------|:--------:|:-------------:|:-----:|
| **L1-36** RMSNorm | 2.13x | **2.45x** | +15% |
| **L1-82** DepthwiseConv2D | 1.01x | **1.22x** | +21% |
| **L1-97** ScaledDotProductAttention | **1.44x** | 1.25x | -13% |
| **L2-34** ConvTranspose3d+LN+GELU+Scale | **1.23x** | 1.22x | ~tie |
| **L2-37** Matmul+Swish+Sum+GroupNorm | 2.34x | **2.99x** | +28% |
| **L2-84** Gemm+BN+Scale+Softmax | **3.38x** | 2.84x | -16% |
| **L2-99** Matmul+GELU+Softmax | **3.30x** | 2.77x | -16% |
| **L3-43** MinGPT Causal Attention | 2.07x | **3.74x** | +81% |
| **L3-44** MiniGPT Transformer Block | 1.75x | **3.67x** | +110% |
| **L3-20** MobileNetV2 | 1.01x | **1.21x** | +20% |
| | | | |
| **Geometric mean** | 1.80x | **2.11x** | **+17%** |

### Why the KB agent produced faster kernels

The critical differentiator was a single KB-informed optimization: **TF32 tensor cores** (`torch.backends.cuda.matmul.allow_tf32 = True`). The baseline agent used fp16 casting (`.half()`) for matmul acceleration, while the KB agent discovered TF32, a hardware feature that doubles matmul throughput with no precision loss for inference workloads.

This explains the pattern in the results:

- **L3-43/L3-44 (transformer blocks)**: TF32 matmuls gave 3.7x speedup vs baseline's 1.8-2.1x. These problems are matmul-dominated, so TF32 has the largest impact.
- **L2-84/L2-99 (GEMM-heavy)**: The baseline's fp16 approach was actually slightly faster here (3.3-3.4x vs 2.8x), because fp16 provides more throughput than TF32 at the cost of precision.
- **L2-37**: The KB agent fused Swish+Bias+GroupNorm into a single Triton kernel (2.99x), while the baseline only fused Swish+Bias and called PyTorch GroupNorm separately (2.34x).
- **L1-82/L3-20 (near-1.0x baseline)**: The KB agent wrote actual custom kernels (depthwise conv, ReLU6 with float4 vectorization + CUDA Graphs), while the baseline fell back to cuDNN/identity kernels.

---

## How to replicate

Full logs and outputs from our runs are available [here](https://drive.google.com/file/d/18hgQnKt1kmyQJ50IaiXkpqhZP0lBL5tT/view?usp=sharing).

### Prerequisites

- Python 3.10+
- [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) installed and configured
- An Anthropic API key **or** AWS Bedrock access
- A Leeroopedia API key
- A machine with an NVIDIA GPU (A100 recommended)

### 1. Set up environment variables

Create a `.env` file in the parent `examples/` directory (shared across all benchmarks):

```
LEEROOPEDIA_API_KEY="kpsk_..."
ANTHROPIC_API_KEY="sk-ant-..."

# Only needed if using --bedrock mode
# AWS_BEARER_TOKEN_BEDROCK="..."
# CLAUDE_CODE_USE_BEDROCK=1
# AWS_REGION=us-east-1
```

### 2. Generate fixtures

The benchmark needs pre-computed reference inputs, outputs, weights, and timing for each problem:

```bash
pip install python-dotenv torch
python generate_kernelbench_fixtures.py
```

This downloads the 10 problem files from KernelBench, generates deterministic test data, and measures reference timing.

### 3. Run the benchmark

```bash
# Default: uses Anthropic API key
python run_benchmark.py

# Alternative: use AWS Bedrock
python run_benchmark.py --bedrock
```

This runs the full pipeline:

1. **Phase 1 (Baseline)** -- Runs Claude Code with `kernelbench_proposal.md` as the task. The agent writes and tests GPU kernels in an isolated sandbox with no KB access.
2. **Phase 2 (With KB)** -- Runs Claude Code with the same proposal plus Leeroopedia MCP tools. The agent can call `search_knowledge`, `build_plan`, `propose_hypothesis`, `diagnose_failure`, etc.

Both agents work in isolated `/tmp` sandboxes. Solutions are copied to `workspaces/baseline/` and `workspaces/with_kb/` after each phase.

> **Tip:** For better Leeroopedia usage, append the contents of [`leeroopedia_tool_usage.md`](leeroopedia_tool_usage.md) to the end of `kernelbench_proposal.md`. This teaches the agent the recommended per-problem workflow for KB lookups.