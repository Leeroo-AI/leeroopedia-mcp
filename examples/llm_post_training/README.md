# LLM Training Pipeline Benchmark

This benchmark measures how a curated knowledge base (Leeroopedia MCP) improves an AI coding agent's ability to build and execute an end-to-end LLM post-training pipeline.

Both agents receive the **same task**: implement a complete SFT + DPO fine-tuning, LoRA merge, vLLM serving, and IFEval evaluation pipeline for `Qwen/Qwen2.5-1.5B` on 8 A100 GPUs. The only difference is that one agent has access to Leeroopedia via MCP tools.

## Results

![Benchmark Results](analysis.png)

### Why +Leeroopedia produced a better model

The coding agent with Leeroopedia MCP informed training decisions across every stage of the pipeline:

- **Conservative DPO learning rate**: 2e-6 vs 5e-5 (25x lower). The baseline's aggressive LR likely caused divergence during preference optimization. The KB agent queried hyperparameter priors and chose a rate appropriate for DPO on a 1.5B model.
- **Proper DPO LoRA setup**: The KB agent merged the SFT adapter into base weights first, then created a fresh LoRA adapter for DPO. This avoids stacking adapters and ensures clean gradient flow.
- **Dataset shuffle with fixed seed**: Explicit `seed=42` shuffle before sampling, avoiding ordering biases in ultrachat_200k.
- **Train/eval split for DPO**: ~58k train / ~3k eval, enabling loss monitoring during preference training.

---

## How to replicate

Full logs and outputs from our runs are available [here](https://drive.google.com/file/d/1fjny2SKiYcwTpCv_3Ny2o0ao5o1KiuCt/view?usp=sharing).

### Prerequisites

- Python 3.10+
- [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) installed and configured
- An Anthropic API key **or** AWS Bedrock access
- A Leeroopedia API key
- A machine with 8 GPUs (the pipeline uses multi-GPU distributed training)

### 1. Set up environment variables

Create a `.env` file in this directory:

```
LEEROOPEDIA_API_KEY="kpsk_..."
HF_TOKEN="hf_..."
ANTHROPIC_API_KEY="sk-ant-..."

# Only needed if using --bedrock mode
# AWS_BEARER_TOKEN_BEDROCK="..."
# CLAUDE_CODE_USE_BEDROCK=1
# AWS_REGION=us-east-1
```

### 2. Install Python dependencies

```bash
pip install python-dotenv
```

The agent installs additional packages (TRL, PEFT, vLLM, accelerate, datasets, lm-eval, etc.) during execution as needed.

### 3. Run the benchmark

```bash
# Default: uses Anthropic API key
python run_benchmark.py

# Alternative: use AWS Bedrock
python run_benchmark.py --bedrock
```

This runs the full pipeline:

1. **Phase 1 (Baseline)** -- Runs Claude Code with `proposal.md` as the task. The agent builds and executes the full ML pipeline in an isolated sandbox with no KB access.
2. **Phase 2 (With KB)** -- Runs Claude Code with the same `proposal.md` plus Leeroopedia MCP tools connected. The agent can call `search_knowledge`, `build_plan`, `query_hyperparameter_priors`, `diagnose_failure`, etc.

Both agents work in isolated `/tmp` sandboxes. Generated code and model artifacts are copied to `workspaces/proposal_benchmark/baseline/` and `workspaces/proposal_benchmark/with_kb/` after each phase.

> **Tip:** For better Leeroopedia usage, append the contents of [`leeroopedia_tools_usage.md`](leeroopedia_tools_usage.md) to the end of `proposal.md`. This teaches the agent when and how to call the KB tools (the benchmark already does this automatically for the with-KB phase).
