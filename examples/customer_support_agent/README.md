# Customer Support Triage Benchmark

This benchmark measures how a curated knowledge base (Leeroopedia MCP) improves an AI coding agent's ability to build a production-quality multi-agent system from a specification.

Both agents receive the **same task**: build a FastAPI service that classifies 200 customer support tickets into 27 fine-grained intent categories using a multi-agent team with handoffs, state persistence, and structured output. The only difference is that one agent has access to Leeroopedia's framework-specific docs via MCP tools.

## Results

![Benchmark Results](analysis.png)

### Why +Leeroopedia MCP performed better

The core differentiator was a single KB lookup that taught the agent a framework-specific pattern: **typed handoff callbacks with `input_type`**.

- **Baseline** : The router tells the LLM to include `"INTENT:get_refund"` as free text, then a regex tries to extract it after the conversation. This is fragile: the LLM sometimes omits the prefix, formats it differently, or the regex misses it entirely.
- **With Leeroopedia** : Each handoff is registered with `input_type=IntentHandoffInput` (a Pydantic model). The SDK **forces** the LLM to fill a structured `intent_category` field when calling the handoff tool. An `on_handoff` callback captures it programmatically at handoff time.

The Leeroopedia also taught proper state serialization (`to_state()` instead of `str(msg)`), correct `output_type` usage for structured specialist responses, and fixed three runtime errors during debugging.

---

## How to replicate

Full logs and outputs from our runs are available [here](https://drive.google.com/file/d/1rl3TsDL3Xm_JPbl24npfIpEVQZuVZsOR/view?usp=sharing).

### Prerequisites

- Python 3.10+
- [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) installed and configured
- An Anthropic API key **or** AWS Bedrock access
- A Leeroopedia API key
- An OpenAI API key (used by the agents to call GPT for the multi-agent framework)

### 1. Set up environment variables

Copy `.env.example` or create a `.env` file in this directory:

```
OPENAI_API_KEY=sk-...
LEEROOPEDIA_API_KEY="kpsk_..."
ANTHROPIC_API_KEY="sk-ant-..."

# Only needed if using --bedrock mode
# AWS_BEARER_TOKEN_BEDROCK="..."
# CLAUDE_CODE_USE_BEDROCK=1
# AWS_REGION=us-east-1
```

### 2. Install Python dependencies

```bash
pip install python-dotenv datasets
```

The benchmark pre-installs additional packages (`fastapi`, `uvicorn`, `openai`, `openai-agents`, `pydantic`, `httpx`, `sse-starlette`) before running the agents so both get the same environment.

### 3. Run the benchmark

```bash
# Default: uses Anthropic API key
python run_benchmark.py

# Alternative: use AWS Bedrock
python run_benchmark.py --bedrock
```

This runs the full pipeline:

1. **Phase 0** -- Builds a deterministic 200-ticket test corpus from the [Bitext Customer Support Intent Dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) (~7-8 tickets per intent across 27 categories).
2. **Phase 1 (Baseline)** -- Runs Claude Code with `proposal.md` as the task. The agent builds and tests the service in an isolated sandbox with no KB access.
3. **Phase 2 (With Leeroopedia)** -- Runs Claude Code with the same `proposal.md` plus Leeroopedia MCP tools. The agent can call `search_knowledge`, `build_plan`, `diagnose_failure`, etc.

Both agents work in isolated `/tmp` sandboxes. Generated code is copied to `workspaces/proposal_benchmark/baseline/` and `workspaces/proposal_benchmark/with_kb/` after each phase.

> **Tip:** For better Leeroopedia usage, append the contents of [`leeroopedia_tools_reference.md`](leeroopedia_tools_reference.md) to the end of `proposal.md`. This teaches the agent when and how to call the KB tools (the benchmark already does this automatically for the with-KB phase).
