# Leeroopedia MCP Server

<p align="center">
  <strong>Give your AI coding agent access to best-practices of ML and AI.</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/leeroopedia-mcp/"><img src="https://img.shields.io/pypi/v/leeroopedia-mcp?color=blue" alt="PyPI"></a>
  <a href="https://discord.gg/hqVbPNNEZM"><img src="https://dcbadge.limes.pink/api/server/hqVbPNNEZM?style=flat" alt="Discord"></a>
  <a href="https://github.com/Leeroo-AI/leeroopedia-mcp"><img src="https://img.shields.io/github/commit-activity/m/Leeroo-AI/leeroopedia-mcp" alt="GitHub commit activity"></a>
  <a href="https://www.ycombinator.com/companies/leeroo"><img src="https://img.shields.io/badge/Y%20Combinator-X25-orange?logo=ycombinator&logoColor=white" alt="Y Combinator X25"></a>
</p>

---

> **$20 free credit on sign-up** : that's plenty of searches, plans, and diagnoses. Skip the guesswork on your next fine-tuning run or inference deployment. No credit card required. [Get your API key →](https://app.leeroopedia.com)

## What is Leeroopedia?

**Your ML & AI Knowledge Wiki.** Learnt by AI, built by AI, for AI.

Expert-level knowledge across the full ML & AI stack — from fine-tuning and distributed training, to inference serving and GPU kernel optimization, to building agents and RAG pipelines. **1000+ frameworks and libraries**, all in one place.

This MCP server turns your AI coding agent (Claude Code, Cursor) into an ML/AI expert engineer.

Browse the full knowledge base at [leeroopedia.com](https://leeroopedia.com).

### Want to go end-to-end?

Leeroopedia gives your agent the **knowledge**. [**Kapso**](https://github.com/leeroo-ai/kapso) gives it the **ability to act on it** : research, experiment, and deploy. Together: a complete ML/AI engineer agent.

## Benchmarks

- **ML Inference Optimization** — Write CUDA/Triton kernels for 10 KernelBench problems. **2.11x** geomean speedup vs 1.80x (**+17%**), with/without Leeroopedia MCP. [→ results](examples/ml_inference_optimization/)

- **LLM Post-Training** — End-to-end SFT + DPO + LoRA merge + vLLM serving + IFEval on 8×A100. **21.3 vs 18.5** IFEval strict-prompt accuracy, **34.6 vs 30.9** strict-instruction accuracy, **272.7 vs 231.6** throughput. [→ results](examples/llm_post_training/)

- **Self-Evolving RAG** — Build a RAG service that automatically improves itself over multiple rounds. **45.16 vs 40.51** Precision@5, **40.32 vs 35.29** Recall@5, in **52 vs 62 min** wall time. [→ results](examples/self_evolve_rag/)

- **Customer Support Agent** — Multi-agent triage system classifying 200 tickets into 27 intents. **98 vs 83** benchmark performance, **11s vs 61s** per query. [→ results](examples/customer_support_agent/)

## Quick Start

### 1. Install

No installation needed if you have [uv](https://docs.astral.sh/uv/). The MCP configs below use `uvx` to auto-download and run.

**Alternative** (manual install):

```bash
pip install leeroopedia-mcp
```

### 2. Get Your API Key

1. Go to [app.leeroopedia.com](https://app.leeroopedia.com)
2. Create an account or log in
3. Navigate to **Dashboard > API Keys**
4. Copy your API key (format: `kpsk_...`)

### 3. Configure Claude Code

Add to your `~/.claude.json` or project `.mcp.json`:

```json
{
  "mcpServers": {
    "leeroopedia": {
      "command": "uvx",
      "args": ["leeroopedia-mcp"],
      "env": {
        "LEEROOPEDIA_API_KEY": "kpsk_your_key_here"
      }
    }
  }
}
```

### 4. Configure Cursor

Add to your Cursor settings (`.cursor/mcp.json`):

```json
{
  "mcpServers": {
    "leeroopedia": {
      "command": "uvx",
      "args": ["leeroopedia-mcp"],
      "env": {
        "LEEROOPEDIA_API_KEY": "kpsk_your_key_here"
      }
    }
  }
}
```

## Available Tools

The MCP server provides **8 agentic tools**. Each tool (except `get_page`) triggers an AI agent on the backend that searches the knowledge base from multiple angles, reads relevant pages, and synthesizes a structured response.

### Search & Retrieve

<details>
<summary><b><code>search_knowledge</code></b> — Search the KB for framework docs, APIs, and best practices</summary>
<br>

An AI agent synthesizes a grounded answer with `[PageID]` citations.

| Parameter | Required | Description |
|-----------|----------|-------------|
| `query` | Yes | What you want to find out |
| `context` | No | Optional context about what you're building |

</details>

<details>
<summary><b><code>get_page</code></b> — Retrieve a specific KB page by ID</summary>
<br>

Direct lookup — no AI agent needed. Use this to drill into `[PageID]` citations from other tools.

| Parameter | Required | Description |
|-----------|----------|-------------|
| `page_id` | Yes | Exact page ID (e.g., `Workflow/QLoRA_Finetuning`, `Principle/LoRA_Rank_Selection`) |

</details>

### Plan & Review

<details>
<summary><b><code>build_plan</code></b> — Build a step-by-step ML execution plan</summary>
<br>

Returns an overview, key specs, numbered steps, and validation criteria — all grounded in KB evidence.

| Parameter | Required | Description |
|-----------|----------|-------------|
| `goal` | Yes | What you want to accomplish |
| `constraints` | No | Constraints or requirements (e.g., hardware limits, time budget) |

</details>

<details>
<summary><b><code>review_plan</code></b> — Review a plan against KB best practices</summary>
<br>

Catches incorrect assumptions before you write code. Returns approvals, risks, and improvement suggestions.

| Parameter | Required | Description |
|-----------|----------|-------------|
| `proposal` | Yes | The plan or proposal to review |
| `goal` | Yes | The intended goal of the plan |

</details>

### Verify & Debug

<details>
<summary><b><code>verify_code_math</code></b> — Verify code against ML/math concepts</summary>
<br>

Checks your code against documented behavior and reference implementations. Returns a Pass/Fail verdict with analysis.

| Parameter | Required | Description |
|-----------|----------|-------------|
| `code_snippet` | Yes | The code to verify |
| `concept_name` | Yes | The mathematical/ML concept being implemented |

</details>

<details>
<summary><b><code>diagnose_failure</code></b> — Diagnose training/deployment failures</summary>
<br>

Matches symptoms against known failure patterns and misconfigurations. Returns diagnosis, fix steps, and prevention advice.

| Parameter | Required | Description |
|-----------|----------|-------------|
| `symptoms` | Yes | Description of the failure symptoms |
| `logs` | Yes | Relevant log output or error messages |

</details>

### Explore & Optimize

<details>
<summary><b><code>propose_hypothesis</code></b> — Propose ranked next-step hypotheses</summary>
<br>

When you're stuck, get alternative approaches ranked by fit — backed by documented patterns. Returns ranked ideas with rationale and suggested experiments.

| Parameter | Required | Description |
|-----------|----------|-------------|
| `current_status` | Yes | Where the project stands now |
| `recent_experiments` | No | Description of recent experiments and their outcomes |

</details>

<details>
<summary><b><code>query_hyperparameter_priors</code></b> — Query hyperparameter values, ranges & heuristics</summary>
<br>

Start with battle-tested defaults instead of guessing. Returns a suggestion table with KB-grounded justification.

| Parameter | Required | Description |
|-----------|----------|-------------|
| `query` | Yes | Hyperparameter question (e.g., "learning rate for LoRA fine-tuning Llama-3 8B") |

</details>

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `LEEROOPEDIA_API_KEY` | Yes | — | Your Leeroopedia API key |
| `LEEROOPEDIA_API_URL` | No | `https://api.leeroopedia.com` | API endpoint |
| `LEEROOPEDIA_POLL_MAX_WAIT` | No | `300` | Max seconds to wait for a search task |
| `LEEROOPEDIA_POLL_INTERVAL` | No | `0.5` | Initial poll interval in seconds (grows via backoff) |

## Troubleshooting

| Error | Fix |
|-------|-----|
| `LEEROOPEDIA_API_KEY is required` | Set `LEEROOPEDIA_API_KEY` in your MCP config `env` block |
| `Invalid or revoked API key` (401) | Re-copy your key from [app.leeroopedia.com](https://app.leeroopedia.com) |
| `Insufficient credits` (402) | Purchase more credits at [app.leeroopedia.com](https://app.leeroopedia.com) |
| `Rate limit exceeded` (429) | Wait for the retry period before making more requests |
| `Search timed out` (504) | Try a more specific query, or increase `LEEROOPEDIA_POLL_MAX_WAIT` |

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details on how to get started.

This project follows our [Code of Conduct](CODE_OF_CONDUCT.md).

## License

This project is licensed under the [MIT License](LICENSE).
