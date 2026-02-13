# Leeroopedia MCP Server

<p align="center">
  <strong>Give your AI coding agent access to curated ML/AI knowledge.</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/leeroopedia-mcp/"><img src="https://img.shields.io/pypi/v/leeroopedia-mcp?color=blue" alt="PyPI"></a>
  <a href="https://discord.gg/hqVbPNNEZM"><img src="https://dcbadge.limes.pink/api/server/hqVbPNNEZM?style=flat" alt="Discord"></a>
  <a href="https://github.com/Leeroo-AI/leeroopedia-mcp"><img src="https://img.shields.io/github/commit-activity/m/Leeroo-AI/leeroopedia-mcp" alt="GitHub commit activity"></a>
  <a href="https://www.ycombinator.com/companies/leeroo"><img src="https://img.shields.io/badge/Y%20Combinator-X25-orange?logo=ycombinator&logoColor=white" alt="Y Combinator X25"></a>
</p>

---

## What is Leeroopedia?

**Your ML & Data Knowledge Wiki.** Learnt by AI, built by AI, for AI. A centralized playbook of best practices and expert-level knowledge for Machine Learning and Data domains.

Browse the full knowledge base at [leeroopedia.com](https://leeroopedia.com). Apply for early beta access.

This MCP server lets AI coding agents (Claude Code, Cursor) search that knowledge base directly while they work — no copy-pasting needed.

---

## Quick Start

### 1. Install

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
      "command": "leeroopedia-mcp",
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
      "command": "leeroopedia-mcp",
      "env": {
        "LEEROOPEDIA_API_KEY": "kpsk_your_key_here"
      }
    }
  }
}
```

---

## Available Tools

The MCP server provides **8 agentic tools**. Each tool (except `get_page`) triggers an AI agent on the backend that searches the knowledge base from multiple angles, reads relevant pages, and synthesizes a structured response.

| Tool | What it does |
|------|-------------|
| [`search_knowledge`](#search_knowledge) | Search the KB for framework docs, APIs, and best practices |
| [`build_plan`](#build_plan) | Build a step-by-step ML execution plan |
| [`review_plan`](#review_plan) | Review a plan against KB best practices |
| [`verify_code_math`](#verify_code_math) | Verify code against authoritative math/ML descriptions |
| [`diagnose_failure`](#diagnose_failure) | Diagnose training/deployment failures |
| [`propose_hypothesis`](#propose_hypothesis) | Propose ranked next-step hypotheses |
| [`query_hyperparameter_priors`](#query_hyperparameter_priors) | Query hyperparameter values, ranges & heuristics |
| [`get_page`](#get_page) | Retrieve a specific KB page by ID |

<details>
<summary><b>search_knowledge</b> — Search the knowledge base</summary>

<br>

Search the knowledge base for framework documentation, API references, config formats, and best practices. An AI agent synthesizes a grounded answer with `[PageID]` citations.

| Parameter | Required | Description |
|-----------|----------|-------------|
| `query` | Yes | What you want to find out |
| `context` | No | Optional context about what you're building |

</details>

<details>
<summary><b>build_plan</b> — Build a step-by-step ML execution plan</summary>

<br>

Build a step-by-step ML execution plan grounded in knowledge base evidence. Returns an overview, key specs, numbered steps, and validation criteria.

| Parameter | Required | Description |
|-----------|----------|-------------|
| `goal` | Yes | What you want to accomplish |
| `constraints` | No | Constraints or requirements (e.g., hardware limits, time budget) |

</details>

<details>
<summary><b>review_plan</b> — Review a plan against best practices</summary>

<br>

Review a proposed ML plan against knowledge base best practices. Returns approvals, risks, and improvement suggestions.

| Parameter | Required | Description |
|-----------|----------|-------------|
| `proposal` | Yes | The plan or proposal to review |
| `goal` | Yes | The intended goal of the plan |

</details>

<details>
<summary><b>verify_code_math</b> — Verify code against ML/math concepts</summary>

<br>

Verify code correctness against authoritative ML/math concept descriptions. Returns a Pass/Fail verdict with analysis.

| Parameter | Required | Description |
|-----------|----------|-------------|
| `code_snippet` | Yes | The code to verify |
| `concept_name` | Yes | The mathematical/ML concept being implemented |

</details>

<details>
<summary><b>diagnose_failure</b> — Diagnose training/deployment failures</summary>

<br>

Diagnose ML training or deployment failures using knowledge base evidence. Returns diagnosis, fix steps, and prevention advice.

| Parameter | Required | Description |
|-----------|----------|-------------|
| `symptoms` | Yes | Description of the failure symptoms |
| `logs` | Yes | Relevant log output or error messages |

</details>

<details>
<summary><b>propose_hypothesis</b> — Propose ranked research hypotheses</summary>

<br>

Propose ranked research hypotheses grounded in knowledge base evidence. Returns ranked ideas with rationale and suggested experiments.

| Parameter | Required | Description |
|-----------|----------|-------------|
| `current_status` | Yes | Where the project stands now |
| `recent_experiments` | No | Description of recent experiments and their outcomes |

</details>

<details>
<summary><b>query_hyperparameter_priors</b> — Query hyperparameter heuristics</summary>

<br>

Query documented hyperparameter values, ranges, and tuning heuristics. Returns a suggestion table with KB-grounded justification.

| Parameter | Required | Description |
|-----------|----------|-------------|
| `query` | Yes | Hyperparameter question (e.g., "learning rate for LoRA fine-tuning Llama-3 8B") |

</details>

<details>
<summary><b>get_page</b> — Retrieve a KB page by ID</summary>

<br>

Retrieve the full content of a specific knowledge base page by its exact ID. A direct lookup — no AI agent needed.

| Parameter | Required | Description |
|-----------|----------|-------------|
| `page_id` | Yes | Exact page ID (e.g., `Workflow/QLoRA_Finetuning`, `Principle/LoRA_Rank_Selection`) |

</details>

---

## How It Works

The MCP server uses an **async task-based API**:

1. Your agent calls a tool (e.g., `search_knowledge`)
2. The MCP client sends `POST /v1/search` with the tool name and arguments
3. The backend queues the search task and returns a `task_id` immediately
4. The client polls `GET /v1/search/task/{task_id}` with exponential backoff
5. When the task completes, results are returned to your agent

This architecture allows the backend AI agents to take the time they need for thorough research without blocking or timing out.

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `LEEROOPEDIA_API_KEY` | Yes | — | Your Leeroopedia API key |
| `LEEROOPEDIA_API_URL` | No | `https://api.leeroopedia.com` | API endpoint |
| `LEEROOPEDIA_POLL_MAX_WAIT` | No | `300` | Max seconds to wait for a search task |
| `LEEROOPEDIA_POLL_INTERVAL` | No | `0.5` | Initial poll interval in seconds (grows via backoff) |

---

<details>
<summary><strong>Troubleshooting</strong></summary>

<br>

**"LEEROOPEDIA_API_KEY is required"**

Set your API key in the MCP config:

```json
{
  "mcpServers": {
    "leeroopedia": {
      "command": "leeroopedia-mcp",
      "env": {
        "LEEROOPEDIA_API_KEY": "kpsk_..."
      }
    }
  }
}
```

**"Invalid or revoked API key" (401)**

Double-check your API key at [app.leeroopedia.com](https://app.leeroopedia.com). Re-copy if needed.

**"Insufficient credits" (402)**

Purchase more credits at [app.leeroopedia.com](https://app.leeroopedia.com).

**"Rate limit exceeded" (429)**

Wait for the retry period before making more requests.

**"Search timed out" (504)**

The search task didn't complete within the poll window. Try a more specific query, or increase `LEEROOPEDIA_POLL_MAX_WAIT`.

</details>

---

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details on how to get started.

This project follows our [Code of Conduct](CODE_OF_CONDUCT.md).

## License

This project is licensed under the [MIT License](LICENSE).
