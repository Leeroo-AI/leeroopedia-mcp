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

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `LEEROOPEDIA_API_KEY` | Yes | — | Your Leeroopedia API key |
| `LEEROOPEDIA_API_URL` | No | `https://api.leeroopedia.com` | API endpoint |

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

</details>

---

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details on how to get started.

This project follows our [Code of Conduct](CODE_OF_CONDUCT.md).

## License

This project is licensed under the [MIT License](LICENSE).
