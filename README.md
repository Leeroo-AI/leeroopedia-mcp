# Leeroopedia MCP Server

MCP server for searching Leeroopedia's curated ML/AI knowledge base.

## Quick Start

### 1. Install

```bash
pip install leeroopedia-mcp
```

### 2. Get Your API Key

1. Go to [leeroopedia.com](https://leeroopedia.com)
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

## Available Tools

### wiki_idea_search

Search for conceptual knowledge:
- **Principles**: Theoretical concepts, fundamental ideas
- **Heuristics**: Best practices, rules of thumb

Example: "LoRA fine-tuning principles"

### wiki_code_search

Search for implementation knowledge:
- **Implementations**: Code patterns, API usage, algorithms
- **Environments**: Setup guides, configuration, dependencies

Example: "PyTorch LoRA implementation"

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `LEEROOPEDIA_API_KEY` | Yes | - | Your Leeroopedia API key |
| `LEEROOPEDIA_API_URL` | No | `https://api.leeroopedia.com` | API endpoint |

## Troubleshooting

### "LEEROOPEDIA_API_KEY is required"

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

### "Invalid or revoked API key" (401)

Double-check your API key in the Dashboard. Re-copy if needed.

### "Insufficient credits" (402)

Purchase more credits at https://leeroopedia.com/dashboard

### "Rate limit exceeded" (429)

Wait for the retry period before making more requests.

## License

MIT
