# Contributing to Leeroopedia MCP

Thank you for your interest in contributing! This guide will help you get started.

## Getting Started

### Prerequisites

- Python 3.10+
- Git

### Development Setup

```bash
# Clone the repository
git clone https://github.com/Leeroo-AI/leeroopedia-mcp.git
cd leeroopedia-mcp

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in editable mode
pip install -e .
```

### Environment Variables

Create a `.env` file or export directly:

```bash
LEEROOPEDIA_API_KEY=kpsk_your_key_here
```

Get your API key at [app.leeroopedia.com](https://app.leeroopedia.com).

## Making Changes

### Code Style

- Write clean, simple, readable code
- Keep files small and focused (<200 lines when possible)
- Use clear, consistent naming
- Add helpful comments to explain non-obvious logic

### Linting

```bash
black leeroopedia_mcp/
flake8 leeroopedia_mcp/
```

## Submitting Changes

### Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes
4. Run linting
5. Commit with a clear message
6. Push to your fork
7. Open a Pull Request

### Commit Messages

- Use present tense ("Add feature" not "Added feature")
- Keep the first line under 72 characters
- Reference issues when applicable (`Fixes #123`)

### PR Guidelines

- Keep PRs focused on a single change
- Include a description of what changed and why
- Update documentation if needed

## Project Structure

```
leeroopedia-mcp/
├── leeroopedia_mcp/      # Main package
│   ├── __init__.py       # Version
│   ├── server.py         # MCP server entry point
│   ├── client.py         # HTTP client (async task API)
│   ├── config.py         # Configuration from env vars
│   └── tools.py          # MCP tool definitions (8 agentic tools)
├── .github/workflows/    # CI/CD (PyPI publish on release)
├── pyproject.toml        # Package metadata
└── README.md
```

## Getting Help

- **Discord**: [Join our community](https://discord.gg/hqVbPNNEZM)
- **Issues**: Open a GitHub issue for bugs or feature requests

## Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
