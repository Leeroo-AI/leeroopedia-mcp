"""
Leeroopedia MCP Server

MCP server for searching Leeroopedia's curated ML/AI knowledge base.
Runs over the stdio transport.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    # Single source of truth: the version field in pyproject.toml
    __version__ = version("leeroopedia-mcp")
except PackageNotFoundError:
    # Running from a source checkout that was never installed
    __version__ = "0.0.0+unknown"
