#!/usr/bin/env python3
"""
Leeroopedia MCP Server

MCP server for searching Leeroopedia's curated ML/AI knowledge base.
Runs as a stdio server for Claude Code / Cursor integration.

Usage:
    LEEROOPEDIA_API_KEY=kpsk_... leeroopedia-mcp

Environment Variables:
    LEEROOPEDIA_API_KEY: Required. Your Leeroopedia API key.
    LEEROOPEDIA_API_URL: Optional. API URL (default: https://api.leeroopedia.com)
"""

import asyncio
import logging
import sys
from typing import Any, Dict, List

from . import __version__
from .config import Config, validate_config_or_exit
from .client import (
    LeeroopediaClient,
    AuthenticationError,
    InsufficientCreditsError,
    RateLimitError,
    TaskTimeoutError,
    APIError,
)
from .tools import get_tool_definitions, REQUIRED_ARGUMENTS, TOOL_NAMES

# Configure logging to stderr (stdout is for MCP protocol)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

# MCP imports
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import CallToolResult, ListToolsResult, Tool, TextContent
    HAS_MCP = True
except ImportError:
    HAS_MCP = False
    Server = None
    Tool = None
    TextContent = None
    CallToolResult = None
    ListToolsResult = None

# The two supported SDK lines register handlers differently:
#   mcp 1.x - decorators on the Server instance (@server.list_tools())
#   mcp 2.x - constructor arguments (Server(..., on_list_tools=...))
# Everything else in this module is shared between them.
USES_DECORATOR_API = HAS_MCP and hasattr(Server, "list_tools")


def build_tools() -> List["Tool"]:
    """Build the MCP Tool objects advertised to the client."""
    return [
        Tool(
            name=t["name"],
            description=t["description"],
            inputSchema=t["inputSchema"],
        )
        for t in get_tool_definitions()
    ]


async def run_tool(
    client: LeeroopediaClient,
    name: str,
    arguments: Dict[str, Any],
) -> str:
    """
    Run one tool call and return the text shown to the agent.

    Forwards the full arguments dict to the backend. Every failure is
    returned as readable text instead of raised, so the agent can act on it.
    Independent of the MCP SDK version in use.

    Args:
        client: Backend HTTP client
        name: Tool name requested by the agent
        arguments: Tool arguments as sent by the agent

    Returns:
        Result or error text
    """
    if name not in TOOL_NAMES:
        return f"Unknown tool: {name}. Available: {', '.join(sorted(TOOL_NAMES))}"

    # mcp 1.x validates arguments against the tool schema before the handler
    # runs; 2.x does not. Check here so a malformed call never reaches the
    # backend (and never spends a credit) on either SDK.
    missing = [arg for arg in REQUIRED_ARGUMENTS[name] if arg not in arguments]
    if missing:
        return f"Missing required argument(s) for {name}: {', '.join(missing)}"

    try:
        response = await client.search(tool=name, arguments=arguments)

        text = response.results or "No results returned."
        # The gateway may not report a balance - only show it when known
        if response.credits_remaining is not None:
            text += f"\n\n---\n*Credits remaining: {response.credits_remaining}*"
        return text

    except AuthenticationError as e:
        return f"Authentication error: {e}\n\nPlease check your LEEROOPEDIA_API_KEY."

    except InsufficientCreditsError as e:
        return f"Insufficient credits: {e}\n\nPurchase more at https://app.leeroopedia.com"

    except RateLimitError as e:
        return f"Rate limit exceeded. Retry after {e.retry_after} seconds."

    except TaskTimeoutError as e:
        logger.warning(f"Search task timed out: {e}")
        return (
            f"Search timed out ({e.task_id}). "
            f"The search may still be processing. Try again or use a more specific query."
        )

    except APIError as e:
        logger.error(f"API error: {e}")
        return f"API error: {e}"

    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return f"Unexpected error: {e}"


def create_mcp_server(config: Config) -> "Server":
    """
    Create and configure the MCP server.

    Args:
        config: Validated configuration

    Returns:
        Configured MCP Server instance
    """
    if not HAS_MCP:
        raise ImportError("MCP package not installed. Install with: pip install mcp")

    client = LeeroopediaClient(config)

    if USES_DECORATOR_API:
        # mcp 1.x
        mcp = Server("leeroopedia")

        @mcp.list_tools()
        async def list_tools() -> List[Tool]:
            """List available tools."""
            return build_tools()

        @mcp.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            """Handle tool calls."""
            text = await run_tool(client, name, arguments)
            return [TextContent(type="text", text=text)]

        return mcp

    # mcp 2.x
    async def on_list_tools(ctx: Any, params: Any) -> ListToolsResult:
        """List available tools."""
        return ListToolsResult(tools=build_tools())

    async def on_call_tool(ctx: Any, params: Any) -> CallToolResult:
        """Handle tool calls."""
        text = await run_tool(client, params.name, params.arguments or {})
        return CallToolResult(content=[TextContent(type="text", text=text)])

    return Server(
        "leeroopedia",
        # 2.x reports an empty server version unless one is given
        version=__version__,
        on_list_tools=on_list_tools,
        on_call_tool=on_call_tool,
    )


async def run_server(config: Config) -> None:
    """Run the MCP server with stdio transport."""
    if not HAS_MCP:
        raise ImportError("MCP package not installed. Install with: pip install mcp")

    logger.info("Starting Leeroopedia MCP Server...")

    mcp = create_mcp_server(config)

    async with stdio_server() as (read_stream, write_stream):
        logger.info("MCP server running on stdio transport")
        await mcp.run(read_stream, write_stream, mcp.create_initialization_options())


def main():
    """CLI entry point."""
    config = validate_config_or_exit()

    try:
        asyncio.run(run_server(config))
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
