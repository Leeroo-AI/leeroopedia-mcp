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

from .config import Config, validate_config_or_exit
from .client import (
    LeeroopediaClient,
    AuthenticationError,
    InsufficientCreditsError,
    RateLimitError,
    TaskTimeoutError,
    APIError,
)
from .tools import get_tool_definitions, TOOL_NAMES

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
    from mcp.types import Tool, TextContent
    HAS_MCP = True
except ImportError:
    HAS_MCP = False
    Server = None
    Tool = None
    TextContent = None


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

    mcp = Server("leeroopedia")
    client = LeeroopediaClient(config)

    @mcp.list_tools()
    async def list_tools() -> List[Tool]:
        """List available tools."""
        definitions = get_tool_definitions()
        return [
            Tool(
                name=t["name"],
                description=t["description"],
                inputSchema=t["inputSchema"],
            )
            for t in definitions
        ]

    @mcp.call_tool()
    async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle tool calls — forward full arguments dict to the backend."""
        if name not in TOOL_NAMES:
            return [TextContent(
                type="text",
                text=f"Unknown tool: {name}. Available: {', '.join(sorted(TOOL_NAMES))}",
            )]

        try:
            response = await client.search(tool=name, arguments=arguments)

            footer = f"\n\n---\n*Credits remaining: {response.credits_remaining}*"
            return [TextContent(
                type="text",
                text=response.results + footer,
            )]

        except AuthenticationError as e:
            return [TextContent(
                type="text",
                text=f"Authentication error: {e}\n\nPlease check your LEEROOPEDIA_API_KEY.",
            )]

        except InsufficientCreditsError as e:
            return [TextContent(
                type="text",
                text=f"Insufficient credits: {e}\n\nPurchase more at https://app.leeroopedia.com",
            )]

        except RateLimitError as e:
            return [TextContent(
                type="text",
                text=f"Rate limit exceeded. Retry after {e.retry_after} seconds.",
            )]

        except TaskTimeoutError as e:
            logger.warning(f"Search task timed out: {e.task_id}")
            return [TextContent(
                type="text",
                text=f"Search timed out ({e.task_id}). "
                     f"The search may still be processing. Try again or use a more specific query.",
            )]

        except APIError as e:
            logger.error(f"API error: {e}")
            return [TextContent(
                type="text",
                text=f"API error: {e}",
            )]

        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            return [TextContent(
                type="text",
                text=f"Unexpected error: {e}",
            )]

    return mcp


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
