"""MCP server behaviour, exercised over a real in-memory MCP session (server.py)."""

import asyncio

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

import leeroopedia_mcp.server as server_module
from leeroopedia_mcp.client import (
    APIError,
    AuthenticationError,
    InsufficientCreditsError,
    RateLimitError,
    SearchResponse,
    TaskTimeoutError,
)
from leeroopedia_mcp.tools import TOOL_NAMES


def ok(results="# Answer [Page/One]", credits_remaining=99):
    return SearchResponse(
        success=True,
        results=results,
        latency_ms=42,
        credits_remaining=credits_remaining,
    )


class FakeClient:
    """Stands in for LeeroopediaClient so no HTTP is involved."""

    def __init__(self, outcome):
        self.outcome = outcome
        self.calls = []

    async def search(self, tool, arguments):
        self.calls.append((tool, arguments))
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


@pytest.fixture
def session_call(config, monkeypatch):
    """
    Start the server with a fake backend client, connect an MCP client
    session to it in memory, and run `action(session)`.
    """
    def _run(action, outcome=None):
        backend = FakeClient(outcome if outcome is not None else ok())
        monkeypatch.setattr(server_module, "LeeroopediaClient", lambda config: backend)

        async def _main():
            server = server_module.create_mcp_server(config)
            async with create_connected_server_and_client_session(server) as session:
                return await action(session)

        return asyncio.run(_main()), backend
    return _run


@pytest.fixture
def call_tool(session_call):
    """Call one tool and return (text of the reply, fake backend)."""
    def _call(outcome=None, name="search_knowledge", arguments=None):
        async def action(session):
            return await session.call_tool(name, arguments if arguments is not None else {"query": "q"})

        result, backend = session_call(action, outcome)
        assert len(result.content) == 1
        return result.content[0].text, backend
    return _call


def test_lists_all_eight_tools(session_call):
    async def action(session):
        return await session.list_tools()

    result, _ = session_call(action)

    assert {tool.name for tool in result.tools} == TOOL_NAMES
    assert len(result.tools) == 8


def test_tool_call_forwards_name_and_arguments_untouched(call_tool):
    arguments = {"goal": "fine-tune", "constraints": "1 GPU"}

    _, backend = call_tool(name="build_plan", arguments=arguments)

    assert backend.calls == [("build_plan", arguments)]


def test_result_is_returned_with_a_credits_footer(call_tool):
    text, _ = call_tool(ok(results="# Answer [Page/One]", credits_remaining=99))

    assert text == "# Answer [Page/One]\n\n---\n*Credits remaining: 99*"


def test_footer_is_omitted_when_the_balance_is_unknown(call_tool):
    text, _ = call_tool(ok(results="# Answer", credits_remaining=None))

    assert text == "# Answer"


def test_zero_balance_is_still_shown(call_tool):
    text, _ = call_tool(ok(credits_remaining=0))

    assert text.endswith("*Credits remaining: 0*")


def test_empty_result_is_reported_instead_of_a_blank_reply(call_tool):
    text, _ = call_tool(ok(results="", credits_remaining=5))

    assert text.startswith("No results returned.")
    assert "Credits remaining: 5" in text


def test_unknown_tool_is_rejected_without_calling_the_backend(call_tool):
    text, backend = call_tool(name="nope", arguments={})

    assert text.startswith("Unknown tool: nope")
    assert "search_knowledge" in text
    assert backend.calls == []


@pytest.mark.parametrize(
    "error, expected",
    [
        (AuthenticationError("Invalid or revoked API key"), "Authentication error: Invalid or revoked API key"),
        (InsufficientCreditsError(), "Insufficient credits: No credits remaining"),
        (RateLimitError(17), "Rate limit exceeded. Retry after 17 seconds."),
        (TaskTimeoutError("task-9", 300), "Search timed out (task-9)."),
        (APIError("agent crashed", "task_failure", 500), "API error: agent crashed"),
        (RuntimeError("something odd"), "Unexpected error: something odd"),
    ],
    ids=["auth", "credits", "rate-limit", "timeout", "api-error", "unexpected"],
)
def test_backend_errors_become_readable_text(call_tool, error, expected):
    # Errors are returned as text so the agent can read and act on them,
    # instead of the server crashing or surfacing a stack trace.
    text, _ = call_tool(error)

    assert text.startswith(expected)
