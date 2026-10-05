"""Tool definitions exposed over MCP (tools.py)."""

import pytest

from leeroopedia_mcp.server import build_tools
from leeroopedia_mcp.tools import REQUIRED_ARGUMENTS, TOOL_NAMES, get_tool_definitions

DEFINITIONS = get_tool_definitions()


def test_eight_tools_are_defined():
    assert len(DEFINITIONS) == 8


def test_definitions_match_the_dispatch_table():
    # server.py only forwards names in TOOL_NAMES, so a tool missing from
    # either side would be listed but rejected, or callable but unlisted.
    names = [d["name"] for d in DEFINITIONS]

    assert len(names) == len(set(names)), "duplicate tool name"
    assert set(names) == TOOL_NAMES


def test_required_arguments_cover_every_tool():
    assert set(REQUIRED_ARGUMENTS) == TOOL_NAMES
    assert REQUIRED_ARGUMENTS["search_knowledge"] == ["query"]
    assert REQUIRED_ARGUMENTS["review_plan"] == ["proposal", "goal"]


def test_definitions_build_valid_mcp_tools():
    # Goes through the installed SDK's Tool model, so a definition the SDK
    # rejects fails here rather than when a client connects.
    tools = build_tools()

    assert [tool.name for tool in tools] == [d["name"] for d in DEFINITIONS]
    for tool, definition in zip(tools, DEFINITIONS):
        dumped = tool.model_dump(by_alias=True)
        assert dumped["description"] == definition["description"]
        assert dumped["inputSchema"] == definition["inputSchema"]


@pytest.mark.parametrize("definition", DEFINITIONS, ids=lambda d: d["name"])
class TestEachTool:
    def test_has_a_description(self, definition):
        assert definition["description"].strip()

    def test_schema_is_an_object_with_string_properties(self, definition):
        schema = definition["inputSchema"]

        assert schema["type"] == "object"
        assert schema["properties"], "tool takes no arguments"
        for name, prop in schema["properties"].items():
            assert prop["type"] == "string", name
            assert prop["description"].strip(), name

    def test_required_arguments_are_declared(self, definition):
        schema = definition["inputSchema"]

        assert schema["required"], "every tool needs at least one required argument"
        assert set(schema["required"]) <= set(schema["properties"])
