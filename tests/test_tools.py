"""Tool definitions exposed over MCP (tools.py)."""

import pytest
from mcp.types import Tool

from leeroopedia_mcp.tools import TOOL_NAMES, get_tool_definitions

DEFINITIONS = get_tool_definitions()


def test_eight_tools_are_defined():
    assert len(DEFINITIONS) == 8


def test_definitions_match_the_dispatch_table():
    # server.py only forwards names in TOOL_NAMES, so a tool missing from
    # either side would be listed but rejected, or callable but unlisted.
    names = [d["name"] for d in DEFINITIONS]

    assert len(names) == len(set(names)), "duplicate tool name"
    assert set(names) == TOOL_NAMES


@pytest.mark.parametrize("definition", DEFINITIONS, ids=lambda d: d["name"])
class TestEachTool:
    def test_is_a_valid_mcp_tool(self, definition):
        tool = Tool(**definition)

        assert tool.name == definition["name"]
        assert tool.inputSchema == definition["inputSchema"]

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
