import json

import pytest
from mcp import Client

from statistical_mcp.server import mcp


@pytest.mark.asyncio
async def test_tool_discovery_and_equivalent_structured_text() -> None:
    async with Client(mcp) as client:
        tools = (await client.list_tools()).tools
        assert {tool.name for tool in tools} == {
            "prophet_model",
            "prophet_next",
            "correlation",
            "contingency",
            "classification_metrics",
        }
        assert all(tool.input_schema and tool.output_schema for tool in tools)
        result = await client.call_tool("correlation", {"x": [1, 2, 3], "y": [2, 4, 6]})
        assert result.is_error is False
        assert result.structured_content is not None
        assert json.loads(result.content[0].text) == result.structured_content
        assert "\n" not in result.content[0].text


@pytest.mark.asyncio
async def test_validation_failure_is_a_tool_error() -> None:
    async with Client(mcp) as client:
        result = await client.call_tool("correlation", {"x": [1, 2], "y": [1]})
        assert result.is_error is True
        assert "lengths must match" in result.content[0].text
