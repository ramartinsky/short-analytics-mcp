import json
import subprocess
import sys
import textwrap

import pytest
from mcp import Client

from statistical_mcp.server import mcp


@pytest.mark.asyncio
async def test_tool_discovery_and_equivalent_structured_text() -> None:
    async with Client(mcp) as client:
        tools = (await client.list_tools()).tools
        assert {tool.name for tool in tools} == {
            # "prophet_model",
            # "prophet_next",
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


def test_server_starts_without_prophet_dependencies() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
                import importlib.abc
                import sys

                class BlockProphetImports(importlib.abc.MetaPathFinder):
                    def find_spec(self, fullname, path=None, target=None):
                        if fullname.split(".")[0] in {"prophet", "cmdstanpy", "matplotlib"} or fullname == "statistical_mcp.prophet_tools":
                            raise RuntimeError(f"Server attempted an optional import: {fullname}")

                sys.meta_path.insert(0, BlockProphetImports())
                from statistical_mcp.server import mcp
                """
            ),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.asyncio
async def test_validation_failure_is_a_tool_error() -> None:
    async with Client(mcp) as client:
        result = await client.call_tool("correlation", {"x": [1, 2], "y": [1]})
        assert result.is_error is True
        assert "lengths must match" in result.content[0].text
