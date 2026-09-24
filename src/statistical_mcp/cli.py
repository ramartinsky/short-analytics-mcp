"""Command-line entrypoint supporting stdio and Streamable HTTP."""

from __future__ import annotations

import argparse

from .server import mcp


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Run the stateless statistical MCP server.")
    result.add_argument("--transport", choices=("streamable-http", "stdio"), default="streamable-http")
    result.add_argument("--host", default="0.0.0.0", help="HTTP bind host (ignored for stdio).")
    result.add_argument("--port", type=int, default=8000, help="HTTP bind port (ignored for stdio).")
    result.add_argument("--max-request-bytes", type=int, default=4 * 1024 * 1024, help="Maximum HTTP MCP request body.")
    return result


def main() -> None:
    args = parser().parse_args()
    if args.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        if not 1 <= args.port <= 65535:
            parser().error("--port must be between 1 and 65535")
        if args.max_request_bytes <= 0:
            parser().error("--max-request-bytes must be positive")
        mcp.run(
            transport="streamable-http",
            host=args.host,
            port=args.port,
            streamable_http_path="/mcp",
            stateless_http=True,
            json_response=True,
            max_request_body_size=args.max_request_bytes,
        )


if __name__ == "__main__":
    main()

