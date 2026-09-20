"""Lightweight stdio proxy bridging local MCP clients to the background SSE daemon."""

import anyio
from mcp.client.sse import sse_client
from mcp.server.stdio import stdio_server


async def run_stdio_proxy(daemon_url: str = "http://127.0.0.1:8765/sse") -> None:
    """Proxy JSON-RPC messages between stdio and the daemon SSE server."""
    async with stdio_server() as (stdio_read, stdio_write):
        async with sse_client(daemon_url) as (sse_read, sse_write):
            async with anyio.create_task_group() as tg:

                async def forward_stdio_to_sse() -> None:
                    try:
                        async for message in stdio_read:
                            if isinstance(message, Exception):
                                continue
                            await sse_write.send(message)
                    finally:
                        tg.cancel_scope.cancel()

                async def forward_sse_to_stdio() -> None:
                    try:
                        async for message in sse_read:
                            if isinstance(message, Exception):
                                continue
                            await stdio_write.send(message)
                    finally:
                        tg.cancel_scope.cancel()

                tg.start_soon(forward_stdio_to_sse)
                tg.start_soon(forward_sse_to_stdio)
