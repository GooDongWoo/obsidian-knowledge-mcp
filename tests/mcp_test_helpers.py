"""Real local transport harnesses; never start the user's model daemon."""

from contextlib import asynccontextmanager
import asyncio
import socket

import anyio
import uvicorn
from sse_starlette.sse import AppStatus


@asynccontextmanager
async def serve_http(mcp, requests=None, *, json_response=False):
    # sse-starlette's process-wide drain flag assumes one Uvicorn server per
    # process. These tests reuse the interpreter for many servers; an old
    # shutdown watcher otherwise closes a later server's legacy SSE response.
    # FastMCP's lifespan still terminates every transport on explicit shutdown.
    AppStatus.disable_automatic_graceful_drain()
    AppStatus.should_exit = False
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    port = listener.getsockname()[1]
    app = mcp.http_app(
        path="/mcp", host_origin_protection=True,
        allowed_hosts=["127.0.0.1", "localhost"], allowed_origins=[],
        json_response=json_response,
    )
    if requests is not None:
        original_app = app

        async def record_app(scope, receive, send):
            if scope["type"] != "http":
                return await original_app(scope, receive, send)
            record = {"method": scope["method"], "headers": dict(scope["headers"]), "body": b""}
            requests.append(record)

            async def record_receive():
                message = await receive()
                if message["type"] == "http.request":
                    record["body"] += message.get("body", b"")
                return message

            await original_app(scope, record_receive, send)

        app = record_app
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="on"))
    task = asyncio.create_task(server.serve(sockets=[listener]))
    try:
        with anyio.fail_after(10):
            while not server.started:
                if task.done():
                    await task
                await anyio.sleep(0.01)
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        server.should_exit = True
        with anyio.CancelScope(shield=True):
            await task
        listener.close()
