"""Stdio protocol server bridging clients to the shared HTTP model daemon."""

from __future__ import annotations

from contextlib import asynccontextmanager
import asyncio
import logging
import math
import os
import socket
import ssl
from typing import Any
import urllib.parse
import urllib.request

import anyio
from fastmcp.server import create_proxy
from fastmcp.client.transports import StreamableHttpTransport
from fastmcp.server.middleware import Middleware
from fastmcp.tools import ToolResult

logger = logging.getLogger(__name__)

DEFAULT_RESTART_MESSAGE = (
    "[Knowledge MCP] 데몬 서버가 꺼져 있어 즉시 재기동을 시작했습니다. "
    "잠시 후(10~20초 뒤) 다시 시도해 주세요."
)


def is_daemon_running(port: int = 8765, host: str = "127.0.0.1") -> bool:
    """Check readiness without protocol sessions, ping, or event streams."""
    try:
        with socket.create_connection((host, port), timeout=0.05):
            pass
        with urllib.request.urlopen(f"http://{host}:{port}/health", timeout=0.2) as response:
            return response.status == 200
    except Exception:
        return False


def start_daemon_process(settings: Any = None, port: int = 8765,
                         host: str = "127.0.0.1", timeout: float | None = None) -> int:
    from .daemon import start_daemon_process as start

    return start(settings=settings, port=port, host=host, timeout=timeout)


class _DaemonGuard(Middleware):
    """Keep restart and timeout policy outside SDK protocol dispatch."""

    def __init__(self, host: str, port: int, settings: Any, timeout: float):
        self.host, self.port, self.settings, self.timeout = host, port, settings, timeout
        self.restart_task: asyncio.Task | None = None

    async def alive(self) -> bool:
        return await anyio.to_thread.run_sync(
            lambda: is_daemon_running(port=self.port, host=self.host)
        )

    def restart(self) -> None:
        if self.restart_task is not None and not self.restart_task.done():
            return

        async def start() -> None:
            try:
                await anyio.to_thread.run_sync(
                    lambda: start_daemon_process(self.settings, port=self.port, host=self.host),
                    abandon_on_cancel=True,
                )
            except Exception:
                logger.exception("Could not restart the knowledge daemon")
        self.restart_task = asyncio.create_task(start())

    async def close(self) -> None:
        if self.restart_task is not None:
            self.restart_task.cancel()
            await asyncio.gather(self.restart_task, return_exceptions=True)

    async def on_call_tool(self, context, call_next):
        if not await self.alive():
            self.restart()
            return ToolResult(content=DEFAULT_RESTART_MESSAGE, is_error=True)
        try:
            with anyio.fail_after(self.timeout):
                return await call_next(context)
        except TimeoutError:
            if not await self.alive():
                self.restart()
            return ToolResult(
                content=f"[Knowledge MCP] 요청 처리 시간 초과 ({self.timeout}초). "
                        "데몬 서버가 응답하지 않습니다. 작업 완료 여부는 상태를 확인해 주세요.",
                is_error=True,
            )


def create_stdio_proxy(daemon_url: str = "http://127.0.0.1:8765/mcp",
                       settings: Any = None, timeout: float | None = None):
    parsed = urllib.parse.urlparse(daemon_url)
    if timeout is None:
        try:
            timeout = float(os.environ.get("KNOWLEDGE_PROXY_TIMEOUT", "15.0"))
        except ValueError:
            timeout = 15.0
        if not math.isfinite(timeout) or timeout <= 0:
            timeout = 15.0
    elif not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Proxy timeout must be finite and positive")

    guard = _DaemonGuard(parsed.hostname or "127.0.0.1", parsed.port or 8765, settings, timeout)

    @asynccontextmanager
    async def lifespan(server):
        # FastMCP owns its lifespan stack; do not span its yield with an AnyIO
        # cancel scope, which conflicts with the SDK's nested shutdown scopes.
        try:
            yield
        finally:
            await guard.close()

    # The official factory keeps fresh backend sessions for era isolation.
    # Reuse only the certificate context: rebuilding the Windows trust store
    # for every local HTTP call otherwise adds hundreds of milliseconds.
    target = StreamableHttpTransport(daemon_url, verify=ssl.create_default_context())
    # A transport target (not a configured Client) retains SDK era mirroring.
    # Do not pin a mode or intercept server/discover here.
    return create_proxy(
        target, name="obsidian-knowledge-proxy", provider_error_strategy="raise",
        middleware=[guard], lifespan=lifespan,
    )


async def run_stdio_proxy(daemon_url: str = "http://127.0.0.1:8765/mcp",
                          settings: Any = None, timeout: float | None = None) -> None:
    await create_stdio_proxy(daemon_url, settings, timeout).run_async(
        transport="stdio", show_banner=False,
    )
