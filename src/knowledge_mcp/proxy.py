"""Lightweight stdio proxy bridging local MCP clients to the background SSE daemon."""

from __future__ import annotations

import json
import logging
import os
import socket
from typing import Any
import urllib.parse
import urllib.request

import anyio
from mcp.client.sse import sse_client
from mcp.server.stdio import stdio_server
from mcp.shared.session import SessionMessage
from mcp.types import ErrorData, JSONRPCError, JSONRPCMessage, JSONRPCResponse

logger = logging.getLogger(__name__)

DEFAULT_RESTART_MESSAGE = (
    "[Knowledge MCP] 데몬 서버가 꺼져 있어 즉시 재기동을 시작했습니다. "
    "잠시 후(10~20초 뒤) 다시 시도해 주세요."
)


def is_daemon_running(port: int = 8765, host: str = "127.0.0.1") -> bool:
    """Fast check if the daemon is reachable and responding."""
    try:
        with socket.create_connection((host, port), timeout=0.05):
            pass
    except OSError:
        return False

    try:
        req = urllib.request.Request(f"http://{host}:{port}/health")
        with urllib.request.urlopen(req, timeout=0.2) as response:
            return response.status == 200
    except Exception:
        return False


def start_daemon_process(
    settings: Any = None,
    port: int = 8765,
    host: str = "127.0.0.1",
    timeout: float = 60.0,
) -> int:
    """Trigger background daemon process start without importing heavy packages at module level."""
    from .daemon import start_daemon_process as _start

    return _start(settings=settings, port=port, host=host, timeout=timeout)


def _extract_id(message: Any) -> Any:
    """Extract JSON-RPC id from any message representation."""
    inner = getattr(message, "message", message)
    inner = getattr(inner, "root", inner)
    if hasattr(inner, "id"):
        return inner.id
    if isinstance(inner, dict):
        return inner.get("id")
    if isinstance(inner, str):
        try:
            data = json.loads(inner)
            if isinstance(data, dict):
                return data.get("id")
        except Exception:
            pass
    return None


def _extract_method(message: Any) -> str | None:
    """Extract a JSON-RPC method from any message representation."""
    inner = getattr(message, "message", message)
    inner = getattr(inner, "root", inner)
    if hasattr(inner, "method"):
        return inner.method
    if isinstance(inner, dict):
        return inner.get("method")
    if isinstance(inner, str):
        try:
            data = json.loads(inner)
            if isinstance(data, dict):
                return data.get("method")
        except Exception:
            pass
    return None


def _is_tools_call(message: Any) -> tuple[bool, Any]:
    """Check if message is a tools/call request, returning (is_call, request_id)."""
    inner = getattr(message, "message", message)
    inner = getattr(inner, "root", inner)
    if hasattr(inner, "method"):
        if inner.method == "tools/call":
            return True, getattr(inner, "id", None)
    elif isinstance(inner, dict):
        if inner.get("method") == "tools/call":
            return True, inner.get("id")
    elif isinstance(inner, str) and "tools/call" in inner:
        try:
            data = json.loads(inner)
            if isinstance(data, dict) and data.get("method") == "tools/call":
                return True, data.get("id")
        except Exception:
            pass
    return False, None


def _make_method_not_found_response(req_id: Any) -> SessionMessage:
    """Tell newer clients to fall back when the legacy daemon lacks discovery."""
    response = JSONRPCError(
        jsonrpc="2.0",
        id=req_id if req_id is not None else 0,
        error=ErrorData(code=-32601, message="Method not found"),
    )
    return SessionMessage(JSONRPCMessage(root=response))


def _make_error_response(req_id: Any, text: str) -> SessionMessage:
    """Create a JSON-RPC response SessionMessage indicating an error."""
    response = JSONRPCResponse(
        jsonrpc="2.0",
        id=req_id if req_id is not None else 0,
        result={
            "content": [
                {
                    "type": "text",
                    "text": text,
                }
            ],
            "isError": True,
        },
    )
    return SessionMessage(JSONRPCMessage(root=response))


async def run_stdio_proxy(
    daemon_url: str = "http://127.0.0.1:8765/sse",
    settings: Any = None,
    timeout: float | None = None,
) -> None:
    """Proxy JSON-RPC messages between stdio and the daemon SSE server."""
    parsed = urllib.parse.urlparse(daemon_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 8765

    proxy_timeout = timeout
    if proxy_timeout is None:
        try:
            proxy_timeout = float(os.environ.get("KNOWLEDGE_PROXY_TIMEOUT", "15.0"))
        except ValueError:
            proxy_timeout = 15.0

    async with stdio_server() as (stdio_read, stdio_write):
        async with anyio.create_task_group() as tg:
            in_flight: dict[Any, anyio.Event] = {}
            timed_out_ids: set[Any] = set()
            active_sse_write: Any = None
            connected_event = anyio.Event()
            reconnect_trigger = anyio.Event()
            _starting_daemon = False

            def trigger_start_daemon() -> None:
                nonlocal _starting_daemon
                if _starting_daemon:
                    return
                _starting_daemon = True

                async def _runner() -> None:
                    nonlocal _starting_daemon
                    try:
                        await anyio.to_thread.run_sync(
                            lambda: start_daemon_process(settings, port=port, host=host)
                        )
                    except Exception:
                        pass
                    finally:
                        _starting_daemon = False

                tg.start_soon(_runner)

            async def sse_worker() -> None:
                nonlocal active_sse_write, connected_event, reconnect_trigger
                while not tg.cancel_scope.cancel_called:
                    try:
                        async with sse_client(daemon_url) as (sse_read, sse_write):
                            active_sse_write = sse_write
                            connected_event.set()
                            async for message in sse_read:
                                if isinstance(message, Exception):
                                    continue
                                resp_id = _extract_id(message)
                                if resp_id is not None:
                                    if resp_id in timed_out_ids:
                                        timed_out_ids.discard(resp_id)
                                        continue
                                    evt = in_flight.pop(resp_id, None)
                                    if evt is not None:
                                        evt.set()
                                await stdio_write.send(message)
                    except (anyio.get_cancelled_exc_class(), anyio.ClosedResourceError):
                        break
                    except Exception as exc:
                        logger.debug("SSE connection error or disconnection: %s", exc)
                    finally:
                        active_sse_write = None
                        connected_event = anyio.Event()

                    # Wait before reconnecting or until reconnect_trigger is triggered
                    with anyio.move_on_after(1.0):
                        await reconnect_trigger.wait()
                    reconnect_trigger = anyio.Event()

            async def stdio_worker() -> None:
                nonlocal reconnect_trigger
                try:
                    async for message in stdio_read:
                        if isinstance(message, Exception):
                            continue

                        if _extract_method(message) == "server/discover":
                            await stdio_write.send(
                                _make_method_not_found_response(_extract_id(message))
                            )
                            continue

                        is_call, req_id = _is_tools_call(message)
                        if is_call:
                            # Pre-flight Fast Health Check (< 10ms)
                            is_alive = await anyio.to_thread.run_sync(
                                lambda: is_daemon_running(port=port, host=host)
                            )
                            if not is_alive:
                                trigger_start_daemon()
                                err_resp = _make_error_response(req_id, DEFAULT_RESTART_MESSAGE)
                                await stdio_write.send(err_resp)
                                continue

                        # Ensure SSE is connected
                        sse_w = active_sse_write
                        if sse_w is None:
                            reconnect_trigger.set()
                            with anyio.move_on_after(2.0):
                                await connected_event.wait()
                            sse_w = active_sse_write

                        if sse_w is None:
                            if req_id is not None:
                                err_resp = _make_error_response(
                                    req_id,
                                    "[Knowledge MCP] 데몬 서버(SSE)에 연결할 수 없습니다. 재기동을 시도합니다.",
                                )
                                await stdio_write.send(err_resp)
                                trigger_start_daemon()
                            continue

                        # Execution Safety Timeout tracking
                        if req_id is not None:
                            evt = anyio.Event()
                            in_flight[req_id] = evt

                            async def _watch_timeout(rid: Any, event_wait: anyio.Event) -> None:
                                with anyio.move_on_after(proxy_timeout):
                                    await event_wait.wait()
                                    return
                                if rid in in_flight:
                                    in_flight.pop(rid, None)
                                    timed_out_ids.add(rid)
                                    timeout_resp = _make_error_response(
                                        rid,
                                        f"[Knowledge MCP] 요청 처리 시간 초과 ({proxy_timeout}초). 데몬 서버가 응답하지 않습니다.",
                                    )
                                    await stdio_write.send(timeout_resp)
                                    # Check daemon health; if dead, trigger restart
                                    still_alive = await anyio.to_thread.run_sync(
                                        lambda: is_daemon_running(port=port, host=host)
                                    )
                                    if not still_alive:
                                        trigger_start_daemon()

                            tg.start_soon(_watch_timeout, req_id, evt)

                        try:
                            await sse_w.send(message)
                        except Exception:
                            if req_id is not None:
                                in_flight.pop(req_id, None)
                                err_resp = _make_error_response(
                                    req_id,
                                    "[Knowledge MCP] 데몬 서버로 메시지 전송에 실패했습니다.",
                                )
                                await stdio_write.send(err_resp)
                finally:
                    tg.cancel_scope.cancel()

            tg.start_soon(sse_worker)
            tg.start_soon(stdio_worker)
