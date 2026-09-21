import inspect
from contextlib import asynccontextmanager
from unittest.mock import patch
import anyio
import pytest
from mcp.shared.session import SessionMessage
from mcp.types import JSONRPCError, JSONRPCMessage, JSONRPCRequest


def test_proxy_does_not_import_heavy_libraries():
    """Verify that importing proxy does not load torch, onnxruntime, or fastembed."""
    import subprocess
    import sys
    import knowledge_mcp.proxy

    source = inspect.getsource(knowledge_mcp.proxy)
    assert "torch" not in source
    assert "onnxruntime" not in source
    assert "fastembed" not in source
    assert "sentence_transformers" not in source

    # Verify in a clean interpreter that no heavy ML libraries are imported
    cmd = [
        sys.executable,
        "-c",
        "import sys, knowledge_mcp.proxy; "
        "assert 'torch' not in sys.modules; "
        "assert 'onnxruntime' not in sys.modules; "
        "assert 'sentence_transformers' not in sys.modules; "
        "assert 'fastembed' not in sys.modules",
    ]
    subprocess.run(cmd, check=True)


@pytest.mark.anyio
async def test_run_stdio_proxy_bidirectional():
    from knowledge_mcp.proxy import run_stdio_proxy

    stdio_in_send, stdio_in_recv = anyio.create_memory_object_stream(10)
    stdio_out_send, stdio_out_recv = anyio.create_memory_object_stream(10)

    sse_in_send, sse_in_recv = anyio.create_memory_object_stream(10)
    sse_out_send, sse_out_recv = anyio.create_memory_object_stream(10)

    @asynccontextmanager
    async def fake_stdio_server():
        yield stdio_in_recv, stdio_out_send

    @asynccontextmanager
    async def fake_sse_client(url):
        yield sse_in_recv, sse_out_send

    with patch("knowledge_mcp.proxy.stdio_server", fake_stdio_server), \
         patch("knowledge_mcp.proxy.sse_client", fake_sse_client):

        async with anyio.create_task_group() as tg:
            tg.start_soon(run_stdio_proxy, "http://127.0.0.1:8765/sse")

            # 1. Send message from stdio -> should arrive at sse_out
            msg_from_stdio = "msg_from_client"
            await stdio_in_send.send(msg_from_stdio)
            received_by_sse = await sse_out_recv.receive()
            assert received_by_sse == msg_from_stdio

            # 2. Send message from sse -> should arrive at stdio_out
            msg_from_sse = "msg_from_server"
            await sse_in_send.send(msg_from_sse)
            received_by_stdio = await stdio_out_recv.receive()
            assert received_by_stdio == msg_from_sse

            # 3. Close stdio stream to trigger shutdown
            await stdio_in_send.aclose()


@pytest.mark.anyio
async def test_server_discover_returns_method_not_found_without_forwarding_to_legacy_daemon():
    """Codex discovery must fall back immediately instead of hanging on the legacy SSE daemon."""
    from knowledge_mcp.proxy import run_stdio_proxy

    stdio_in_send, stdio_in_recv = anyio.create_memory_object_stream(10)
    stdio_out_send, stdio_out_recv = anyio.create_memory_object_stream(10)
    sse_in_send, sse_in_recv = anyio.create_memory_object_stream(10)
    sse_out_send, sse_out_recv = anyio.create_memory_object_stream(10)

    @asynccontextmanager
    async def fake_stdio_server():
        yield stdio_in_recv, stdio_out_send

    @asynccontextmanager
    async def fake_sse_client(url):
        yield sse_in_recv, sse_out_send

    request = SessionMessage(
        JSONRPCMessage(
            root=JSONRPCRequest(
                jsonrpc="2.0",
                id=7,
                method="server/discover",
                params={"protocolVersion": "2026-07-28"},
            )
        )
    )

    with patch("knowledge_mcp.proxy.stdio_server", fake_stdio_server), \
         patch("knowledge_mcp.proxy.sse_client", fake_sse_client):
        async with anyio.create_task_group() as tg:
            tg.start_soon(run_stdio_proxy, "http://127.0.0.1:8765/sse", None, 0.01)
            await stdio_in_send.send(request)

            await anyio.sleep(0.05)
            assert sse_out_recv.statistics().current_buffer_used == 0
            response = await stdio_out_recv.receive()
            inner = response.message.root
            assert isinstance(inner, JSONRPCError)
            assert inner.id == 7
            assert inner.error.code == -32601

            await stdio_in_send.aclose()


@pytest.mark.anyio
async def test_preflight_check_fails_fast_and_triggers_restart():
    """Verify that when daemon is dead, pre-flight check triggers restart and returns error response immediately (< 0.5s)."""
    from knowledge_mcp.proxy import run_stdio_proxy

    stdio_in_send, stdio_in_recv = anyio.create_memory_object_stream(10)
    stdio_out_send, stdio_out_recv = anyio.create_memory_object_stream(10)

    @asynccontextmanager
    async def fake_stdio_server():
        yield stdio_in_recv, stdio_out_send

    restart_calls = []

    def fake_start_daemon(*args, **kwargs):
        restart_calls.append((args, kwargs))
        return 1234

    @asynccontextmanager
    async def fake_sse_client(url):
        raise ConnectionError("Daemon offline")
        yield

    with patch("knowledge_mcp.proxy.stdio_server", fake_stdio_server), \
         patch("knowledge_mcp.proxy.sse_client", fake_sse_client), \
         patch("knowledge_mcp.proxy.is_daemon_running", return_value=False), \
         patch("knowledge_mcp.proxy.start_daemon_process", fake_start_daemon):

        async with anyio.create_task_group() as tg:
            tg.start_soon(run_stdio_proxy, "http://127.0.0.1:8765/sse")

            tools_call_msg = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "qdrant-find", "arguments": {"query": "test"}},
            }

            t0 = anyio.current_time()
            await stdio_in_send.send(tools_call_msg)

            response = await stdio_out_recv.receive()
            elapsed = anyio.current_time() - t0

            # Timing check: must return immediately (< 0.5s)
            assert elapsed < 0.5

            # Error response check
            inner = response.message.root if hasattr(response, "message") else response
            assert inner.id == 1
            assert inner.result.get("isError") is True
            content = inner.result.get("content", [])
            assert any(
                "데몬 서버가 꺼져 있어 즉시 재기동을 시작했습니다" in c.get("text", "")
                for c in content
            )

            # Background thread started the daemon
            await anyio.sleep(0.05)
            assert len(restart_calls) == 1

            await stdio_in_send.aclose()


@pytest.mark.anyio
async def test_execution_safety_timeout_when_daemon_hangs():
    """Verify that when daemon hangs during execution, timeout returns error response."""
    from knowledge_mcp.proxy import run_stdio_proxy

    stdio_in_send, stdio_in_recv = anyio.create_memory_object_stream(10)
    stdio_out_send, stdio_out_recv = anyio.create_memory_object_stream(10)

    sse_in_send, sse_in_recv = anyio.create_memory_object_stream(10)
    sse_out_send, sse_out_recv = anyio.create_memory_object_stream(10)

    @asynccontextmanager
    async def fake_stdio_server():
        yield stdio_in_recv, stdio_out_send

    @asynccontextmanager
    async def fake_sse_client(url):
        yield sse_in_recv, sse_out_send

    restart_calls = []

    def fake_start_daemon(*args, **kwargs):
        restart_calls.append((args, kwargs))
        return 1234

    # First is_daemon_running is True (pre-flight passes), then False (daemon died/hung during timeout check)
    with patch("knowledge_mcp.proxy.stdio_server", fake_stdio_server), \
         patch("knowledge_mcp.proxy.sse_client", fake_sse_client), \
         patch("knowledge_mcp.proxy.is_daemon_running", side_effect=[True, False]), \
         patch("knowledge_mcp.proxy.start_daemon_process", fake_start_daemon):

        async with anyio.create_task_group() as tg:
            # Short timeout of 0.1s for fast and reliable testing
            tg.start_soon(run_stdio_proxy, "http://127.0.0.1:8765/sse", None, 0.1)

            tools_call_msg = {
                "jsonrpc": "2.0",
                "id": 42,
                "method": "tools/call",
                "params": {"name": "qdrant-find", "arguments": {"query": "hang"}},
            }

            await stdio_in_send.send(tools_call_msg)

            # Daemon receives request on SSE
            received_by_sse = await sse_out_recv.receive()
            assert received_by_sse == tools_call_msg

            # Daemon hangs and does not respond -> timeout triggers
            response = await stdio_out_recv.receive()

            inner = response.message.root if hasattr(response, "message") else response
            assert inner.id == 42
            assert inner.result.get("isError") is True
            content = inner.result.get("content", [])
            assert any(
                "시간 초과" in c.get("text", "")
                for c in content
            )

            # Check that daemon restart was triggered after timeout found it dead
            await anyio.sleep(0.05)
            assert len(restart_calls) == 1

            await stdio_in_send.aclose()


@pytest.mark.anyio
async def test_execution_normal_response_cancels_timeout():
    """Verify that a normal response from the daemon cancels the timeout without error."""
    from knowledge_mcp.proxy import run_stdio_proxy

    stdio_in_send, stdio_in_recv = anyio.create_memory_object_stream(10)
    stdio_out_send, stdio_out_recv = anyio.create_memory_object_stream(10)

    sse_in_send, sse_in_recv = anyio.create_memory_object_stream(10)
    sse_out_send, sse_out_recv = anyio.create_memory_object_stream(10)

    @asynccontextmanager
    async def fake_stdio_server():
        yield stdio_in_recv, stdio_out_send

    @asynccontextmanager
    async def fake_sse_client(url):
        yield sse_in_recv, sse_out_send

    with patch("knowledge_mcp.proxy.stdio_server", fake_stdio_server), \
         patch("knowledge_mcp.proxy.sse_client", fake_sse_client), \
         patch("knowledge_mcp.proxy.is_daemon_running", return_value=True):

        async with anyio.create_task_group() as tg:
            tg.start_soon(run_stdio_proxy, "http://127.0.0.1:8765/sse", None, 0.5)

            req = {
                "jsonrpc": "2.0",
                "id": 100,
                "method": "tools/call",
                "params": {"name": "qdrant-find", "arguments": {"query": "hello"}},
            }
            await stdio_in_send.send(req)

            received = await sse_out_recv.receive()
            assert received == req

            normal_resp = {
                "jsonrpc": "2.0",
                "id": 100,
                "result": {"content": [{"type": "text", "text": "success"}], "isError": False},
            }
            await sse_in_send.send(normal_resp)

            client_received = await stdio_out_recv.receive()
            assert client_received == normal_resp

            # Wait past timeout to ensure no spurious timeout message arrives
            await anyio.sleep(0.6)
            assert stdio_out_recv.statistics().current_buffer_used == 0

            await stdio_in_send.aclose()
