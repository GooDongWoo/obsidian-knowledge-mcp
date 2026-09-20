import inspect
from contextlib import asynccontextmanager
from unittest.mock import patch
import anyio
import pytest


def test_proxy_does_not_import_heavy_libraries():
    """Verify that importing proxy does not load torch, onnxruntime, or fastembed."""
    import knowledge_mcp.proxy

    source = inspect.getsource(knowledge_mcp.proxy)
    assert "torch" not in source
    assert "onnxruntime" not in source
    assert "fastembed" not in source
    assert "sentence_transformers" not in source


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
