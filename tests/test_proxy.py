import subprocess
import asyncio
import sys

import anyio
import pytest
from fastmcp import Client, Context, FastMCP

from mcp_test_helpers import serve_http


@pytest.fixture
def anyio_backend():
    return "asyncio"


def test_proxy_does_not_import_heavy_libraries():
    subprocess.run(
        [sys.executable, "-c", "import sys, knowledge_mcp.proxy; "
         "assert not {'torch', 'onnxruntime', 'fastembed', 'sentence_transformers'} & sys.modules.keys()"],
        check=True, timeout=30,
    )


@pytest.mark.anyio
@pytest.mark.parametrize("mode,modern", [("auto", True), ("2026-07-28", True), ("legacy", False)])
async def test_proxy_mirrors_client_protocol_to_http_backend(mode, modern, monkeypatch):
    from knowledge_mcp.proxy import create_stdio_proxy

    backend = FastMCP("backend")

    @backend.tool
    async def echo(text: str, ctx: Context) -> dict:
        return {"text": text, "protocol": ctx.request_context.protocol_version}

    monkeypatch.setattr("knowledge_mcp.proxy.is_daemon_running", lambda **_: True)
    async with serve_http(backend) as url:
        async with Client(create_stdio_proxy(url), mode=mode) as client:
            assert (client.protocol_version == "2026-07-28") is modern
            result = await client.call_tool("echo", {"text": "한글"})
            assert result.data["text"] == "한글"
            assert result.data["protocol"] == client.protocol_version


@pytest.mark.anyio
async def test_offline_daemon_returns_tool_error_and_starts_only_once(monkeypatch):
    from knowledge_mcp.proxy import create_stdio_proxy

    backend = FastMCP("backend")

    @backend.tool
    async def echo() -> str:
        return "success"

    calls = []

    def restart(*args, **kwargs):
        calls.append(kwargs)
        import time
        time.sleep(0.2)

    async with serve_http(backend) as url:
        proxy = create_stdio_proxy(url)
        async with Client(proxy) as client:
            await client.list_tools()
            monkeypatch.setattr("knowledge_mcp.proxy.is_daemon_running", lambda **_: False)
            monkeypatch.setattr("knowledge_mcp.proxy.start_daemon_process", restart)
            with anyio.fail_after(2):
                first = await client.call_tool("echo", {}, raise_on_error=False)
                second = await client.call_tool("echo", {}, raise_on_error=False)
                while not calls:
                    await anyio.sleep(0.01)
            assert first.is_error and second.is_error
            assert "재기동" in first.content[0].text
            assert len(calls) == 1


@pytest.mark.anyio
async def test_timeout_does_not_replay_call_or_poison_next_request(monkeypatch):
    from knowledge_mcp.proxy import create_stdio_proxy

    backend = FastMCP("backend")
    calls = []

    @backend.tool
    async def slow(delay: float) -> str:
        calls.append(delay)
        await anyio.sleep(delay)
        return "done"

    monkeypatch.setattr("knowledge_mcp.proxy.is_daemon_running", lambda **_: True)
    async with serve_http(backend) as url:
        async with Client(create_stdio_proxy(url, timeout=2.0)) as client:
            expired = await client.call_tool("slow", {"delay": 4.0}, raise_on_error=False)
            assert expired.is_error
            assert "시간 초과" in expired.content[0].text
            success = await client.call_tool("slow", {"delay": 0.0})
            assert success.data == "done"
            await anyio.sleep(4.05)
            assert calls == [4.0, 0.0]


@pytest.mark.anyio
async def test_unavailable_provider_is_an_error_instead_of_empty_tools():
    from knowledge_mcp.proxy import create_stdio_proxy

    async with Client(create_stdio_proxy("http://127.0.0.1:1/mcp", timeout=0.1)) as client:
        with pytest.raises(Exception) as failure:
            with anyio.fail_after(5):
                await client.list_tools()
        assert not isinstance(failure.value, TimeoutError)


@pytest.mark.anyio
async def test_cancelling_one_call_keeps_concurrent_call_and_connection_usable(monkeypatch):
    from knowledge_mcp.proxy import create_stdio_proxy

    backend = FastMCP("cancellation")
    started = asyncio.Event()
    calls = []

    @backend.tool
    async def echo(delay: float) -> str:
        calls.append(delay)
        if delay:
            started.set()
            await anyio.sleep(delay)
        return "done"

    monkeypatch.setattr("knowledge_mcp.proxy.is_daemon_running", lambda **_: True)
    async with serve_http(backend) as url:
        async with Client(create_stdio_proxy(url)) as client:
            await client.list_tools()
            slow = asyncio.create_task(client.call_tool("echo", {"delay": 5.0}))
            with anyio.fail_after(3):
                await started.wait()
                assert (await client.call_tool("echo", {"delay": 0.0})).data == "done"
            slow.cancel()
            with pytest.raises(asyncio.CancelledError):
                await slow
            assert (await client.call_tool("echo", {"delay": 0.0})).data == "done"
            assert calls == [5.0, 0.0, 0.0]


@pytest.mark.parametrize("value", ["invalid", "nan", "inf", "0", "-1"])
def test_invalid_environment_timeout_uses_bounded_default(value, monkeypatch):
    from knowledge_mcp.proxy import create_stdio_proxy

    monkeypatch.setenv("KNOWLEDGE_PROXY_TIMEOUT", value)
    proxy = create_stdio_proxy()
    guard = proxy.middleware[0]
    assert guard.timeout == 15.0
