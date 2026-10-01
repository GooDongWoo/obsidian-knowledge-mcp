import subprocess
import asyncio
import sys
import threading

import anyio
import pytest
from fastmcp import Client, Context, FastMCP
from mcp.shared.exceptions import MCPError

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

    monkeypatch.setattr("knowledge_mcp.proxy.is_daemon_running", lambda **_: True)
    monkeypatch.setattr("knowledge_mcp.proxy.start_daemon_process", restart)
    async with serve_http(backend) as url:
        proxy = create_stdio_proxy(url)
        async with Client(proxy) as client:
            await client.list_tools()
            monkeypatch.setattr("knowledge_mcp.proxy.is_daemon_running", lambda **_: False)
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
async def test_unavailable_provider_is_an_error_instead_of_empty_tools(monkeypatch):
    from knowledge_mcp.proxy import create_stdio_proxy

    monkeypatch.setattr("knowledge_mcp.proxy.start_daemon_process", lambda *_, **__: None)
    async with Client(create_stdio_proxy("http://127.0.0.1:1/mcp", timeout=0.1), mode="2026-07-28") as client:
        with pytest.raises(Exception) as failure:
            with anyio.fail_after(5):
                await client.list_tools()
        assert not isinstance(failure.value, TimeoutError)


@pytest.mark.anyio
async def test_offline_list_discovery_and_call_share_restart_and_recover(monkeypatch):
    from knowledge_mcp.proxy import create_stdio_proxy

    backend = FastMCP("recovery")

    @backend.tool
    async def echo(text: str) -> str:
        return text

    alive = True
    started, release = threading.Event(), threading.Event()
    restarts = []

    def restart(*args, **kwargs):
        restarts.append(kwargs)
        started.set()
        assert release.wait(5)

    monkeypatch.setattr("knowledge_mcp.proxy.is_daemon_running", lambda **_: alive)
    monkeypatch.setattr("knowledge_mcp.proxy.start_daemon_process", restart)
    try:
        async with serve_http(backend) as url:
            proxy = create_stdio_proxy(url)
            async with Client(proxy, mode="legacy") as legacy, Client(proxy) as modern:
                assert [tool.name for tool in await legacy.list_tools()] == ["echo"]
                assert [tool.name for tool in await modern.list_tools()] == ["echo"]
                alive = False
                with anyio.fail_after(3):
                    results = await asyncio.gather(
                        legacy.list_tools(), modern.session.send_discover("2026-07-28"),
                        modern.call_tool("echo", {"text": "offline"}, raise_on_error=False),
                        return_exceptions=True,
                    )
                    while not started.is_set():
                        await anyio.sleep(.01)
                for error in results[:2]:
                    assert isinstance(error, MCPError)
                    assert "재기동" in str(error)
                assert results[2].is_error and "재기동" in results[2].content[0].text
                assert len(restarts) == 1
                alive = True
                release.set()
                assert [tool.name for tool in await legacy.list_tools()] == ["echo"]
                assert [tool.name for tool in await modern.list_tools()] == ["echo"]
                assert "capabilities" in await modern.session.send_discover("2026-07-28")
                assert (await legacy.call_tool("echo", {"text": "recovered"})).data == "recovered"
                assert (await modern.call_tool("echo", {"text": "recovered"})).data == "recovered"
    finally:
        release.set()


@pytest.mark.anyio
@pytest.mark.parametrize("method", ["list", "discover", "call"])
async def test_backend_disappearing_after_health_probe_uses_restart_message(monkeypatch, method):
    from knowledge_mcp.proxy import create_stdio_proxy

    probes, restarts = [], []

    def health(**kwargs):
        probes.append(kwargs)
        return len(probes) == 1

    monkeypatch.setattr("knowledge_mcp.proxy.is_daemon_running", health)
    monkeypatch.setattr("knowledge_mcp.proxy.start_daemon_process", lambda *_, **kw: restarts.append(kw))
    proxy = create_stdio_proxy("http://127.0.0.1:1/mcp")
    async with Client(proxy, mode="2026-07-28") as client:
        with anyio.fail_after(5):
            if method == "call":
                result = await client.call_tool("echo", {}, raise_on_error=False)
                assert result.is_error and "재기동" in result.content[0].text
            else:
                with pytest.raises(MCPError, match="재기동"):
                    if method == "list":
                        await client.list_tools()
                    else:
                        await client.session.send_discover("2026-07-28")
            while not restarts:
                await anyio.sleep(.01)
        assert len(restarts) == 1


@pytest.mark.anyio
@pytest.mark.parametrize("method", ["list", "discover", "call"])
async def test_healthy_listener_with_failed_backend_session_returns_restart_error(monkeypatch, method):
    from knowledge_mcp.proxy import create_stdio_proxy
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    class UnavailableSession:
        def http_app(self, **kwargs):
            async def health(request):
                return JSONResponse({"status": "starting"})

            async def unavailable(request):
                return JSONResponse({"error": "fixture session unavailable"}, status_code=503)

            return Starlette(routes=[Route("/health", health), Route("/mcp", unavailable, methods=["POST"])])

    restarts = []
    monkeypatch.setattr("knowledge_mcp.proxy.start_daemon_process", lambda *_, **kw: restarts.append(kw))
    async with serve_http(UnavailableSession()) as url:
        async with Client(create_stdio_proxy(url), mode="2026-07-28") as client:
            with anyio.fail_after(5):
                if method == "call":
                    result = await client.call_tool("echo", {}, raise_on_error=False)
                    assert result.is_error and "재기동" in result.content[0].text
                else:
                    with pytest.raises(MCPError, match="재기동"):
                        if method == "list":
                            await client.list_tools()
                        else:
                            await client.session.send_discover("2026-07-28")
                while not restarts:
                    await anyio.sleep(.01)
            assert len(restarts) == 1


@pytest.mark.anyio
@pytest.mark.parametrize("mode", ["auto", "legacy"])
async def test_healthy_backend_validation_and_tool_errors_do_not_restart(monkeypatch, mode):
    from knowledge_mcp.proxy import create_stdio_proxy
    from fastmcp.server.middleware import Middleware
    from starlette.responses import JSONResponse

    class BackendValidation(Middleware):
        async def on_call_tool(self, context, call_next):
            if context.message.arguments["number"] == 0:
                raise MCPError(code=-32602, message="fixture rejects zero")
            return await call_next(context)

    backend = FastMCP("errors", middleware=[BackendValidation()])

    @backend.custom_route("/health", methods=["GET"])
    async def health(request):
        return JSONResponse({"status": "indexing"})

    @backend.tool
    async def echo(number: int) -> str:
        raise ValueError("tool fixture failed")

    restarts = []
    monkeypatch.setattr("knowledge_mcp.proxy.start_daemon_process", lambda *_, **kw: restarts.append(kw))
    async with serve_http(backend) as url:
        async with Client(create_stdio_proxy(url), mode=mode) as client:
            await client.list_tools()
            invalid = await client.call_tool("echo", {"number": "bad"}, raise_on_error=False)
            assert invalid.is_error and "재기동" not in invalid.content[0].text
            remote_invalid = await client.call_tool("echo", {"number": 0}, raise_on_error=False)
            assert remote_invalid.is_error and "재기동" not in remote_invalid.content[0].text
            failure = await client.call_tool("echo", {"number": 1}, raise_on_error=False)
            assert failure.is_error and "재기동" not in failure.content[0].text
            assert restarts == []


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
