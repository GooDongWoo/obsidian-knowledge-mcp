"""Application contracts through real HTTP and subprocess stdio framing."""

import json
import os
from pathlib import Path
import socket
import subprocess
import sys

import anyio
from fastmcp import Client
from fastmcp.client.transports import StdioTransport
import httpx2
import pytest
from mcp.shared.exceptions import MCPError

from knowledge_mcp.config import Settings
from knowledge_mcp.server import create_application
from knowledge_mcp.state import OperationLog
from mcp_test_helpers import serve_http
from test_server import FakeStore


@pytest.mark.anyio
@pytest.mark.parametrize("mode", ["auto", "legacy"])
async def test_slow_bootstrap_and_parser_keep_health_discovery_and_status_live(tmp_path, monkeypatch, mode):
    import asyncio
    import threading
    from knowledge_mcp import cli, daemon
    from knowledge_mcp.documents import parse_source
    from knowledge_mcp.indexer import KnowledgeIndexer
    from knowledge_mcp.qdrant_store import CollectionInventory
    from test_indexer import WordTokenizer

    settings = Settings(tmp_path / "vault", tmp_path / "runtime", Settings.DEFAULT_QDRANT_URL,
                        "slow-wire", Settings.DEFAULT_DENSE_MODEL, "test", project_root=tmp_path)
    settings.vault_root.mkdir()
    (settings.vault_root / "sample.md").write_text("# Example\nBody", encoding="utf-8")
    bootstrap_release, parse_release = threading.Event(), threading.Event()
    parsing = threading.Event()
    loop = asyncio.get_running_loop()

    class IndexStore:
        async def ensure_schema(self):
            assert asyncio.get_running_loop() is loop
        async def replace_generation(self, *args, **kwargs):
            assert asyncio.get_running_loop() is loop
            return ["point"]
        async def collection_inventory(self):
            assert asyncio.get_running_loop() is loop
            return CollectionInventory()
        async def generation_matches(self, *args):
            return True
        async def cleanup_orphans(self, generations, *, inventory):
            pass

    def parser(path, selected):
        parsing.set()
        assert parse_release.wait(8), "test did not release parser"
        return parse_source(path, selected)

    def dependencies(selected):
        log = OperationLog(selected.runtime_dir)
        return FakeStore(), log, {"test": KnowledgeIndexer(selected, IndexStore(), WordTokenizer(),
                                                            operation_log=log, parser=parser)}

    monkeypatch.setattr(cli, "ensure_qdrant", lambda _: bootstrap_release.wait(8))
    monkeypatch.setattr(cli, "_dependencies", dependencies)
    application = daemon.create_daemon_application(settings)
    try:
        async with serve_http(application.mcp) as url:
            async with httpx2.AsyncClient() as http, Client(url, mode=mode) as client:
                with anyio.fail_after(2):
                    assert (await http.get(url.replace("/mcp", "/health"))).json()["status"] == "starting"
                    assert len(await client.list_tools()) == 3
                    assert (await client.call_tool("knowledge-index-status", {})).data["state"] == "starting"
                    starting = await client.call_tool("qdrant-find", {"query": "query"}, raise_on_error=False)
                    assert starting.is_error and "starting" in starting.content[0].text
                bootstrap_release.set()
                with anyio.fail_after(3):
                    while not parsing.is_set():
                        await anyio.sleep(.01)
                with anyio.fail_after(2):
                    assert (await http.get(url.replace("/mcp", "/health"))).json()["status"] == "indexing"
                    assert len(await client.list_tools()) == 3
                    status = (await client.call_tool("knowledge-index-status", {})).data
                    assert status["state"] == "indexing"
                    assert status["progress"]["test"]["total"] == 1
                    result = await client.call_tool("qdrant-find", {"query": "query"}, raise_on_error=False)
                    assert result.is_error and "indexing" in result.content[0].text
                parse_release.set()
                with anyio.fail_after(3):
                    while (await client.call_tool("knowledge-index-status", {})).data["state"] != "ready":
                        await anyio.sleep(.01)
                assert not (await client.call_tool("qdrant-find", {"query": "query"})).is_error
                assert (await client.call_tool("knowledge-index-status", {})).data["last_completed"]
    finally:
        bootstrap_release.set()
        parse_release.set()


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def application(tmp_path):
    settings = Settings(tmp_path / "vault", tmp_path / "runtime", Settings.DEFAULT_QDRANT_URL,
                        "wire_test", Settings.DEFAULT_DENSE_MODEL, "wire-test", project_root=tmp_path)
    return create_application(settings, FakeStore(), OperationLog(settings.runtime_dir))


@pytest.mark.anyio
@pytest.mark.parametrize("mode", ["auto", "legacy"])
async def test_actual_daemon_entrypoint_exposes_three_tools_once(tmp_path, mode):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    script = tmp_path / "daemon.py"
    script.write_text(f'''
import sys
from pathlib import Path
sys.path.insert(0, {str(Path(__file__).parent)!r})
from test_server import FakeStore
from knowledge_mcp.config import Settings
from knowledge_mcp.state import OperationLog
from knowledge_mcp.indexer import IndexRunSummary
import asyncio
import knowledge_mcp.cli as cli
from knowledge_mcp.daemon import run_daemon
root = Path({str(tmp_path)!r})
settings = Settings(root / 'vault', root / 'runtime', Settings.DEFAULT_QDRANT_URL,
                    'daemon-wire', Settings.DEFAULT_DENSE_MODEL, 'test', project_root=root)
def record(name):
    with (root / name).open('a') as output:
        output.write('once\\n')
class Reranker:
    model_name = 'test'
    loaded = True
    def warmup(self): record('warmups')
class Indexer:
    active = 0
    async def sync(self, **kwargs):
        self.active += 1
        if self.active > 1: record('overlap')
        try:
            await asyncio.sleep(.05)
            return IndexRunSummary(unchanged=1)
        finally:
            self.active -= 1
def dependencies(settings):
    record('dependencies')
    store = FakeStore()
    store.stores = {{}}
    store.reranker = Reranker()
    return store, OperationLog(settings.runtime_dir), {{settings.dense_model: Indexer()}}
cli.ensure_qdrant = lambda settings: None
cli._dependencies = dependencies
run_daemon(settings, port={port})
''', encoding="utf-8")
    env = {**os.environ, "FASTMCP_CHECK_FOR_UPDATES": "off", "FASTMCP_TELEMETRY_MODE": "off",
           "FASTMCP_SHOW_SERVER_BANNER": "false", "PYTHONUTF8": "1"}
    popen_kwargs = {}
    if sys.platform == "win32":
        popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    else:
        popen_kwargs["start_new_session"] = True
    with (tmp_path / "daemon.log").open("w") as log:
        child = subprocess.Popen([sys.executable, str(script)], env=env,
                                 stdout=log, stderr=log,
                                 **popen_kwargs)
        try:
            async with httpx2.AsyncClient() as http:
                with anyio.fail_after(30):
                    while True:
                        assert child.poll() is None
                        try:
                            if (await http.get(f"http://127.0.0.1:{port}/health")).status_code == 200:
                                break
                        except httpx2.ConnectError:
                            pass
                        await anyio.sleep(0.1)
            async with Client(f"http://127.0.0.1:{port}/mcp", mode=mode) as client:
                assert (client.protocol_version == "2026-07-28") == (mode == "auto")
                assert {tool.name for tool in await client.list_tools()} == {
                    "qdrant-find", "knowledge-index-status", "knowledge-index-sync"}
                import asyncio
                result, second = await asyncio.gather(
                    client.call_tool("knowledge-index-sync", {}),
                    client.call_tool("knowledge-index-sync", {}))
                assert result.data[Settings.DEFAULT_DENSE_MODEL]["unchanged"] == 1
                assert second.data[Settings.DEFAULT_DENSE_MODEL]["unchanged"] == 1
                assert not (await client.call_tool("qdrant-find", {"query": "한글"})).is_error
                assert (await client.call_tool("knowledge-index-status", {})).data["collection"] == "daemon-wire"
            assert (tmp_path / "dependencies").read_text().splitlines() == ["once"]
            assert (tmp_path / "warmups").read_text().splitlines() == ["once"]
            assert not (tmp_path / "overlap").exists()
        finally:
            if child.poll() is None:
                if sys.platform == "win32":
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(child.pid)],
                                   capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
                else:
                    child.terminate()
            child.wait(timeout=10)


@pytest.mark.anyio
@pytest.mark.parametrize("mode,modern", [("auto", True), ("2026-07-28", True), ("legacy", False)])
@pytest.mark.parametrize("json_response", [False, True])
async def test_real_http_search_contract_and_protocol_metadata(application, mode, modern, json_response):
    requests = []
    async with serve_http(application.mcp, requests, json_response=json_response) as url:
        async with Client(url, mode=mode) as client:
            assert (client.protocol_version == "2026-07-28") is modern
            tools = {tool.name: tool for tool in await client.list_tools()}
            assert set(tools) == {"qdrant-find", "knowledge-index-status"}
            schema = tools["qdrant-find"].input_schema
            assert schema["required"] == ["query"]
            assert schema["properties"]["include_private"]["default"] is False
            assert schema["properties"]["limit"]["default"] == 8
            assert schema["properties"]["rerank"]["default"] is True
            result = await client.call_tool("qdrant-find", {"query": "프로젝트 알파"})
            data = result.data
            if isinstance(data, dict):
                data = data["result"]
            assert data[0]["document"] == "본문"
            assert json.loads(result.content[0].text) == data[0]
            assert data[0]["source_path"] == "note.md"
            assert (data[0]["start_line"], data[0]["end_line"], data[0]["score"]) == (3, 5, 0.9)
            status = await client.call_tool("knowledge-index-status", {})
            assert status.data["collection"] == "wire_test"

        rpc_posts = [r for r in requests if r["method"] == "POST" and r["body"]]
        messages = [json.loads(r["body"]) for r in rpc_posts]
        if modern:
            assert not any(m["method"] == "initialize" for m in messages)
            # Auto probes discovery; a pinned modern version adopts it directly.
            if mode == "auto":
                assert any(m["method"] == "server/discover" for m in messages)
            for record, message in zip(rpc_posts, messages):
                meta = message.get("params", {}).get("_meta", {})
                assert meta["io.modelcontextprotocol/protocolVersion"] == "2026-07-28"
                assert record["headers"][b"mcp-protocol-version"] == b"2026-07-28"
                assert record["headers"][b"mcp-method"].decode() == message["method"]
                assert b"mcp-session-id" not in record["headers"]
                if message["method"] == "tools/call":
                    assert record["headers"][b"mcp-name"].decode() == message["params"]["name"]
        else:
            assert any(m["method"] == "initialize" for m in messages)


@pytest.mark.anyio
async def test_empty_result_and_validation_error_keep_tool_contract(application):
    class EmptyStore:
        async def hybrid_search(self, request):
            return []

    application.store = EmptyStore()
    async with serve_http(application.mcp) as url:
        async with Client(url) as client:
            empty = await client.call_tool("qdrant-find", {"query": "없음"})
            assert not empty.is_error
            assert empty.data in ([], {"result": []})
            assert empty.content == []
            # Modern argument validation is a protocol error, not isError content.
            with pytest.raises(MCPError) as invalid:
                await client.call_tool("qdrant-find", {"query": "검색", "limit": 0})
            assert invalid.value.code == -32602


@pytest.mark.anyio
async def test_http_rejects_bad_origin_and_protocol_header_mismatch(application):
    records = []
    async with serve_http(application.mcp, records) as url:
        async with Client(url) as client:
            await client.list_tools()
        record = next(r for r in records if r["body"] and json.loads(r["body"])["method"] == "tools/list")
        headers = {k.decode(): v.decode() for k, v in record["headers"].items()
                   if k not in (b"content-length", b"host")}
        async with httpx2.AsyncClient() as http:
            forbidden = await http.post(url, content=record["body"], headers={**headers, "Origin": "https://evil.example"})
            assert forbidden.status_code == 403
            mismatch = await http.post(url, content=record["body"], headers={**headers, "MCP-Protocol-Version": "1900-01-01"})
            assert mismatch.status_code == 400
            assert "error" in mismatch.json()
            missing = await http.post(url, content=record["body"], headers={
                k: v for k, v in headers.items() if k != "mcp-protocol-version"})
            assert missing.status_code == 400
            unsupported_body = json.loads(record["body"])
            unsupported_body["params"]["_meta"]["io.modelcontextprotocol/protocolVersion"] = "1900-01-01"
            unsupported = await http.post(url, json=unsupported_body, headers={
                **headers, "mcp-protocol-version": "1900-01-01"})
            assert unsupported.status_code == 400
            assert "error" in unsupported.json()


@pytest.mark.anyio
@pytest.mark.parametrize("mode", ["auto", "legacy"])
async def test_actual_stdio_proxy_discovery_search_and_eof(application, mode):
    from starlette.responses import JSONResponse

    @application.mcp.custom_route("/health", methods=["GET"])
    async def health(request):
        return JSONResponse({"status": "ok"})

    async with serve_http(application.mcp) as url:
        environment = os.environ.copy()
        environment.update({"FASTMCP_CHECK_FOR_UPDATES": "off", "FASTMCP_TELEMETRY_MODE": "off",
                            "FASTMCP_SHOW_SERVER_BANNER": "false", "PYTHONIOENCODING": "utf-8"})
        transport = StdioTransport(
            command=sys.executable,
            args=["-c", "import anyio; from knowledge_mcp.proxy import run_stdio_proxy; "
                  "import sys; anyio.run(run_stdio_proxy, sys.argv[1])", url],
            env=environment,
        )
        with anyio.fail_after(20):
            async with Client(transport, mode=mode) as client:
                assert (client.protocol_version == "2026-07-28") is (mode == "auto")
                tools = await client.list_tools()
                assert {t.name for t in tools} == {"qdrant-find", "knowledge-index-status"}
                result = await client.call_tool("qdrant-find", {"query": "한글"})
                assert not result.is_error


@pytest.mark.anyio
@pytest.mark.parametrize("mode", ["auto", "legacy"])
async def test_stdio_proxy_session_failure_shares_restart_and_recovers(application, tmp_path, mode):
    import asyncio
    from starlette.responses import JSONResponse

    offline = tmp_path / "session-offline"
    restarts = tmp_path / "restarts"

    @application.mcp.custom_route("/health", methods=["GET"])
    async def health(request):
        return JSONResponse({"status": "starting"})

    class InterruptibleBackend:
        def http_app(self, **kwargs):
            upstream = application.mcp.http_app(**kwargs)

            async def app(scope, receive, send):
                if scope["type"] == "http" and scope["path"] == "/mcp" and offline.exists():
                    await JSONResponse({"error": "owned fixture unavailable"}, status_code=503)(scope, receive, send)
                else:
                    await upstream(scope, receive, send)

            return app

    script = tmp_path / "proxy.py"
    script.write_text(
        "import anyio, sys, time\nfrom pathlib import Path\n"
        "import knowledge_mcp.proxy as proxy\n"
        f"offline=Path({str(offline)!r})\nrestarts=Path({str(restarts)!r})\n"
        "def restart(*args, **kwargs):\n"
        "    with restarts.open('a') as output: output.write('restart\\n')\n"
        "    deadline=time.monotonic()+10\n"
        "    while offline.exists() and time.monotonic()<deadline: time.sleep(.01)\n"
        "proxy.start_daemon_process=restart\n"
        "anyio.run(proxy.run_stdio_proxy,sys.argv[1])\n", encoding="utf-8",
    )
    environment = {**os.environ, "FASTMCP_CHECK_FOR_UPDATES": "off", "FASTMCP_TELEMETRY_MODE": "off",
                   "FASTMCP_SHOW_SERVER_BANNER": "false", "PYTHONIOENCODING": "utf-8"}
    try:
        async with serve_http(InterruptibleBackend()) as url:
            transport = StdioTransport(command=sys.executable, args=[str(script), url], env=environment)
            with anyio.fail_after(25):
                async with Client(transport, mode=mode) as client:
                    assert len(await client.list_tools()) == 2
                    offline.touch()
                    try:
                        pending = [client.list_tools(), client.call_tool("qdrant-find", {"query": "offline"},
                                                                        raise_on_error=False)]
                        if mode == "auto":
                            pending.append(client.session.send_discover("2026-07-28"))
                        results = await asyncio.gather(*pending, return_exceptions=True)
                        assert isinstance(results[0], MCPError) and "재기동" in str(results[0])
                        assert results[1].is_error and "재기동" in results[1].content[0].text
                        if mode == "auto":
                            assert isinstance(results[2], MCPError) and "재기동" in str(results[2])
                        while not restarts.exists():
                            await anyio.sleep(.01)
                        assert restarts.read_text().splitlines() == ["restart"]
                    finally:
                        offline.unlink(missing_ok=True)
                    assert len(await client.list_tools()) == 2
                    if mode == "auto":
                        assert "capabilities" in await client.session.send_discover("2026-07-28")
                    assert not (await client.call_tool("qdrant-find", {"query": "한글"})).is_error
                    assert restarts.read_text().splitlines() == ["restart"]
    finally:
        offline.unlink(missing_ok=True)


@pytest.mark.anyio
@pytest.mark.parametrize("mode", ["auto", "legacy"])
async def test_standalone_cli_runs_real_stdio_protocol(tmp_path, mode):
    # Replace only external Qdrant/model work; execute the real CLI run path.
    script = tmp_path / "standalone.py"
    script.write_text(
        "from pathlib import Path\nfrom types import SimpleNamespace\n"
        "from knowledge_mcp import cli\nfrom knowledge_mcp.config import Settings\n"
        "from knowledge_mcp.state import OperationLog\n"
        "root=Path(__file__).parent\n"
        "settings=Settings(root/'vault',root/'runtime','http://127.0.0.1:6333','standalone',Settings.DEFAULT_DENSE_MODEL,'test',project_root=root)\n"
        "cli.Settings.from_env=lambda _: settings\ncli.ensure_qdrant=lambda _: None\n"
        "cli._dependencies=lambda _: (SimpleNamespace(),OperationLog(settings.runtime_dir),{})\n"
        "raise SystemExit(cli.main(['serve','--standalone']))\n", encoding="utf-8",
    )
    transport = StdioTransport(command=sys.executable, args=[str(script)],
                               env={**os.environ, "FASTMCP_CHECK_FOR_UPDATES": "off", "FASTMCP_TELEMETRY_MODE": "off"})
    with anyio.fail_after(20):
        async with Client(transport, mode=mode) as client:
            assert (client.protocol_version == "2026-07-28") is (mode == "auto")
            assert {t.name for t in await client.list_tools()} == {"qdrant-find", "knowledge-index-status"}
            status = await client.call_tool("knowledge-index-status", {})
            assert status.data["collection"] == "standalone"


@pytest.mark.anyio
@pytest.mark.parametrize("mode", ["auto", "legacy"])
@pytest.mark.parametrize("policy,applied,error", [
    ("default", True, None),
    ("false", False, None),
    ("init", False, "reranker_init_failed"),
    ("inference", False, "reranker_inference_failed"),
    ("invalid", False, "reranker_invalid_scores"),
])
async def test_reranking_outcomes_keep_text_and_structured_wire_contract(application, monkeypatch, mode, policy, applied, error):
    from knowledge_mcp.retrieval import MultiModelStore
    from test_retrieval import CandidateStore, result
    from types import SimpleNamespace

    class CrossEncoder:
        def __init__(self, name, *, device):
            if policy == "false":
                pytest.fail("explicit false constructed model")
            if policy == "init":
                raise RuntimeError("private init payload")
        def predict(self, pairs):
            if policy == "inference":
                raise RuntimeError("private inference payload")
            return [.2, .8] if policy == "default" else [float("nan")]

    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(CrossEncoder=CrossEncoder))
    application.store = MultiModelStore({application.settings.dense_model: CandidateStore([
        result("a", "one.md", .9), result("b", "two.md", .1)])})
    async with serve_http(application.mcp) as url:
        async with Client(url, mode=mode) as client:
            arguments = {"query": "q"}
            if policy == "false":
                arguments["rerank"] = False
            response = await client.call_tool("qdrant-find", arguments)
            assert not response.is_error
            data = response.data
            if isinstance(data, dict):
                data = data["result"]
            assert [item["point_id"] for item in data] == (["b", "a"] if applied else ["a", "b"])
            assert [item["score"] for item in data] == ([.8, .2] if applied else [.9, .1])
            assert all(item["rerank_requested"] is (policy != "false") for item in data)
            assert all(item["rerank_applied"] is applied and item["rerank_error"] == error for item in data)
            assert [json.loads(block.text) for block in response.content] == data
