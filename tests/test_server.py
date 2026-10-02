import pytest
import sqlite3
from fastembed.common.types import Device

from knowledge_mcp.config import Settings
from knowledge_mcp.search import SearchResult
from knowledge_mcp.state import OperationLog


class FakeStore:
    embedding_provider = type("Provider", (), {"device": Device.CPU})()

    async def hybrid_search(self, request):
        return [SearchResult(
            point_id="p1", document="본문", source_path="note.md",
            document_type="other", file_type="md", security_level="public",
            created_at="2025-01-01T00:00:00+09:00",
            modified_at="2025-01-02T00:00:00+09:00", score=0.9,
            start_line=3, end_line=5,
        )]


@pytest.fixture
def settings(tmp_path):
    return Settings(tmp_path, tmp_path / ".knowledge", Settings.DEFAULT_QDRANT_URL,
                    "test", Settings.DEFAULT_DENSE_MODEL, "codex")


@pytest.mark.anyio
async def test_find_tool_returns_source_location_and_logs_only_metrics(settings):
    from knowledge_mcp.server import create_application

    application = create_application(settings, FakeStore(), OperationLog(settings.runtime_dir))
    result = await application.find(query="알파")
    assert result[0]["source_path"] == "note.md"
    assert result[0]["start_line"] == 3
    db_text = application.operation_log.database_text_for_test()
    assert "알파" not in db_text
    assert "codex" not in db_text
    assert "note.md" not in db_text
    with sqlite3.connect(settings.runtime_dir / "state.sqlite3") as connection:
        assert connection.execute("SELECT result_count, rerank_requested, rerank_applied FROM queries").fetchall() == [(1, 1, 0)]


@pytest.mark.anyio
async def test_find_failure_records_stable_code_without_exception_payload(settings):
    from knowledge_mcp.server import create_application

    class FailingStore:
        async def hybrid_search(self, request):
            raise RuntimeError("sensitive-exception-payload")

    application = create_application(settings, FailingStore(), OperationLog(settings.runtime_dir))
    with pytest.raises(RuntimeError, match="sensitive-exception-payload"):
        await application.find(query="sensitive-query", rerank=True)
    with sqlite3.connect(settings.runtime_dir / "state.sqlite3") as connection:
        assert connection.execute("SELECT result_count, rerank_requested, rerank_applied, error_code FROM queries").fetchall() == [(0, 1, None, "search_failed")]
    assert "sensitive" not in application.operation_log.database_text_for_test()


@pytest.mark.anyio
async def test_find_tool_exposes_only_two_read_only_tools(settings):
    from knowledge_mcp.server import create_application

    application = create_application(settings, FakeStore(), OperationLog(settings.runtime_dir))
    tools = {tool.name: tool for tool in await application.mcp.list_tools()}
    assert set(tools) == {"qdrant-find", "knowledge-index-status"}


@pytest.mark.anyio
async def test_status_tool_is_read_only(settings):
    from knowledge_mcp.server import create_application

    log = OperationLog(settings.runtime_dir)
    application = create_application(settings, FakeStore(), log)
    result = await application.status()
    assert isinstance(result, dict)
    assert result["embedding_device"] == "cpu"
    assert "qdrant-store" not in {tool.name for tool in await application.mcp.list_tools()}


@pytest.mark.anyio
async def test_find_exposes_model_and_rerank_options(settings):
    from knowledge_mcp.server import create_application

    class RecordingStore(FakeStore):
        async def hybrid_search(self, request):
            self.request = request
            return await super().hybrid_search(request)

    store = RecordingStore()
    application = create_application(settings, store, OperationLog(settings.runtime_dir))
    await application.find(query="검색", embedding_model="dragonkue/BGE-m3-ko", rerank=True)
    assert store.request.embedding_model == "dragonkue/BGE-m3-ko"
    assert store.request.rerank is True
    tool = next(tool for tool in await application.mcp.list_tools() if tool.name == "qdrant-find")
    assert "embedding_model" in tool.parameters["properties"]
    assert "dragonkue/BGE-m3-ko" in str(tool.parameters["properties"]["embedding_model"])
    assert "rerank" in tool.parameters["properties"]


@pytest.mark.anyio
async def test_status_reports_each_model_collection(settings):
    from knowledge_mcp.server import create_application
    from knowledge_mcp.retrieval import MultiModelStore

    class Client:
        async def count(self, collection, *, exact):
            return type("Count", (), {"count": 11})()

    class Store:
        client = Client()
        embedding_provider = FakeStore.embedding_provider

        def __init__(self, collection):
            self.settings = type("Settings", (), {"collection_name": collection})()

    store = MultiModelStore({"dragonkue/BGE-m3-ko": Store("bge")})
    log = OperationLog(settings.runtime_dir)
    log.record_index(collection_name="bge", generation=None, added=1, changed=0, deleted=0)
    application = create_application(settings, store, log)
    result = await application.status()
    assert result["models"]["dragonkue/BGE-m3-ko"]["point_count"] == 11
    assert result["reranker"] == {"model": "dragonkue/bge-reranker-v2-m3-ko", "loaded": False, "device": None, "last_fallback": None}
    assert result["collection"] == "bge"
    assert result["point_count"] == 11
    assert result["status"] == "completed"
    assert result["models"]["dragonkue/BGE-m3-ko"]["status"] == "completed"


@pytest.mark.anyio
async def test_default_model_follows_settings(settings):
    from knowledge_mcp.server import create_application

    class RecordingStore(FakeStore):
        async def hybrid_search(self, request):
            self.request = request
            return await super().hybrid_search(request)

    store = RecordingStore()
    application = create_application(settings, store, OperationLog(settings.runtime_dir))
    await application.find(query="검색")
    assert store.request.embedding_model == "dragonkue/BGE-m3-ko"


@pytest.mark.anyio
async def test_connection_failure_after_index_failure_is_not_an_indexing_error(settings):
    from knowledge_mcp.server import create_application

    class DisconnectedStore:
        async def hybrid_search(self, request):
            raise ConnectionError("Qdrant is unavailable")

    application = create_application(settings, DisconnectedStore(), OperationLog(settings.runtime_dir))
    application.runtime_status = {"state": "error", "last_error": "schema_check_failed"}
    with pytest.raises(ConnectionError, match="Qdrant is unavailable"):
        await application.find(query="query")


@pytest.mark.anyio
async def test_cached_status_refreshes_current_reranker_without_index_reads(settings, monkeypatch):
    from knowledge_mcp.server import create_application
    from knowledge_mcp.retrieval import MultiModelStore
    from test_retrieval import CandidateStore, result
    from types import SimpleNamespace
    import sys

    class CrossEncoder:
        def __init__(self, name, *, device):
            pass
        def predict(self, pairs):
            raise RuntimeError("private inference payload")

    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(CrossEncoder=CrossEncoder))
    store = MultiModelStore({settings.dense_model: CandidateStore([result("a", "one", .9)])})
    application = create_application(settings, store, OperationLog(settings.runtime_dir))
    application.runtime_status = {"state": "ready"}
    application.index_snapshot = {"reranker": {"loaded": False}, "point_count": 17}
    await application.find(query="q")
    status = await application.status()
    assert status["point_count"] == 17
    assert status["reranker"] == {"model": store.reranker.model_name, "loaded": True, "device": "cpu", "last_fallback": "reranker_inference_failed"}


@pytest.mark.anyio
async def test_concurrent_metrics_use_result_outcomes_and_keep_payloads_private(settings):
    import asyncio
    from knowledge_mcp.server import create_application
    from test_retrieval import result

    class Store:
        async def hybrid_search(self, request):
            await asyncio.sleep(0)
            if request.query == "private success":
                return [result("private_id", "private.md", .5).model_copy(update={"rerank_requested": True, "rerank_applied": True})]
            return [result("private_id", "private.md", .9).model_copy(update={"rerank_requested": True, "rerank_applied": False, "rerank_error": "reranker_inference_failed"})]

    application = create_application(settings, Store(), OperationLog(settings.runtime_dir))
    await asyncio.gather(application.find(query="private success"), application.find(query="private fallback"))
    with sqlite3.connect(settings.runtime_dir / "state.sqlite3") as connection:
        rows = connection.execute("SELECT result_count, rerank_requested, rerank_applied, error_code FROM queries").fetchall()
    assert sorted(rows, key=lambda row: row[2]) == [(1, 1, 0, "reranker_inference_failed"), (1, 1, 1, None)]
    assert "private" not in application.operation_log.database_text_for_test()


@pytest.mark.anyio
async def test_empty_search_logs_requested_without_inventing_application(settings):
    from knowledge_mcp.server import create_application

    class Store:
        async def hybrid_search(self, request):
            return []

    application = create_application(settings, Store(), OperationLog(settings.runtime_dir))
    assert await application.find(query="q") == []
    with sqlite3.connect(settings.runtime_dir / "state.sqlite3") as connection:
        assert connection.execute("SELECT rerank_requested, rerank_applied, error_code FROM queries").fetchall() == [(1, 0, None)]
