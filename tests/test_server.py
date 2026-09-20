import pytest
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
async def test_find_tool_returns_source_location_and_logs_query(settings):
    from knowledge_mcp.server import create_application

    application = create_application(settings, FakeStore(), OperationLog(settings.runtime_dir))
    result = await application.find(query="알파")
    assert result[0]["source_path"] == "note.md"
    assert result[0]["start_line"] == 3
    db_text = application.operation_log.database_text_for_test()
    assert db_text.find("알파") >= 0
    assert db_text.find("codex") >= 0


@pytest.mark.anyio
async def test_find_tool_exposes_only_two_read_only_tools(settings):
    from knowledge_mcp.server import create_application

    application = create_application(settings, FakeStore(), OperationLog(settings.runtime_dir))
    tools = await application.mcp.get_tools()
    assert set(tools) == {"qdrant-find", "knowledge-index-status"}


@pytest.mark.anyio
async def test_status_tool_is_read_only(settings):
    from knowledge_mcp.server import create_application

    log = OperationLog(settings.runtime_dir)
    application = create_application(settings, FakeStore(), log)
    result = await application.status()
    assert isinstance(result, dict)
    assert result["embedding_device"] == "cpu"
    assert "qdrant-store" not in (await application.mcp.get_tools())


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
    tool = (await application.mcp.get_tools())["qdrant-find"]
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
    assert result["reranker"] == {"model": "dragonkue/bge-reranker-v2-m3-ko", "loaded": False}
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
