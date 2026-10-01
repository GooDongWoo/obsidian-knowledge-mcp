"""Native model boundaries retain ownership during initialization/cancellation."""

import asyncio
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from knowledge_mcp.embeddings import LocalSentenceTransformerProvider
from knowledge_mcp.reranker import LocalReranker
from knowledge_mcp.state import async_index_lock
from test_retrieval import result


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_warmup_and_rerank_construct_one_cross_encoder(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    constructors = []

    class CrossEncoder:
        def __init__(self, name, *, device):
            constructors.append(self)
            entered.set()
            assert release.wait(5)

        def predict(self, pairs):
            return [.5] * len(pairs)

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(CrossEncoder=CrossEncoder))
    reranker = LocalReranker()
    warmup = asyncio.create_task(asyncio.to_thread(reranker.warmup))
    scoring = None
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        scoring = asyncio.create_task(reranker.rerank("query", [result("a", "a.md", .1)]))
        await asyncio.sleep(.1)
        assert len(constructors) == 1
        release.set()
        await warmup
        assert (await scoring)[0].score == .5
        assert len(constructors) == 1
    finally:
        release.set()
        await asyncio.gather(warmup, *([scoring] if scoring else []), return_exceptions=True)


@pytest.mark.anyio
async def test_cancelled_rerank_drains_predict_before_next_request(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    predictions = []

    class CrossEncoder:
        def __init__(self, name, *, device):
            pass

        def predict(self, pairs):
            predictions.append(pairs)
            if len(predictions) == 1:
                entered.set()
                assert release.wait(5)
            return [.5] * len(pairs)

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(CrossEncoder=CrossEncoder))
    reranker = LocalReranker()
    first = asyncio.create_task(reranker.rerank("first", [result("a", "a.md", .1)]))
    second = None
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        first.cancel()
        second = asyncio.create_task(reranker.rerank("second", [result("b", "b.md", .1)]))
        await asyncio.sleep(.1)
        assert not first.done()
        assert len(predictions) == 1
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert (await second)[0].score == .5
    finally:
        release.set()
        await asyncio.gather(first, *([second] if second else []), return_exceptions=True)


@pytest.mark.anyio
@pytest.mark.parametrize("method,argument", [("embed_documents", ["document"]), ("embed_query", "query")])
async def test_first_embedding_model_construction_stays_off_event_loop(monkeypatch, method, argument):
    owner_thread = threading.get_ident()

    class SentenceTransformer:
        def __init__(self, name):
            assert threading.get_ident() != owner_thread, "model construction blocked the event loop"

        def encode(self, documents, **kwargs):
            return np.zeros((len(documents), 1024))

    monkeypatch.setattr("knowledge_mcp.embeddings._limit_cpu_threads", lambda: None)
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=SentenceTransformer))
    provider = LocalSentenceTransformerProvider("test-model")
    vectors = await getattr(provider, method)(argument)
    assert len(vectors) == (1 if method == "embed_documents" else 1024)


@pytest.mark.anyio
async def test_embedding_warmup_and_first_query_construct_one_model(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    constructors = []

    class SentenceTransformer:
        def __init__(self, name):
            constructors.append(self)
            entered.set()
            assert release.wait(5)

        def encode(self, documents, **kwargs):
            return np.zeros((len(documents), 1024))

    monkeypatch.setattr("knowledge_mcp.embeddings._limit_cpu_threads", lambda: None)
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=SentenceTransformer))
    provider = LocalSentenceTransformerProvider("test-model")
    warmup = asyncio.create_task(asyncio.to_thread(provider.warmup))
    query = None
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        query = asyncio.create_task(provider.embed_query("query"))
        await asyncio.sleep(.1)
        assert len(constructors) == 1
        release.set()
        await warmup
        assert len(await query) == 1024
        assert len(constructors) == 1
    finally:
        release.set()
        await asyncio.gather(warmup, *([query] if query else []), return_exceptions=True)


@pytest.fixture(params=["sentence", "fastembed"])
def blocked_provider(monkeypatch, request):
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    calls = []

    class SentenceTransformer:
        def __init__(self, name):
            pass

        def encode(self, documents, **kwargs):
            calls.append(documents)
            entered.set()
            assert release.wait(5)
            finished.set()
            return np.zeros((len(documents), 1024))

    monkeypatch.setattr("knowledge_mcp.embeddings._limit_cpu_threads", lambda: None)
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=SentenceTransformer))
    if request.param == "sentence":
        provider = LocalSentenceTransformerProvider("test-model")
    else:
        from fastembed.common.types import Device
        from knowledge_mcp.embeddings import LocalFastEmbedProvider

        class TextEmbedding(SentenceTransformer):
            def __init__(self, name, **kwargs):
                super().__init__(name)

            def passage_embed(self, documents, **kwargs):
                yield from self.encode(documents)

            def query_embed(self, documents):
                yield from self.encode(documents)

        monkeypatch.setattr("knowledge_mcp.embeddings.select_device", lambda: Device.CPU)
        monkeypatch.setattr("knowledge_mcp.embeddings.TextEmbedding", TextEmbedding)
        provider = LocalFastEmbedProvider("test-model")
    yield provider, entered, release, finished, calls
    release.set()


@pytest.mark.anyio
@pytest.mark.parametrize("method,argument", [("embed_documents", ["document"]), ("embed_query", "query")])
async def test_cancelled_embedding_keeps_writer_until_native_encode_finishes(tmp_path, blocked_provider, method, argument):
    provider, entered, release, finished, calls = blocked_provider
    next_entered = asyncio.Event()

    async def writer():
        async with async_index_lock(tmp_path):
            await getattr(provider, method)(argument)

    async def next_writer():
        async with async_index_lock(tmp_path):
            next_entered.set()

    first = asyncio.create_task(writer())
    second = None
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        first.cancel()
        second = asyncio.create_task(next_writer())
        await asyncio.sleep(.15)
        assert not first.done()
        assert not next_entered.is_set()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert finished.is_set()
        await second
    finally:
        release.set()
        await asyncio.gather(first, *([second] if second else []), return_exceptions=True)


@pytest.mark.anyio
async def test_lifespan_keeps_index_client_open_until_encoding_finishes(tmp_path, monkeypatch, blocked_provider):
    from fastmcp import Client
    from knowledge_mcp import cli, daemon, qdrant_store
    from knowledge_mcp.config import Settings
    from knowledge_mcp.indexer import KnowledgeIndexer
    from knowledge_mcp.state import OperationLog
    from test_indexer import WordTokenizer

    provider, entered, release, finished, calls = blocked_provider
    closed = asyncio.Event()
    stop = asyncio.Event()
    settings = Settings(tmp_path / "vault", tmp_path / "runtime", Settings.DEFAULT_QDRANT_URL,
                        "cancel-native-test", Settings.DEFAULT_DENSE_MODEL, "test", project_root=tmp_path)
    settings.vault_root.mkdir()
    (settings.vault_root / "sample.md").write_text("# Sample\nExample body", encoding="utf-8")

    class QdrantClient:
        async def count(self, *args, **kwargs):
            return SimpleNamespace(count=0)

        async def close(self):
            closed.set()

    monkeypatch.setattr(qdrant_store, "AsyncQdrantClient", lambda **kwargs: QdrantClient())
    store = qdrant_store.KnowledgeStore(settings, provider)

    async def ensure_schema():
        pass

    monkeypatch.setattr(store, "ensure_schema", ensure_schema)
    log = OperationLog(settings.runtime_dir)
    indexer = KnowledgeIndexer(settings, store, WordTokenizer(), operation_log=log)
    monkeypatch.setattr(cli, "ensure_qdrant", lambda _: None)
    monkeypatch.setattr(cli, "_dependencies", lambda _: (store, log, {"test": indexer}))
    application = daemon.create_daemon_application(settings)

    async def lifespan():
        async with Client(application.mcp):
            await stop.wait()

    task = asyncio.create_task(lifespan())
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        stop.set()
        await asyncio.sleep(.15)
        assert not closed.is_set(), "Qdrant client closed while encoding still owns it"
        assert not task.done()
        release.set()
        await asyncio.wait_for(task, 2)
        assert finished.is_set() and closed.is_set()
        assert log.index_status()["error_code"] == "sync_cancelled"
        assert indexer.manifest.completed_files() == {}
    finally:
        release.set()
        stop.set()
        await asyncio.gather(task, return_exceptions=True)
