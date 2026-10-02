from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from knowledge_mcp.config import Settings
from knowledge_mcp.documents import (
    MAX_FILE_SIZE_BYTES,
    Chunk,
    discover_sources,
    parse_source,
)
from knowledge_mcp.indexer import KnowledgeIndexer


@pytest.fixture
def settings(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    project = tmp_path / "project"
    project.mkdir()
    return Settings.from_paths(vault_root=vault, project_root=project)


def test_file_size_limit_in_discover_and_parse(settings, monkeypatch):
    large_file = settings.vault_root / "large.md"
    large_file.write_text("Hello world", encoding="utf-8")

    real_stat = large_file.stat()
    fake_stat = type("FakeStat", (), {
        "st_size": MAX_FILE_SIZE_BYTES + 1024,
        "st_ctime": real_stat.st_ctime,
        "st_mtime": real_stat.st_mtime,
        "st_mode": real_stat.st_mode,
    })()

    orig_stat = Path.stat

    def fake_stat_func(self, *args, **kwargs):
        if self.name == "large.md":
            return fake_stat
        return orig_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", fake_stat_func)

    # 1. discover_sources should skip the large file
    discovered = discover_sources(settings)
    assert large_file.resolve() not in discovered

    # 2. parse_source should return skipped_large_file without reading
    doc = parse_source(large_file, settings)
    assert doc.status == "skipped_large_file"
    assert "30MB" in (doc.error or "")
    assert doc.sections == ()


@pytest.mark.anyio
async def test_indexer_handles_skipped_large_file(settings, monkeypatch):
    large_file = settings.vault_root / "large.md"
    large_file.write_text("dummy", encoding="utf-8")

    class FakeStore:
        async def ensure_schema(self):
            pass

        async def collection_inventory(self):
            from knowledge_mcp.qdrant_store import CollectionInventory
            return CollectionInventory()

        async def generation_matches(self, *args):
            return False

        async def replace_generation(self, *args, **kwargs):
            return []

        async def cleanup_orphans(self, *args, **kwargs):
            pass

    class FakeTokenizer:
        def encode(self, text):
            return [1, 2, 3]

        def decode(self, tokens):
            return "text"

    indexer = KnowledgeIndexer(settings, FakeStore(), FakeTokenizer())

    monkeypatch.setattr("knowledge_mcp.indexer.discover_sources", lambda _: [large_file])
    fake_doc = type("Doc", (), {
        "status": "skipped_large_file",
        "source_path": "large.md",
        "file_type": "md",
        "sections": (),
    })()
    monkeypatch.setattr(indexer, "parser", lambda path, s: fake_doc)

    summary = await indexer.sync()
    assert summary.skipped == 1
    assert summary.failed == 0


@pytest.mark.anyio
async def test_qdrant_store_chunks_upsert_into_batches_of_64(settings):
    from knowledge_mcp.qdrant_store import KnowledgeStore

    class FakeProvider:
        def get_vector_size(self):
            return 1024

        async def embed_documents(self, docs):
            return [[0.1] * 1024 for _ in docs]

    store = KnowledgeStore(settings, FakeProvider())
    upsert_batches = []

    async def fake_upsert(collection, points, wait=True):
        upsert_batches.append(len(points))

    store.client.upsert = fake_upsert
    store.client.delete = AsyncMock()

    chunks = [
        Chunk(
            source_path="test.md",
            document=f"chunk {i}",
            embedding_text=f"chunk {i}",
            chunk_index=i,
            document_type="other",
            document_type_source="fallback",
            file_type="md",
            security_level="public",
            created_at="2025-01-01T00:00:00+09:00",
            modified_at="2025-01-01T00:00:00+09:00",
        )
        for i in range(150)
    ]

    await store.replace_generation("test.md", "gen1", chunks, file_hash="hash1")
    assert upsert_batches == [64, 64, 22]


def test_torch_thread_limit_called():
    from knowledge_mcp.embeddings import _limit_cpu_threads

    with patch("torch.set_num_threads") as mock_set_threads:
        _limit_cpu_threads()
        assert mock_set_threads.called
        args, _ = mock_set_threads.call_args
        assert args[0] <= 4


def test_resource_settings_preserve_positional_calls_and_parse_env(tmp_path, monkeypatch):
    from dataclasses import replace
    monkeypatch.setattr("knowledge_mcp.config._load_env_file", lambda *_: None)
    monkeypatch.setenv("KNOWLEDGE_VAULT_ROOT", str(tmp_path / "vault"))
    monkeypatch.setenv("KNOWLEDGE_PROJECT_ROOT", str(tmp_path / "project"))
    values = {
        "KNOWLEDGE_EMBEDDING_BATCH_SIZE": "12", "KNOWLEDGE_RERANKER_BATCH_SIZE": "6",
        "KNOWLEDGE_CPU_THREADS": "2", "KNOWLEDGE_CUDA_MEMORY_FRACTION": "0.24",
        "KNOWLEDGE_ONNX_GPU_MEM_LIMIT": "268435456",
        "KNOWLEDGE_ONNX_ARENA_EXTEND_STRATEGY": "kSameAsRequested",
        "KNOWLEDGE_ONNX_INTRA_OP_NUM_THREADS": "3",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    configured = Settings.from_env("test")
    assert (configured.embedding_batch_size, configured.reranker_batch_size, configured.cpu_threads) == (12, 6, 2)
    assert configured.cuda_memory_fraction == .24
    assert (configured.onnx_gpu_mem_limit, configured.onnx_arena_extend_strategy, configured.onnx_intra_op_num_threads) == (268435456, "kSameAsRequested", 3)
    positional = Settings(tmp_path / "vault", tmp_path / "runtime", Settings.DEFAULT_QDRANT_URL,
                          "test", Settings.DEFAULT_DENSE_MODEL, "test", tmp_path / "project")
    assert positional.project_root == tmp_path / "project"
    assert replace(positional, embedding_batch_size=1).embedding_batch_size == 1


@pytest.mark.parametrize("name,value", [
    ("embedding_batch_size", 0), ("embedding_batch_size", 257),
    ("reranker_batch_size", -1), ("cpu_threads", 0),
    ("onnx_intra_op_num_threads", 0), ("onnx_gpu_mem_limit", 0),
    ("onnx_arena_extend_strategy", "invalid"), ("cuda_memory_fraction", 0),
    ("cuda_memory_fraction", 1.01), ("cuda_memory_fraction", float("nan")),
    ("embedding_batch_size", True), ("cpu_threads", 1.5),
])
def test_resource_settings_reject_invalid_values(settings, name, value):
    from dataclasses import replace
    with pytest.raises(ValueError, match=name):
        replace(settings, **{name: value})


def test_resource_settings_do_not_change_index_generation(settings):
    from dataclasses import replace
    source = settings.vault_root / "synthetic.md"
    source.write_text("synthetic input", encoding="utf-8")
    original = KnowledgeIndexer(settings, object(), object())
    tuned = KnowledgeIndexer(replace(settings, embedding_batch_size=4, cpu_threads=2,
                                    cuda_memory_fraction=.24), object(), object())
    assert original._fingerprint(source) == tuned._fingerprint(source)


def test_torch_resources_clamp_threads_and_set_cuda_budget_once(monkeypatch):
    import sys
    from types import SimpleNamespace
    from concurrent.futures import ThreadPoolExecutor
    from knowledge_mcp import embeddings
    calls = []
    thread_count = [8]
    def set_threads(count):
        calls.append(("threads", count))
        thread_count[0] = count
    fake_torch = SimpleNamespace(
        get_num_threads=lambda: thread_count[0], set_num_threads=set_threads,
        cuda=SimpleNamespace(is_available=lambda: True,
                             set_per_process_memory_fraction=lambda fraction: calls.append(("fraction", fraction))),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(embeddings.os, "cpu_count", lambda: 2)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(embeddings.configure_torch_resources, 4, .24) for _ in range(2)]
        for future in futures:
            future.result()
    assert calls.count(("threads", 2)) == 1
    assert calls.count(("fraction", .24)) == 1


def test_onnx_runtime_preparation_preserves_torch_resource_settings(monkeypatch):
    import sys
    from types import SimpleNamespace
    from knowledge_mcp import embeddings
    thread_count = [8]
    monkeypatch.setattr(embeddings.os, "cpu_count", lambda: 8)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        get_num_threads=lambda: thread_count[0], set_num_threads=lambda count: thread_count.__setitem__(0, count)))
    monkeypatch.setattr(embeddings.ort, "preload_dlls", lambda: None)
    embeddings.configure_torch_resources(2)
    embeddings.prepare_cuda_runtime()
    assert thread_count == [2]


def test_cpu_only_resources_do_not_touch_cuda_allocator(monkeypatch):
    import sys
    from types import SimpleNamespace
    from knowledge_mcp import embeddings
    def forbidden(*_):
        pytest.fail("CPU inference touched CUDA allocator")
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        set_num_threads=lambda *_: None, get_num_threads=lambda: 4,
        cuda=SimpleNamespace(is_available=lambda: False, set_per_process_memory_fraction=forbidden)))
    embeddings.configure_torch_resources(4, .24)


@pytest.mark.parametrize("backend", ["embedding", "reranker"])
def test_integer_cuda_fraction_reaches_real_torch_validation(settings, monkeypatch, backend):
    """Replace CUDA driver/model construction only; retain Torch type validation."""
    from dataclasses import replace
    import sys
    from types import SimpleNamespace
    import torch
    from torch.cuda import memory
    from knowledge_mcp import embeddings
    from knowledge_mcp.reranker import LocalReranker
    native_calls = []
    monkeypatch.setattr(memory, "_lazy_init", lambda: None)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch._C, "_cuda_setMemoryFraction",
                        lambda fraction, device: native_calls.append((fraction, device)), raising=False)
    monkeypatch.setattr(torch, "get_num_threads", lambda: 4)
    monkeypatch.setattr(embeddings.os, "cpu_count", lambda: 4)
    monkeypatch.setattr(embeddings, "_configured_cuda_budget", None)
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(
        SentenceTransformer=lambda *_: object(), CrossEncoder=lambda *_, **__: object()))
    configured = replace(settings, cuda_memory_fraction=1)
    provider = (embeddings.LocalSentenceTransformerProvider(settings.dense_model, settings=configured)
                if backend == "embedding" else LocalReranker(settings=configured))
    provider.warmup()
    assert native_calls == [(1.0, 0)]
    assert type(native_calls[0][0]) is float


def test_model_inference_is_serialized_across_embedding_and_reranker(monkeypatch):
    import threading
    import sys
    from types import SimpleNamespace
    from concurrent.futures import ThreadPoolExecutor
    import numpy as np
    from knowledge_mcp.embeddings import LocalSentenceTransformerProvider
    from knowledge_mcp.reranker import LocalReranker
    entered, release, attempted, predicted = (threading.Event() for _ in range(4))
    provider = LocalSentenceTransformerProvider("dragonkue/BGE-m3-ko")
    def encode(*_, **__):
        entered.set()
        assert release.wait(5)
        return np.zeros((1, 1024))
    provider._model = SimpleNamespace(encode=encode)
    reranker = LocalReranker()
    reranker._model = SimpleNamespace(predict=lambda *_, **__: predicted.set() or [.2])
    def score():
        attempted.set()
        return reranker._score("synthetic", [SimpleNamespace(document="synthetic")])
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(provider._encode, ["synthetic"])
        assert entered.wait(5)
        second = executor.submit(score)
        try:
            assert attempted.wait(5)
            assert not predicted.wait(.1), "reranking overlapped native embedding inference"
        finally:
            release.set()
        first.result()
        assert second.result() == [.2]
    assert predicted.is_set()


@pytest.mark.anyio
async def test_cli_dependencies_apply_selected_resource_settings(settings, monkeypatch):
    from dataclasses import replace
    import sys
    from types import SimpleNamespace
    import numpy as np
    from knowledge_mcp.cli import _dependencies
    calls = []
    class Model:
        tokenizer = object()
        def __init__(self, name):
            pass
        def encode(self, texts, *, batch_size, normalize_embeddings):
            calls.append(("embedding", batch_size))
            return np.zeros((len(texts), 1024))
    class CrossEncoder:
        def __init__(self, name, *, device):
            pass
        def predict(self, pairs, *, batch_size):
            calls.append(("reranker", batch_size))
            return [.2]
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=Model, CrossEncoder=CrossEncoder))
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    configured = replace(settings, embedding_batch_size=3, reranker_batch_size=2)
    stores, _, _ = _dependencies(configured)
    assert not stores.reranker.loaded
    store = stores.stores[settings.dense_model]
    from knowledge_mcp.search import SearchResult
    candidate = SearchResult(point_id="test", document="synthetic", source_path="synthetic.md",
                             document_type="other", file_type="md", security_level="public",
                             created_at="2025-01-01T00:00:00+09:00", modified_at="2025-01-01T00:00:00+09:00", score=.1)
    try:
        await store.embedding_provider.embed_documents(["synthetic"])
        await stores.reranker.rerank("synthetic", [candidate])
    finally:
        await store.client.close()
    assert calls == [("embedding", 3), ("reranker", 2)]
