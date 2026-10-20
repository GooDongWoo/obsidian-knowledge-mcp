"""Precision identity and observability without real model construction."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import pytest

from knowledge_mcp.config import Settings
from knowledge_mcp.indexer import KnowledgeIndexer
from knowledge_mcp.state import OperationLog


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def settings(tmp_path):
    vault, project = tmp_path / "vault", tmp_path / "project"
    vault.mkdir()
    project.mkdir()
    return Settings.from_paths(vault_root=vault, project_root=project)


def test_precision_changes_generation_and_indexing_version_only(settings):
    source = settings.vault_root / "note.md"
    source.write_text("same source", encoding="utf-8")
    fp32 = KnowledgeIndexer(settings, object(), object(), embedding_precision="float32")
    bf16 = KnowledgeIndexer(settings, object(), object(), embedding_precision="bfloat16")
    assert fp32._fingerprint(source)[0] == bf16._fingerprint(source)[0]
    assert fp32._fingerprint(source)[1] != bf16._fingerprint(source)[1]
    assert fp32._source_state(source)["indexing_version"] != bf16._source_state(source)["indexing_version"]
    assert fp32._source_state(source)["metadata_signature"] == bf16._source_state(source)["metadata_signature"]
    tuned = replace(settings, inference_precision="auto", embedding_batch_size=3,
                    reranker_batch_size=2, cpu_threads=2, cuda_memory_fraction=.1)
    same_dtype = KnowledgeIndexer(tuned, object(), object(), embedding_precision="float32")
    assert same_dtype._fingerprint(source) == fp32._fingerprint(source)
    assert same_dtype._source_state(source) == fp32._source_state(source)


@pytest.mark.parametrize("dtype", ["float32", "bfloat16", None])
def test_dependencies_use_precision_resolved_by_tokenizer(settings, monkeypatch, dtype):
    from knowledge_mcp import cli, embeddings, qdrant_store

    class Provider(embeddings.LocalSentenceTransformerProvider):
        def __init__(self):
            self.effective_dtype = None

        def get_tokenizer(self):
            self.effective_dtype = dtype
            return tokenizer

    tokenizer = object()
    provider = Provider()
    monkeypatch.setattr(embeddings, "create_embedding_provider", lambda *args, **kwargs: provider)
    monkeypatch.setattr(qdrant_store, "KnowledgeStore", lambda selected, p: SimpleNamespace(settings=selected))
    if dtype is None:
        with pytest.raises(RuntimeError, match="precision"):
            cli._dependencies(settings)
    else:
        _, _, indexers = cli._dependencies(settings)
        indexer = indexers[settings.dense_model]
        assert indexer.tokenizer is tokenizer
        assert indexer._indexing_inputs()["embedding_precision"] == dtype


def test_dependencies_preserve_non_torch_provider_contract(settings, monkeypatch):
    from knowledge_mcp import cli, embeddings, qdrant_store

    provider = SimpleNamespace(get_tokenizer=lambda: object())
    monkeypatch.setattr(embeddings, "create_embedding_provider", lambda *args, **kwargs: provider)
    monkeypatch.setattr(qdrant_store, "KnowledgeStore", lambda selected, p: SimpleNamespace(settings=selected))
    _, _, indexers = cli._dependencies(settings)
    assert "embedding_precision" not in indexers[settings.dense_model]._indexing_inputs()


@pytest.mark.parametrize("policy,inherited", [("float32", "auto"), ("auto", "float32")])
def test_daemon_child_receives_settings_precision(settings, monkeypatch, policy, inherited):
    from knowledge_mcp.daemon import start_daemon_process

    monkeypatch.setenv("KNOWLEDGE_INFERENCE_PRECISION", inherited)
    with patch("knowledge_mcp.daemon.is_daemon_running", side_effect=[False, False, True]), \
         patch("knowledge_mcp.daemon.subprocess.Popen", return_value=SimpleNamespace(pid=42)) as launch:
        assert start_daemon_process(replace(settings, inference_precision=policy), timeout=5) == 42
    assert launch.call_args.kwargs["env"]["KNOWLEDGE_INFERENCE_PRECISION"] == policy


@pytest.mark.anyio
@pytest.mark.parametrize("multi", [False, True])
async def test_live_and_cached_status_track_precision_without_loading(settings, multi):
    from knowledge_mcp.server import create_application

    class Provider:
        precision_policy = "auto"
        effective_dtype = None
        device = None

        @property
        def model(self):
            raise AssertionError("status must not load a model")

    class Client:
        async def count(self, *args, **kwargs):
            return SimpleNamespace(count=7)

    provider = Provider()
    selected = SimpleNamespace(settings=settings, embedding_provider=provider, client=Client())
    store = SimpleNamespace(stores={settings.dense_model: selected}) if multi else selected
    application = create_application(settings, store, OperationLog(settings.runtime_dir))
    snapshot = await application.status()
    details = snapshot["models"][settings.dense_model] if multi else snapshot
    assert details["embedding_precision_policy"] == "auto"
    assert details["embedding_effective_dtype"] is None
    application.index_snapshot = snapshot
    application.runtime_status = {"state": "ready"}
    application.operation_log = None  # Cached status cannot query SQLite.
    selected.client = None  # Cached status cannot query Qdrant.
    provider.effective_dtype, provider.device = "bfloat16", "cuda"
    status = await application.status()
    details = status["models"][settings.dense_model] if multi else status
    assert details["point_count"] == 7
    assert details["embedding_device"] == "cuda"
    assert details["embedding_effective_dtype"] == "bfloat16"
    # Refreshes must not mutate the published index snapshot.
    old = snapshot["models"][settings.dense_model] if multi else snapshot
    assert old["embedding_effective_dtype"] is None


@pytest.mark.anyio
async def test_dtype_transition_reembeds_all_sources_then_restart_is_unchanged(settings):
    from test_indexer import WordTokenizer
    from test_qdrant_integration import make_store

    settings = replace(settings, collection_name="precision_test_" + uuid4().hex)
    for name in ["one.md", "two.md"]:
        (settings.vault_root / name).write_text("---\nsecurity: private\n---\nSame document", encoding="utf-8")
    store = await make_store(settings)
    embedded = []
    original = store.embedding_provider.embed_documents

    async def recording(documents):
        embedded.extend(documents)
        return await original(documents)

    store.embedding_provider.embed_documents = recording
    try:
        fp32 = KnowledgeIndexer(settings, store, WordTokenizer(), embedding_precision="float32")
        assert (await fp32.sync()).added == 2
        before = fp32.manifest.completed_files()
        count = len(embedded)
        bf16 = KnowledgeIndexer(settings, store, WordTokenizer(), embedding_precision="bfloat16")
        result = await bf16.sync()
        assert (result.changed, result.failed) == (2, 0)
        assert len(embedded) == count * 2
        after = bf16.manifest.completed_files()
        assert all(before[name].generation != after[name].generation for name in before)
        restarted = KnowledgeIndexer(settings, store, WordTokenizer(), embedding_precision="bfloat16")
        result = await restarted.sync()
        assert (result.unchanged, result.changed, result.failed) == (2, 0, 0)
        assert len(embedded) == count * 2
        records, _ = await store.client.scroll(settings.collection_name, limit=100, with_vectors=True)
        assert {point.payload["metadata"]["source_path"] for point in records} == {"one.md", "two.md"}
        assert all(point.payload["metadata"]["security_level"] == "private" for point in records)
        assert all(point.vector["bm25"].indices for point in records)
    finally:
        if await store.client.collection_exists(settings.collection_name):
            await store.client.delete_collection(settings.collection_name)
        await store.client.close()


@pytest.mark.anyio
async def test_warmup_tokenizer_and_first_request_share_one_precision_resolution(monkeypatch):
    import asyncio
    import sys
    import threading

    import numpy as np

    from knowledge_mcp.embeddings import LocalSentenceTransformerProvider

    entered, release = threading.Event(), threading.Event()
    constructors = []

    class SentenceTransformer:
        tokenizer = SimpleNamespace(encode=lambda *args, **kwargs: [1])

        def __init__(self, name, *, device, model_kwargs):
            constructors.append((device, model_kwargs))
            entered.set()
            assert release.wait(5)

        def encode(self, documents, **kwargs):
            return np.full((len(documents), 1024), 1 / 32, dtype=np.float32)

    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("knowledge_mcp.embeddings.configure_torch_resources", lambda *_: None)
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=SentenceTransformer))
    provider = LocalSentenceTransformerProvider("test-model")
    warmup = asyncio.create_task(asyncio.to_thread(provider.warmup))
    tasks = [warmup]
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        tasks.extend([asyncio.create_task(asyncio.to_thread(provider.get_tokenizer)),
                      asyncio.create_task(provider.embed_query("query"))])
        await asyncio.sleep(.1)
        assert len(constructors) == 1
        assert provider.effective_dtype is None
        release.set()
        await asyncio.gather(*tasks)
        assert len(constructors) == 1
        assert provider.effective_dtype == "float32"
    finally:
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)
