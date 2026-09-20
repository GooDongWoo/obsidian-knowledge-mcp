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

        async def generation_matches(self, *args):
            return False

        async def replace_generation(self, *args, **kwargs):
            return []

        async def cleanup_orphans(self, *args):
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
