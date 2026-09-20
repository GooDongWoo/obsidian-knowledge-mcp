"""Indexer regressions against temporary sources, SQLite and localhost Qdrant."""

from dataclasses import replace
from hashlib import sha256
from uuid import uuid4

import pytest

from knowledge_mcp.config import Settings
from knowledge_mcp.documents import chunk_document, parse_source
from knowledge_mcp.qdrant_store import KnowledgeStore
from knowledge_mcp.state import Manifest, index_lock
from test_qdrant_integration import FakeEmbeddingProvider


@pytest.fixture
def anyio_backend():
    return "asyncio"


class WordTokenizer:
    def encode(self, text):
        return text.split()

    def decode(self, tokens):
        return " ".join(tokens)


@pytest.fixture
async def indexer(tmp_path):
    from knowledge_mcp.indexer import KnowledgeIndexer

    root = tmp_path / "vault"
    root.mkdir()
    project = tmp_path / "project"
    project.mkdir()
    (root / "sample.md").write_text("# Sample\nOriginal source body", encoding="utf-8")
    settings = Settings(root, tmp_path / "runtime", Settings.DEFAULT_QDRANT_URL,
                        "knowledge_indexer_test_" + uuid4().hex, Settings.DEFAULT_DENSE_MODEL, "test",
                        project_root=project)
    store = KnowledgeStore(settings, FakeEmbeddingProvider())
    instance = KnowledgeIndexer(settings, store, WordTokenizer())
    try:
        yield instance
    finally:
        if await store.client.collection_exists(settings.collection_name):
            await store.client.delete_collection(settings.collection_name)
        await store.client.close()


async def points(indexer):
    records, _ = await indexer.store.client.scroll(indexer.settings.collection_name, limit=100)
    return records


def test_default_manifest_uses_store_collection(tmp_path):
    from knowledge_mcp.indexer import KnowledgeIndexer

    root = tmp_path / "vault"
    root.mkdir()
    settings = Settings(root, tmp_path / "runtime", Settings.DEFAULT_QDRANT_URL,
                        "custom_collection", Settings.DEFAULT_DENSE_MODEL, "test",
                        project_root=tmp_path)
    indexer = KnowledgeIndexer(settings, object(), WordTokenizer())
    selected = Manifest(settings.runtime_dir, settings.collection_name)
    selected.mark_complete("prior.md", "old-hash", "old-generation", 1)

    assert "prior.md" in indexer.manifest.completed_files()


@pytest.mark.anyio
async def test_sync_accepts_outer_index_lock(tmp_path, monkeypatch):
    import knowledge_mcp.indexer as module

    class EmptyStore:
        async def ensure_schema(self):
            pass

        async def cleanup_orphans(self, _committed):
            pass

    root = tmp_path / "vault"
    root.mkdir()
    settings = Settings(root, tmp_path / "runtime", Settings.DEFAULT_QDRANT_URL,
                        "custom_collection", Settings.DEFAULT_DENSE_MODEL, "test",
                        project_root=tmp_path)
    indexer = module.KnowledgeIndexer(settings, EmptyStore(), WordTokenizer())

    def unexpected_lock(_runtime_dir):
        raise AssertionError("sync acquired a second index lock")

    monkeypatch.setattr(module, "index_lock", unexpected_lock)
    with index_lock(settings.runtime_dir):
        result = await indexer.sync(assume_locked=True)

    assert (result.added, result.failed) == (0, 0)


def fail(*args, **kwargs):
    raise ValueError("SENSITIVE_DOCUMENT_BODY")


async def async_fail(*args, **kwargs):
    fail()


@pytest.mark.anyio
async def test_unchanged_file_is_not_reembedded(indexer, monkeypatch):
    first = await indexer.sync()
    before = await points(indexer)
    monkeypatch.setattr(indexer.store.embedding_provider, "embed_documents", async_fail)
    second = await indexer.sync()
    assert (first.added, first.failed) == (1, 0)
    assert (second.unchanged, second.added, second.changed, second.failed) == (1, 0, 0, 0)
    assert await points(indexer) == before
    assert indexer.operation_log.index_status()["status"] == "completed"


@pytest.mark.anyio
async def test_modified_deleted_and_excluded_sources(indexer):
    await indexer.sync()
    root = indexer.settings.vault_root
    old = (await points(indexer))[0].id
    (root / "sample.md").write_text("Updated source body", encoding="utf-8")
    result = await indexer.sync()
    current = await points(indexer)
    assert (result.changed, result.failed) == (1, 0)
    assert current[0].id != old
    assert current[0].payload["document"] == "Updated source body"
    (root / "sample.md").unlink()
    result = await indexer.sync()
    assert result.deleted == 1
    assert await points(indexer) == []
    (root / "sample.md").write_text("Back again", encoding="utf-8")
    await indexer.sync()
    (indexer.settings.project_root / ".knowledgeignore").write_text("sample.md\n", encoding="utf-8")
    result = await indexer.sync()
    assert result.deleted == 1
    assert await points(indexer) == []
    assert indexer.manifest.completed_files() == {}


@pytest.mark.anyio
@pytest.mark.parametrize("stage", ["parse", "embed", "upload"])
async def test_failed_update_preserves_previous_and_isolates_other_files(indexer, monkeypatch, stage):
    await indexer.sync()
    previous = (await points(indexer))[0]
    root = indexer.settings.vault_root
    (root / "sample.md").write_text("SENSITIVE_DOCUMENT_BODY changed", encoding="utf-8")
    (root / "working.md").write_text("Healthy source", encoding="utf-8")
    if stage == "parse":
        def parser(path, settings):
            return fail() if path.name == "sample.md" else parse_source(path, settings)
        monkeypatch.setattr(indexer, "parser", parser)
    elif stage == "embed":
        original_embed = indexer.store.embedding_provider.embed_documents
        async def embed(texts):
            return fail() if "SENSITIVE_DOCUMENT_BODY" in texts[0] else await original_embed(texts)
        monkeypatch.setattr(indexer.store.embedding_provider, "embed_documents", embed)
    else:
        original_upsert = indexer.store.client.upsert
        async def upsert(name, records, **kwargs):
            if records[0].payload["metadata"]["source_path"] == "sample.md":
                await original_upsert(name, records[:1], **kwargs)
                fail()
            return await original_upsert(name, records, **kwargs)
        monkeypatch.setattr(indexer.store.client, "upsert", upsert)
    result = await indexer.sync()
    retained = await points(indexer)
    assert (result.added, result.changed, result.failed) == (1, 0, 1)
    assert len(retained) == 2
    assert previous in retained
    assert "SENSITIVE_DOCUMENT_BODY" not in indexer.operation_log.database_text_for_test()
    assert "SENSITIVE_DOCUMENT_BODY" not in repr(result)
    assert indexer.operation_log.index_status()["status"] == "partial"


@pytest.mark.anyio
@pytest.mark.parametrize("sidecar", [False, True])
async def test_metadata_changes_reindex_without_source_hash_change(indexer, sidecar):
    root = indexer.settings.vault_root
    (root / "sample.md").unlink()
    source = root / "sample.txt"
    source.write_text("Same source", encoding="utf-8")
    await indexer.sync()
    before = (await points(indexer))[0]
    if sidecar:
        (root / "sample.txt.meta.yaml").write_text("security: private\n", encoding="utf-8")
    else:
        (indexer.settings.project_root / ".knowledge-types.yaml").write_text("types:\n  experience:\n    paths: ['*.txt']\n", encoding="utf-8")
    result = await indexer.sync()
    after = (await points(indexer))[0]
    assert (result.changed, result.failed) == (1, 0)
    assert before.id != after.id
    assert before.payload["metadata"]["file_hash"] == after.payload["metadata"]["file_hash"]
    assert after.payload["metadata"]["file_hash"] == sha256(b"Same source").hexdigest()
    assert after.payload["metadata"]["security_level" if sidecar else "document_type"] == ("private" if sidecar else "experience")


@pytest.mark.anyio
@pytest.mark.parametrize("configuration", ["schema", "dense", "bm25"])
async def test_generation_changes_with_indexing_configuration(indexer, monkeypatch, configuration):
    import knowledge_mcp.indexer as module

    source = indexer.settings.vault_root / "sample.md"
    before_hash, before_generation = indexer._fingerprint(source)
    if configuration == "schema":
        monkeypatch.setattr(module, "PARSER_CHUNKER_SCHEMA", "documents-v2:400:60")
    elif configuration == "dense":
        indexer.settings = replace(indexer.settings, dense_model="test/model-v2")
    else:
        monkeypatch.setitem(module.BM25_OPTIONS, "k", 1.8)
    after_hash, after_generation = indexer._fingerprint(source)
    assert before_generation != after_generation
    assert before_hash == after_hash


@pytest.mark.anyio
async def test_recovers_crash_after_old_deletion_before_manifest_commit(indexer):
    await indexer.sync()
    # The manifest still records G1 and original source bytes. An interrupted G2
    # upload removed G1, then the user reverted the source bytes back to G1.
    # Hash-only planning wrongly skips this file and cleanup deletes surviving G2.
    source = indexer.settings.vault_root / "sample.md"
    chunks = chunk_document(parse_source(source, indexer.settings), indexer.tokenizer)
    await indexer.store.replace_generation("sample.md", "uncommitted-g2", chunks, file_hash="other-hash")
    result = await indexer.sync()
    retained = await points(indexer)
    assert (result.changed, result.failed) == (1, 0)
    assert len(retained) == 1
    assert retained[0].payload["metadata"]["generation"] != "uncommitted-g2"
    assert (await indexer.sync()).unchanged == 1


@pytest.mark.anyio
async def test_failed_crash_recovery_defers_orphan_cleanup(indexer, monkeypatch):
    await indexer.sync()
    source = indexer.settings.vault_root / "sample.md"
    chunks = chunk_document(parse_source(source, indexer.settings), indexer.tokenizer)
    await indexer.store.replace_generation("sample.md", "uncommitted-g2", chunks, file_hash="other-hash")
    before = await points(indexer)
    monkeypatch.setattr(indexer, "parser", fail)
    result = await indexer.sync()
    assert result.failed == 1
    assert await points(indexer) == before


@pytest.mark.anyio
async def test_manifest_commit_failure_does_not_cleanup_uploaded_generation(indexer, monkeypatch):
    await indexer.sync()
    (indexer.settings.vault_root / "sample.md").write_text("Updated source", encoding="utf-8")
    original = indexer.manifest.mark_complete
    monkeypatch.setattr(indexer.manifest, "mark_complete", fail)
    result = await indexer.sync()
    assert result.failed == 1
    assert (await points(indexer))[0].payload["document"] == "Updated source"
    monkeypatch.setattr(indexer.manifest, "mark_complete", original)
    assert (await indexer.sync()).changed == 1
    assert (await indexer.sync()).unchanged == 1


@pytest.mark.anyio
async def test_no_text_pdf_is_skipped_but_corrupt_update_keeps_previous(indexer):
    from pypdf import PdfWriter

    root = indexer.settings.vault_root
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    with (root / "scan.pdf").open("wb") as output:
        writer.write(output)
    result = await indexer.sync()
    assert (result.added, result.skipped, result.failed) == (1, 1, 0)
    (root / "sample.md").write_text("---\nsecurity: unknown\n---\nNew text", encoding="utf-8")
    before = await points(indexer)
    result = await indexer.sync()
    assert result.failed == 1
    assert await points(indexer) == before


@pytest.mark.anyio
async def test_empty_source_deletes_previous_points_and_stays_unchanged(indexer):
    await indexer.sync()
    (indexer.settings.vault_root / "sample.md").write_text("", encoding="utf-8")
    result = await indexer.sync()
    assert result.changed == 1
    assert await points(indexer) == []
    assert (await indexer.sync()).unchanged == 1


@pytest.mark.anyio
async def test_delete_failure_retains_manifest_for_retry(indexer, monkeypatch):
    await indexer.sync()
    (indexer.settings.vault_root / "sample.md").unlink()
    original = indexer.store.delete_source
    monkeypatch.setattr(indexer.store, "delete_source", async_fail)
    result = await indexer.sync()
    assert (result.deleted, result.failed) == (0, 1)
    assert len(await points(indexer)) == 1
    monkeypatch.setattr(indexer.store, "delete_source", original)
    assert (await indexer.sync()).deleted == 1


@pytest.mark.anyio
async def test_source_change_during_parse_is_retried_without_overwriting_old_generation(indexer, monkeypatch):
    await indexer.sync()
    before = await points(indexer)
    source = indexer.settings.vault_root / "sample.md"
    source.write_text("New source", encoding="utf-8")
    def changing_parser(path, settings):
        document = parse_source(path, settings)
        path.write_text("Changed during parse", encoding="utf-8")
        return document
    monkeypatch.setattr(indexer, "parser", changing_parser)
    result = await indexer.sync()
    assert result.failed == 1
    assert await points(indexer) == before
