"""Real localhost Qdrant tests; run the project's Compose service first."""

from dataclasses import replace
from hashlib import sha256
from uuid import uuid4

import pytest
from qdrant_client import AsyncQdrantClient, models

from knowledge_mcp.config import Settings
from knowledge_mcp.documents import Chunk


@pytest.fixture
def anyio_backend():
    return "asyncio"


class FakeEmbeddingProvider:
    """Same async contract as the official provider, without model downloads."""

    def get_vector_size(self):
        return 1024

    async def embed_documents(self, documents):
        return [[1.0] + [0.0] * 1023 for _ in documents]

    async def embed_query(self, query):
        return [1.0] + [0.0] * 1023


def chunk(path, text="프로젝트 알파", **changes):
    return replace(Chunk(
        source_path=path, document=text, embedding_text="제목\n" + text,
        chunk_index=0, document_type="experience", document_type_source="frontmatter",
        file_type="md", security_level="public",
        created_at="2025-03-01T10:30:00+09:00", modified_at="2026-09-14T23:59:00+09:00",
        start_line=3, end_line=5,
    ), **changes)


@pytest.fixture
async def store(tmp_path):
    from knowledge_mcp.qdrant_store import KnowledgeStore

    settings = Settings(tmp_path, tmp_path / ".knowledge", Settings.DEFAULT_QDRANT_URL,
                        "knowledge_test_" + uuid4().hex, Settings.DEFAULT_DENSE_MODEL, "test")
    instance = KnowledgeStore(settings, FakeEmbeddingProvider())
    try:
        await instance.ensure_schema()
        yield instance
    finally:
        await instance.client.delete_collection(settings.collection_name)
        await instance.client.close()


@pytest.mark.anyio
async def test_server_accepts_multilingual_bm25_document():
    """Characterize server inference: no client-side tokenizer can mask failure."""
    name = "knowledge_bm25_probe_" + uuid4().hex
    client = AsyncQdrantClient(url=Settings.DEFAULT_QDRANT_URL, cloud_inference=True)
    try:
        assert (await client.info()).version == "1.19.1"
        await client.create_collection(name, vectors_config={}, sparse_vectors_config={
            "bm25": models.SparseVectorParams(modifier=models.Modifier.IDF),
        })
        document = models.Document(text="프로젝트 알파 Kubernetes", model="qdrant/bm25",
                                   options={"tokenizer": "multilingual", "language": "none"})
        await client.upsert(name, [models.PointStruct(id=1, vector={"bm25": document})], wait=True)
        result = await client.query_points(name, query=document, using="bm25", with_vectors=True)
        assert [point.id for point in result.points] == [1]
        assert result.points[0].vector["bm25"].indices
    finally:
        await client.delete_collection(name)
        await client.close()


@pytest.mark.anyio
async def test_schema_and_default_private_exclusion(store):
    from knowledge_mcp.search import SearchRequest

    await store.replace_generation("public.md", "g1", [chunk("public.md")], file_hash="a" * 64)
    await store.replace_generation("private.md", "g2", [chunk("private.md", security_level="private")], file_hash="b" * 64)
    results = await store.hybrid_search(SearchRequest(query="프로젝트 알파"))
    assert {r.source_path for r in results} == {"public.md"}
    assert results[0].document == "프로젝트 알파"
    assert (results[0].start_line, results[0].end_line) == (3, 5)
    included = await store.hybrid_search(SearchRequest(query="프로젝트 알파", include_private=True))
    assert {r.source_path for r in included} == {"public.md", "private.md"}
    info = await store.client.get_collection(store.settings.collection_name)
    assert info.config.params.vectors["dense"].size == 1024
    assert info.config.params.sparse_vectors["bm25"].modifier == models.Modifier.IDF
    assert {key: value.data_type for key, value in info.payload_schema.items()} == {
        "metadata.source_path": "keyword", "metadata.created_at": "datetime",
        "metadata.modified_at": "datetime", "metadata.document_type": "keyword",
        "metadata.file_type": "keyword", "metadata.security_level": "keyword",
        "metadata.generation": "keyword",
    }


@pytest.mark.anyio
async def test_model_reads_selected_collection(tmp_path):
    from knowledge_mcp.qdrant_store import KnowledgeStore
    from knowledge_mcp.retrieval import MultiModelStore
    from knowledge_mcp.search import SearchRequest

    name = "knowledge_bge_" + uuid4().hex
    model = "dragonkue/BGE-m3-ko"
    store = KnowledgeStore(
        Settings(tmp_path, tmp_path / ".knowledge", Settings.DEFAULT_QDRANT_URL, name, model, "test"),
        FakeEmbeddingProvider(),
    )
    router = MultiModelStore({model: store})
    try:
        await store.ensure_schema()
        await store.replace_generation("bge.md", "g1", [chunk("bge.md")], file_hash="a" * 64)
        found = await router.hybrid_search(SearchRequest(query="프로젝트 알파", embedding_model=model))
        assert {item.source_path for item in found} == {"bge.md"}
    finally:
        if await store.client.collection_exists(store.settings.collection_name):
            await store.client.delete_collection(store.settings.collection_name)
        await store.client.close()


@pytest.mark.anyio
async def test_generation_replacement_failure_cleanup_and_deletion(store):
    source = "public.md"
    first = await store.replace_generation(source, "g1", [chunk(source)], file_hash="a" * 64)
    assert await store.replace_generation(source, "g1", [chunk(source)], file_hash="a" * 64) == first
    with pytest.raises(ValueError):
        await store.replace_generation(source, "g2", [chunk("wrong.md")], file_hash="b" * 64)
    original = store.embedding_provider.embed_documents

    async def failed_embedding(documents):
        raise RuntimeError("synthetic embedding failure")

    store.embedding_provider.embed_documents = failed_embedding
    with pytest.raises(RuntimeError):
        await store.replace_generation(source, "g2", [chunk(source)], file_hash="b" * 64)
    store.embedding_provider.embed_documents = original
    retained = await store.client.retrieve(store.settings.collection_name, first)
    assert len(retained) == 1
    second = await store.replace_generation(source, "g2", [chunk(source, "새 프로젝트")], file_hash="b" * 64)
    assert first != second
    assert not await store.client.retrieve(store.settings.collection_name, first)
    await store.replace_generation("orphan.md", "unfinished", [chunk("orphan.md")], file_hash="c" * 64)
    await store.cleanup_orphans({source: "g2"})
    points, _ = await store.client.scroll(store.settings.collection_name)
    assert [str(p.id) for p in points] == second
    await store.delete_source(source)
    assert (await store.client.count(store.settings.collection_name)).count == 0


@pytest.mark.anyio
async def test_generation_match_checks_exact_point_ids(store):
    point_ids = await store.replace_generation("exact.md", "g1", [chunk("exact.md")], file_hash="a" * 64)
    assert await store.generation_matches("exact.md", "g1", 1, point_ids)
    assert not await store.generation_matches("exact.md", "g1", 1, ["different-point"])


@pytest.mark.anyio
async def test_filtered_rrf_and_per_source_cap(store, monkeypatch):
    from knowledge_mcp.search import SearchRequest

    for name, changes in [
        ("match.md", {}), ("private.md", {"security_level": "private"}),
        ("wrong_type.md", {"document_type": "diary"}),
        ("wrong_format.pdf", {"file_type": "pdf"}),
        ("old.md", {"modified_at": "2026-09-13T23:59:00+09:00"}),
        ("future.md", {"created_at": "2025-03-02T00:00:00+09:00"}),
    ]:
        await store.replace_generation(name, "g1", [chunk(name, chunk_index=i, **changes) for i in range(3)], file_hash="a" * 64)
    original = store.client.query_points
    observed = []

    async def capture(*args, **kwargs):
        observed.append(kwargs)
        return await original(*args, **kwargs)

    monkeypatch.setattr(store.client, "query_points", capture)
    results = await store.hybrid_search(SearchRequest(
        query="프로젝트 알파", document_type=["experience"], file_type=["md"],
        created_from="2025-03-01", created_to="2025-03-01",
        modified_from="2026-09-14", modified_to="2026-09-14", limit=20,
    ))
    assert len(results) == 2
    assert {r.source_path for r in results} == {"match.md"}
    assert all(r.score > 0 for r in results)
    call = observed[0]
    assert call["query"].fusion == models.Fusion.RRF
    assert [branch.limit for branch in call["prefetch"]] == [20, 20]
    assert all(branch.filter == call["query_filter"] for branch in call["prefetch"])
    assert [branch.using for branch in call["prefetch"]] == ["dense", "bm25"]
    assert call["prefetch"][1].query.options == {"tokenizer": "multilingual", "language": "none"}


@pytest.mark.anyio
async def test_schema_mismatch_requires_reindex_without_changing_collection(store):
    info = await store.client.get_collection(store.settings.collection_name)
    altered = dict(info.config.metadata)
    altered["dense_model"] = "wrong/model"
    await store.client.update_collection(store.settings.collection_name, metadata=altered)
    with pytest.raises(ValueError, match="reindex"):
        await store.ensure_schema()
    after = await store.client.get_collection(store.settings.collection_name)
    assert after.config.metadata == altered


@pytest.mark.parametrize("changes", [
    {"limit": 0}, {"limit": 21}, {"query": "  "},
    {"created_from": "2026-09-15", "created_to": "2026-09-14"},
    {"modified_from": "2026-09-15", "modified_to": "2026-09-14"},
])
def test_invalid_search_requests_rejected(changes):
    from knowledge_mcp.search import SearchRequest

    with pytest.raises(ValueError):
        SearchRequest(**{"query": "프로젝트", **changes})


@pytest.mark.anyio
async def test_empty_generation_removes_all_existing_source_points(store):
    await store.replace_generation("empty.md", "g1", [chunk("empty.md")], file_hash="a" * 64)
    assert await store.replace_generation("empty.md", "g1", [], file_hash="a" * 64) == []
    assert (await store.client.count(store.settings.collection_name)).count == 0


@pytest.mark.anyio
async def test_interrupted_upsert_keeps_completed_generation_until_cleanup(store, monkeypatch):
    previous = await store.replace_generation("public.md", "g1", [chunk("public.md")], file_hash="a" * 64)
    original = store.client.upsert

    async def interrupted_upsert(collection_name, points, **kwargs):
        # Real server accepts one new point, then the simulated connection fails.
        await original(collection_name, points[:1], **kwargs)
        raise ConnectionError("synthetic interrupted write")

    monkeypatch.setattr(store.client, "upsert", interrupted_upsert)
    with pytest.raises(ConnectionError):
        await store.replace_generation("public.md", "g2", [chunk("public.md", chunk_index=i) for i in range(2)], file_hash="b" * 64)
    assert len(await store.client.retrieve(store.settings.collection_name, previous)) == 1
    assert (await store.client.count(store.settings.collection_name)).count == 2
    await store.cleanup_orphans({"public.md": "g1"})
    points, _ = await store.client.scroll(store.settings.collection_name)
    assert [str(p.id) for p in points] == previous


@pytest.mark.anyio
@pytest.mark.parametrize("existing", [False, True])
async def test_three_dimensional_provider_cannot_create_or_accept_schema(store, monkeypatch, existing):
    name = store.settings.collection_name
    metadata = (await store.client.get_collection(name)).config.metadata
    await store.client.delete_collection(name)
    if existing:
        await store.client.create_collection(
            name, vectors_config={"dense": models.VectorParams(size=3, distance=models.Distance.COSINE)},
            sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
            metadata=metadata,
        )
    monkeypatch.setattr(store.embedding_provider, "get_vector_size", lambda: 3)
    with pytest.raises(ValueError, match="1024"):
        await store.ensure_schema()
    if existing:
        info = await store.client.get_collection(name)
        assert info.config.params.vectors["dense"].size == 3
        assert not info.payload_schema
    else:
        assert not await store.client.collection_exists(name)


@pytest.mark.anyio
async def test_metadata_generation_preserves_file_hash_and_old_points_until_cleanup(store, monkeypatch):
    source = "sidecar.txt"
    source_chunk = chunk(source, file_type="txt")
    file_hash = sha256(source_chunk.document.encode("utf-8")).hexdigest()
    previous = await store.replace_generation(source, "public-sidecar-generation", [source_chunk], file_hash=file_hash)
    original_delete = store.client.delete
    before_cleanup = []

    async def inspect_before_cleanup(*args, **kwargs):
        points, _ = await store.client.scroll(store.settings.collection_name)
        before_cleanup.extend(points)
        return await original_delete(*args, **kwargs)

    monkeypatch.setattr(store.client, "delete", inspect_before_cleanup)
    updated = await store.replace_generation(
        source, "private-sidecar-generation", [replace(source_chunk, security_level="private")], file_hash=file_hash,
    )
    assert set(previous).isdisjoint(updated)
    assert len(before_cleanup) == 2
    old_point = next(p for p in before_cleanup if str(p.id) == previous[0])
    new_point = next(p for p in before_cleanup if str(p.id) == updated[0])
    assert old_point.payload["metadata"]["security_level"] == "public"
    assert old_point.payload["metadata"]["generation"] == "public-sidecar-generation"
    assert new_point.payload["metadata"]["security_level"] == "private"
    assert new_point.payload["metadata"]["generation"] == "private-sidecar-generation"
    assert old_point.payload["metadata"]["file_hash"] == new_point.payload["metadata"]["file_hash"] == file_hash
    assert old_point.payload["document"] == new_point.payload["document"] == source_chunk.document
    assert not await store.client.retrieve(store.settings.collection_name, previous)
    assert len(await store.client.retrieve(store.settings.collection_name, updated)) == 1
