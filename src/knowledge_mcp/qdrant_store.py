"""Local Qdrant schema, safe generation writes, and filtered hybrid retrieval."""

from collections import Counter
from dataclasses import asdict
import json
from typing import Mapping, Sequence
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import AsyncQdrantClient, models

from .config import Settings
from .documents import Chunk
from .embedding_protocol import EmbeddingProvider
from .search import SearchRequest, SearchResult, search_filter


DENSE_SIZE = 1024
BM25_OPTIONS = {"tokenizer": "multilingual", "language": "none"}
PAYLOAD_INDEXES = {
    "source_path": models.PayloadSchemaType.KEYWORD,
    "created_at": models.PayloadSchemaType.DATETIME,
    "modified_at": models.PayloadSchemaType.DATETIME,
    "document_type": models.PayloadSchemaType.KEYWORD,
    "file_type": models.PayloadSchemaType.KEYWORD,
    "security_level": models.PayloadSchemaType.KEYWORD,
    "generation": models.PayloadSchemaType.KEYWORD,
}


class KnowledgeStore:
    def __init__(self, settings: Settings, embedding_provider: EmbeddingProvider):
        if settings.qdrant_url != Settings.DEFAULT_QDRANT_URL:
            raise ValueError("Qdrant must use the configured localhost URL")
        self.settings = settings
        self.embedding_provider = embedding_provider
        self.vault_id = uuid5(NAMESPACE_URL, str(settings.vault_root.resolve()).casefold())
        # This client flag delegates Document inference to our localhost server.
        # It does not select a cloud endpoint or perform local FastEmbed BM25.
        self.client = AsyncQdrantClient(url=settings.qdrant_url, cloud_inference=True)

    async def ensure_schema(self) -> None:
        name = self.settings.collection_name
        metadata = {
            "schema_version": 1, "vault_id": str(self.vault_id),
            "dense_model": self.settings.dense_model,
            "bm25_model": "qdrant/bm25", "bm25_options": BM25_OPTIONS,
        }
        if self.embedding_provider.get_vector_size() != DENSE_SIZE:
            raise ValueError("Dense model requires 1024-dimensional embeddings")
        if not await self.client.collection_exists(name):
            await self.client.create_collection(
                name, vectors_config={"dense": models.VectorParams(size=DENSE_SIZE, distance=models.Distance.COSINE)},
                sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
                metadata=metadata,
            )
        info = await self.client.get_collection(name)
        vectors = info.config.params.vectors
        sparse = info.config.params.sparse_vectors
        if (
            not isinstance(vectors, dict) or set(vectors) != {"dense"}
            or vectors["dense"].size != DENSE_SIZE or vectors["dense"].distance != models.Distance.COSINE
            or not sparse or set(sparse) != {"bm25"} or sparse["bm25"].modifier != models.Modifier.IDF
            or info.config.metadata != metadata
        ):
            raise ValueError("Qdrant schema mismatch; reindex into a new versioned collection")
        for field, expected in PAYLOAD_INDEXES.items():
            existing = info.payload_schema.get(f"metadata.{field}")
            if existing and existing.data_type != expected:
                raise ValueError("Qdrant payload index mismatch; reindex into a new versioned collection")
        for field, expected in PAYLOAD_INDEXES.items():
            if f"metadata.{field}" not in info.payload_schema:
                await self.client.create_payload_index(name, f"metadata.{field}", expected, wait=True)

    async def replace_generation(
        self, source_path: str, generation: str, chunks: Sequence[Chunk], *, file_hash: str,
    ) -> list[str]:
        """Caller holds index_lock and commits the manifest after this returns.

        file_hash identifies source bytes. Generation identifies the complete,
        immutable indexing inputs, including metadata, configuration, parser/
        chunker schema and model identity; callers must change it when any input
        changes. Parsing and all dense embeddings finish before any write.
        A failed upsert never triggers old-point deletion.
        """
        if not source_path or not generation or not file_hash:
            raise ValueError("source_path, generation and file_hash are required")
        if any(c.source_path != source_path or c.security_level not in {"public", "private"} for c in chunks):
            raise ValueError("chunks must belong to the source and have a valid security level")
        indexes = [c.chunk_index for c in chunks]
        if len(set(indexes)) != len(indexes) or any(i < 0 for i in indexes):
            raise ValueError("chunk indexes must be unique and non-negative")
        if not chunks:
            await self.delete_source(source_path)
            return []
        embeddings = await self.embedding_provider.embed_documents([c.embedding_text for c in chunks])
        if len(embeddings) != len(chunks):
            raise ValueError("embedding provider did not return every chunk")
        points = []
        for chunk, dense in zip(chunks, embeddings):
            if len(dense) != DENSE_SIZE:
                raise ValueError("embedding vector dimension mismatch")
            point_id = str(uuid5(self.vault_id, json.dumps([source_path, generation, chunk.chunk_index], ensure_ascii=False)))
            payload = asdict(chunk)
            payload.pop("embedding_text")
            document = payload.pop("document")
            payload.update(generation=generation, file_hash=file_hash)
            points.append(models.PointStruct(id=point_id, payload={"document": document, "metadata": payload}, vector={
                "dense": dense,
                "bm25": models.Document(text=document, model="qdrant/bm25", options=BM25_OPTIONS),
            }))
        BATCH_SIZE = 64
        for i in range(0, len(points), BATCH_SIZE):
            await self.client.upsert(self.settings.collection_name, points[i:i + BATCH_SIZE], wait=True)
        await self.client.delete(self.settings.collection_name, models.FilterSelector(filter=models.Filter(
            must=[_match("source_path", source_path)], must_not=[_match("generation", generation)],
        )), wait=True)
        return [str(point.id) for point in points]

    async def delete_source(self, source_path: str) -> None:
        await self.client.delete(self.settings.collection_name, models.FilterSelector(
            filter=models.Filter(must=[_match("source_path", source_path)]),
        ), wait=True)

    async def generation_matches(
        self, source_path: str, generation: str, point_count: int,
        point_ids: Sequence[str] = (),
    ) -> bool:
        """Check generation count and, when available, its exact point IDs."""
        result = await self.client.count(
            self.settings.collection_name,
            count_filter=models.Filter(must=[_match("source_path", source_path), _match("generation", generation)]),
            exact=True,
        )
        if result.count != point_count:
            return False
        if not point_ids:
            return True
        points, _ = await self.client.scroll(
            self.settings.collection_name,
            scroll_filter=models.Filter(must=[_match("source_path", source_path), _match("generation", generation)]),
            limit=max(point_count, 1), with_payload=False, with_vectors=False,
        )
        return {str(point.id) for point in points} == set(point_ids)

    async def cleanup_orphans(self, completed_generations: Mapping[str, str]) -> None:
        """Remove points absent from the completed manifest while holding index_lock."""
        offset = None
        while True:
            points, offset = await self.client.scroll(
                self.settings.collection_name, offset=offset, limit=256,
                with_payload=["metadata.source_path", "metadata.generation"], with_vectors=False,
            )
            orphans = []
            for point in points:
                metadata = (point.payload or {}).get("metadata", {})
                path = metadata.get("source_path")
                if path not in completed_generations or metadata.get("generation") != completed_generations[path]:
                    orphans.append(point.id)
            if orphans:
                await self.client.delete(self.settings.collection_name, models.PointIdsList(points=orphans), wait=True)
            if offset is None:
                break

    async def hybrid_candidates(self, request: SearchRequest) -> list[SearchResult]:
        """Return up to 40 filtered RRF candidates before source limits or reranking."""
        filters = search_filter(request)
        dense = await self.embedding_provider.embed_query(request.query)
        response = await self.client.query_points(
            self.settings.collection_name,
            prefetch=[
                models.Prefetch(query=dense, using="dense", filter=filters, limit=20),
                models.Prefetch(query=models.Document(text=request.query, model="qdrant/bm25", options=BM25_OPTIONS),
                                using="bm25", filter=filters, limit=20),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF), query_filter=filters,
            limit=40, with_payload=True,
        )
        return [SearchResult(
            point_id=str(point.id), document=(point.payload or {})["document"],
            score=point.score, **(point.payload or {})["metadata"],
        ) for point in response.points]

    async def hybrid_search(self, request: SearchRequest) -> list[SearchResult]:
        results = []
        source_counts = Counter()
        for candidate in await self.hybrid_candidates(request):
            source = candidate.source_path
            if source_counts[source] >= 2:
                continue
            source_counts[source] += 1
            results.append(candidate)
            if len(results) >= request.limit:
                break
        return results


def _match(field: str, value: str) -> models.FieldCondition:
    return models.FieldCondition(key=f"metadata.{field}", match=models.MatchValue(value=value))
