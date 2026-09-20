"""Read-only FastMCP application for the local knowledge collection."""

import time
from dataclasses import dataclass
from typing import Any, Literal

from fastmcp import FastMCP

from .config import Settings
from .search import SearchRequest, SearchResult
from .state import OperationLog


@dataclass(slots=True)
class KnowledgeApplication:
    settings: Settings
    store: Any
    operation_log: OperationLog
    mcp: FastMCP

    async def find(
        self,
        *,
        query: str,
        document_type: list[str] | None = None,
        file_type: list[str] | None = None,
        created_from: Any = None,
        created_to: Any = None,
        modified_from: Any = None,
        modified_to: Any = None,
        include_private: bool = False,
        limit: int = 8,
        embedding_model: Literal["dragonkue/BGE-m3-ko"] | None = None,
        rerank: bool = False,
    ) -> list[dict[str, Any]]:
        request = SearchRequest(
            query=query,
            document_type=document_type,
            file_type=file_type,
            created_from=created_from,
            created_to=created_to,
            modified_from=modified_from,
            modified_to=modified_to,
            include_private=include_private,
            limit=limit,
            embedding_model=self.settings.dense_model if embedding_model is None else embedding_model,
            rerank=rerank,
        )
        started = time.perf_counter()
        results: list[SearchResult] = await self.store.hybrid_search(request)
        self.operation_log.record_query(
            query=request.query,
            filters=request.model_dump(exclude={"query"}),
            results=[result.model_dump() for result in results],
            client_name=self.settings.client_name,
            elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
        )
        return [result.model_dump() for result in results]

    async def status(self) -> dict[str, Any]:
        result = dict(self.operation_log.index_status())
        result.update({"collection": self.settings.collection_name, "dense_model": self.settings.dense_model})
        if hasattr(self.store, "stores"):
            models = {}
            for model_name, store in self.store.stores.items():
                collection = store.settings.collection_name
                details = {"collection": collection, "point_count": None}
                details.update(self.operation_log.index_status(collection))
                device = getattr(store.embedding_provider, "device", None)
                if device is not None:
                    details["embedding_device"] = getattr(device, "value", str(device))
                try:
                    details["point_count"] = (await store.client.count(collection, exact=True)).count
                except Exception:
                    pass
                models[model_name] = details
            result["models"] = models
            model_statuses = [details.get("status") for details in models.values() if details.get("status")]
            if model_statuses:
                result["status"] = "completed" if all(status == "completed" for status in model_statuses) else "partial"
                result["error_code"] = next(
                    (details.get("error_code") for details in models.values() if details.get("error_code")), None
                )
            default_model = models.get(self.settings.dense_model)
            if default_model is not None:
                result.update({"collection": default_model["collection"],
                               "point_count": default_model["point_count"]})
            reranker = self.store.reranker
            result["reranker"] = {"model": reranker.model_name, "loaded": reranker.loaded}
            return result
        provider = getattr(self.store, "embedding_provider", None)
        device = getattr(provider, "device", None)
        if device is not None:
            result["embedding_device"] = getattr(device, "value", str(device))
        if hasattr(self.store, "client"):
            try:
                count = await self.store.client.count(self.settings.collection_name, exact=True)
                result["point_count"] = count.count
            except Exception:
                result["point_count"] = None
        return result


def create_application(settings: Settings, store: Any, operation_log: OperationLog) -> KnowledgeApplication:
    mcp = FastMCP("obsidian-knowledge")
    application = KnowledgeApplication(settings, store, operation_log, mcp)

    @mcp.tool(
        name="qdrant-find",
        description=("Search the local Obsidian Vault with dense plus BM25 hybrid retrieval. "
                     "Scores are RRF fusion scores, or raw cross-encoder relevance scores when rerank is true."),
    )
    async def qdrant_find(
        query: str,
        document_type: list[str] | None = None,
        file_type: list[str] | None = None,
        created_from: str | None = None,
        created_to: str | None = None,
        modified_from: str | None = None,
        modified_to: str | None = None,
        include_private: bool = False,
        limit: int = 8,
        embedding_model: Literal["dragonkue/BGE-m3-ko"] | None = None,
        rerank: bool = False,
    ) -> list[dict[str, Any]]:
        return await application.find(
            query=query,
            document_type=document_type,
            file_type=file_type,
            created_from=created_from,
            created_to=created_to,
            modified_from=modified_from,
            modified_to=modified_to,
            include_private=include_private,
            limit=limit,
            embedding_model=embedding_model,
            rerank=rerank,
        )

    @mcp.tool(name="knowledge-index-status", description="Show the local Vault index status.")
    async def knowledge_index_status() -> dict[str, Any]:
        return await application.status()

    return application
