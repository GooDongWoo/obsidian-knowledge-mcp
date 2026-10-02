"""Read-only FastMCP application for the local knowledge collection."""

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.tools import ToolResult
from mcp.types import TextContent

from .config import Settings
from .state import OperationLog, run_blocking


@dataclass(slots=True)
class KnowledgeApplication:
    settings: Settings
    store: Any
    operation_log: OperationLog | None
    mcp: FastMCP
    runtime_status: dict[str, Any] | None = None
    indexers: dict[str, Any] = field(default_factory=dict)
    index_snapshot: dict[str, Any] = field(default_factory=dict)

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
        rerank: bool = True,
    ) -> list[dict[str, Any]]:
        from .search import SearchRequest, SearchResult

        if self.runtime_status is not None:
            state = self.runtime_status["state"]
            if state == "indexing":
                raise ToolError("Index is indexing; retry shortly. 인덱싱 중, 잠시 후 재시도하세요.")
            if self.store is None:
                if state == "error":
                    raise ToolError("Initialization failed; inspect knowledge-index-status.")
                raise ToolError("Server is starting; retry shortly.")

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
        try:
            results: list[SearchResult] = await self.store.hybrid_search(request)
        except Exception:
            await run_blocking(self.operation_log.record_query,
                elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
                rerank_requested=request.rerank, error_code="search_failed",
            )
            raise
        await run_blocking(self.operation_log.record_query,
            results=results,
            rerank_requested=request.rerank,
            rerank_applied=bool(results) and all(result.rerank_applied for result in results),
            error_code=next((result.rerank_error for result in results if result.rerank_error), None),
            elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
        )
        return [result.model_dump() for result in results]

    async def status(self) -> dict[str, Any]:
        if self.runtime_status is not None:
            # Status must remain available even while SQLite/model setup or a
            # Qdrant request is blocked. The writer publishes completed results.
            return {
                "collection": self.settings.collection_name, "dense_model": self.settings.dense_model,
                **self.index_snapshot,
                **self.runtime_status,
                **self.reranker_status(),
                "progress": {model: dict(getattr(indexer, "progress", {}))
                             for model, indexer in self.indexers.items()},
            }
        return await self.read_index_status()

    def reranker_status(self) -> dict[str, Any]:
        reranker = getattr(self.store, "reranker", None)
        if reranker is None:
            return {}
        return {"reranker": getattr(reranker, "status", {
            "model": reranker.model_name, "loaded": reranker.loaded,
            "device": getattr(reranker, "device", None), "last_fallback": None,
        })}

    async def read_index_status(self) -> dict[str, Any]:
        result = dict(await run_blocking(self.operation_log.index_status))
        result.update({"collection": self.settings.collection_name, "dense_model": self.settings.dense_model})
        if hasattr(self.store, "stores"):
            models = {}
            for model_name, store in self.store.stores.items():
                collection = store.settings.collection_name
                details = {"collection": collection, "point_count": None}
                details.update(await run_blocking(self.operation_log.index_status, collection))
                device = getattr(store.embedding_provider, "device", None)
                if device is not None:
                    details["embedding_device"] = getattr(device, "value", str(device))
                try:
                    async with asyncio.timeout(.25):
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
            result.update(self.reranker_status())
            return result
        provider = getattr(self.store, "embedding_provider", None)
        device = getattr(provider, "device", None)
        if device is not None:
            result["embedding_device"] = getattr(device, "value", str(device))
        if hasattr(self.store, "client"):
            try:
                async with asyncio.timeout(.25):
                    count = await self.store.client.count(self.settings.collection_name, exact=True)
                result["point_count"] = count.count
            except Exception:
                result["point_count"] = None
        return result


def create_application(settings: Settings, store: Any = None, operation_log: OperationLog | None = None,
                       *, lifespan=None) -> KnowledgeApplication:
    mcp = FastMCP("obsidian-knowledge", lifespan=lifespan)
    application = KnowledgeApplication(settings, store, operation_log, mcp)

    @mcp.tool(
        name="qdrant-find",
        description=("Search the local Obsidian Vault with dense plus BM25 hybrid retrieval. "
                     "Reranking defaults to true. Scores are cross-encoder relevance scores when "
                     "rerank_applied is true; otherwise RRF fusion scores. "
                     "Results report rerank_requested, rerank_applied and stable rerank_error."),
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
        rerank: bool = True,
    ) -> list[dict[str, Any]]:
        results = await application.find(
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
        # Keep FastMCP 2's per-result text contract while exposing modern
        # structured output using the inferred list schema's result envelope.
        return ToolResult(
            content=[TextContent(type="text", text=json.dumps(item, ensure_ascii=False, indent=2))
                     for item in results],
            structured_content={"result": results},
        )

    @mcp.tool(name="knowledge-index-status", description="Show the local Vault index status.")
    async def knowledge_index_status() -> dict[str, Any]:
        return await application.status()

    return application
