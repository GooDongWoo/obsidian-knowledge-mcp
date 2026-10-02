"""Request-time model selection and optional reranking."""

from collections import Counter
from typing import Mapping

from .qdrant_store import KnowledgeStore
from .reranker import LocalReranker, RerankError
from .search import SearchRequest, SearchResult


class MultiModelStore:
    def __init__(self, stores: Mapping[str, KnowledgeStore], reranker: LocalReranker | None = None):
        self.stores = dict(stores)
        self.reranker = reranker or LocalReranker()

    async def hybrid_search(self, request: SearchRequest) -> list[SearchResult]:
        candidates = await self.stores[request.embedding_model].hybrid_candidates(request)
        applied, error_code = False, None
        if request.rerank and candidates:
            try:
                candidates = await self.reranker.rerank(request.query, candidates[:40])
                applied = True
            except RerankError as error:
                error_code = error.code
        counts = Counter()
        results = []
        for candidate in candidates:
            if counts[candidate.source_path] >= 2:
                continue
            counts[candidate.source_path] += 1
            results.append(candidate.model_copy(update={
                "rerank_requested": request.rerank,
                "rerank_applied": applied,
                "rerank_error": error_code,
            }))
            if len(results) >= request.limit:
                break
        return results
