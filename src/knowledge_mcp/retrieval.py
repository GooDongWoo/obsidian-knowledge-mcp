"""Request-time model selection and optional reranking."""

from collections import Counter
from typing import Mapping

from .qdrant_store import KnowledgeStore
from .reranker import LocalReranker
from .search import SearchRequest, SearchResult


class MultiModelStore:
    def __init__(self, stores: Mapping[str, KnowledgeStore], reranker: LocalReranker | None = None):
        self.stores = dict(stores)
        self.reranker = reranker or LocalReranker()

    async def hybrid_search(self, request: SearchRequest) -> list[SearchResult]:
        candidates = await self.stores[request.embedding_model].hybrid_candidates(request)
        if request.rerank:
            candidates = await self.reranker.rerank(request.query, candidates[:40])
        counts = Counter()
        results = []
        for candidate in candidates:
            if counts[candidate.source_path] >= 2:
                continue
            counts[candidate.source_path] += 1
            results.append(candidate)
            if len(results) >= request.limit:
                break
        return results
