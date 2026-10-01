"""Lazy local Korean cross-encoder scoring."""

import asyncio
import math
import threading

from .search import SearchResult
from .state import run_blocking


class LocalReranker:
    model_name = "dragonkue/bge-reranker-v2-m3-ko"

    def __init__(self):
        self._model = None
        self._lock = asyncio.Lock()
        self._model_lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def warmup(self) -> None:
        """Preload the cross-encoder model."""
        with self._model_lock:
            if self._model is None:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.model_name, device="cuda")

    async def rerank(self, query: str, candidates: list[SearchResult]) -> list[SearchResult]:
        if not candidates:
            return []
        async with self._lock:
            scores = await run_blocking(self._score, query, candidates)
        if len(scores) != len(candidates):
            raise ValueError("reranker returned the wrong number of scores")
        ranked = [candidate.model_copy(update={"score": float(score)}) for candidate, score in zip(candidates, scores)]
        if any(not math.isfinite(candidate.score) for candidate in ranked):
            raise ValueError("reranker returned a non-finite score")
        return sorted(ranked, key=lambda candidate: candidate.score, reverse=True)

    def _score(self, query: str, candidates: list[SearchResult]):
        self.warmup()
        return self._model.predict([(query, candidate.document) for candidate in candidates])
