"""Lazy local Korean cross-encoder scoring."""

import asyncio
import math
import threading

from .config import Settings
from .embeddings import _MODEL_INFERENCE_LOCK, configure_torch_resources
from .search import SearchResult
from .state import run_blocking


class RerankError(RuntimeError):
    """Stable optional-ranking failure; exception payloads stay private."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


class LocalReranker:
    model_name = "dragonkue/bge-reranker-v2-m3-ko"

    def __init__(self, *, settings: Settings | None = None):
        self.batch_size = settings.reranker_batch_size if settings else 8
        self.cpu_threads = settings.cpu_threads if settings else 4
        self.cuda_memory_fraction = settings.cuda_memory_fraction if settings else None
        self._model = None
        self._lock = asyncio.Lock()
        self._model_lock = threading.Lock()
        self.device: str | None = None
        self._initialization_error: str | None = None
        self._device_fallback: str | None = None
        self._last_fallback: str | None = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def status(self) -> dict:
        return {"model": self.model_name, "loaded": self.loaded,
                "device": self.device, "last_fallback": self._last_fallback}

    def warmup(self) -> None:
        """Preload the cross-encoder model."""
        with self._model_lock:
            if self._initialization_error:
                raise RerankError(self._initialization_error)
            if self._model is None:
                try:
                    self._initialize()
                except Exception:
                    self._initialization_error = "reranker_init_failed"
                    self._last_fallback = self._initialization_error
                    raise RerankError(self._initialization_error) from None

    def _initialize(self) -> None:
        configure_torch_resources(self.cpu_threads, self.cuda_memory_fraction)
        import torch
        from sentence_transformers import CrossEncoder

        if torch.cuda.is_available():
            try:
                model = CrossEncoder(self.model_name, device="cuda")
            except Exception:
                self._device_fallback = "reranker_cuda_init_failed"
            else:
                self.device = "cuda"
                self._model = model
                return
        else:
            self._device_fallback = "reranker_cuda_unavailable"
        model = CrossEncoder(self.model_name, device="cpu")
        self.device = "cpu"
        self._model = model
        self._last_fallback = self._device_fallback

    async def rerank(self, query: str, candidates: list[SearchResult]) -> list[SearchResult]:
        if not candidates:
            return []
        async with self._lock:
            try:
                scores = await run_blocking(self._score, query, candidates)
            except RerankError:
                raise
            except Exception:
                self._last_fallback = "reranker_inference_failed"
                raise RerankError(self._last_fallback) from None
            try:
                if len(scores) != len(candidates):
                    raise ValueError
                scores = [float(score) for score in scores]
                if any(not math.isfinite(score) for score in scores):
                    raise ValueError
            except (TypeError, ValueError, OverflowError):
                self._last_fallback = "reranker_invalid_scores"
                raise RerankError(self._last_fallback) from None
            ranked = [candidate.model_copy(update={"score": score}) for candidate, score in zip(candidates, scores)]
            self._last_fallback = self._device_fallback
            return sorted(ranked, key=lambda candidate: candidate.score, reverse=True)

    def _score(self, query: str, candidates: list[SearchResult]):
        with _MODEL_INFERENCE_LOCK:
            self.warmup()
            return self._model.predict([(query, candidate.document) for candidate in candidates], batch_size=self.batch_size)
