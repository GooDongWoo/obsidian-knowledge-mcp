"""Local embedding providers for the supported dense models."""

import asyncio
import os
import warnings
from typing import Sequence

import onnxruntime as ort
from fastembed import TextEmbedding
from fastembed.common.types import Device

from .config import DENSE_MODELS
from .embedding_protocol import EmbeddingProvider, Tokenizer


def _limit_cpu_threads() -> None:
    """Cap PyTorch CPU thread usage to avoid CPU spikes during inference."""
    try:
        import torch

        max_threads = min(4, os.cpu_count() or 4)
        torch.set_num_threads(max_threads)
    except Exception:
        pass


class _FastEmbedTokenizer:
    def __init__(self, tokenizer):
        self._tokenizer = tokenizer

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text).ids

    def decode(self, tokens: list[int]) -> str:
        return self._tokenizer.decode(tokens)


class _SentenceTransformerTokenizer:
    def __init__(self, tokenizer):
        self._tokenizer = tokenizer

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text, add_special_tokens=False)

    def decode(self, tokens: list[int]) -> str:
        return self._tokenizer.decode(tokens, skip_special_tokens=True)


def prepare_cuda_runtime() -> None:
    """Load CUDA/cuDNN DLLs shipped by NVIDIA Python packages when available."""
    _limit_cpu_threads()
    preload_dlls = getattr(ort, "preload_dlls", None)
    if preload_dlls is None:
        return
    try:
        preload_dlls()
    except (ImportError, OSError) as exc:
        warnings.warn(
            f"CUDA DLL preload failed; provider discovery will fall back: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )


def select_device(available_providers: Sequence[str] | None = None) -> Device:
    """Use CUDA only when ONNX Runtime exposes its CUDA execution provider."""
    if available_providers is None:
        prepare_cuda_runtime()
    providers = available_providers if available_providers is not None else ort.get_available_providers()
    return Device.CUDA if "CUDAExecutionProvider" in providers else Device.CPU


class LocalFastEmbedProvider(EmbeddingProvider):
    """FastEmbed provider that prefers CUDA and safely falls back to CPU."""

    def __init__(self, model_name: str):
        self.device = select_device()
        try:
            self.embedding_model = TextEmbedding(model_name, cuda=self.device)
        except Exception:
            if self.device is not Device.CUDA:
                raise
            warnings.warn(
                "CUDAExecutionProvider could not initialize; falling back to CPU.",
                RuntimeWarning,
                stacklevel=2,
            )
            self.device = Device.CPU
            self.embedding_model = TextEmbedding(model_name, cuda=Device.CPU)

    async def embed_documents(self, documents: list[str]) -> list[list[float]]:
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None, lambda: list(self.embedding_model.passage_embed(documents, batch_size=32))
        )
        return [embedding.tolist() for embedding in embeddings]

    async def embed_query(self, query: str) -> list[float]:
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None, lambda: list(self.embedding_model.query_embed([query]))
        )
        return embeddings[0].tolist()

    def get_vector_name(self) -> str:
        return f"fast-{self.embedding_model.model_name.split('/')[-1].lower()}"

    def get_vector_size(self) -> int:
        return self.embedding_model.embedding_size

    def get_tokenizer(self) -> Tokenizer:
        model = self.embedding_model.model
        if getattr(model, "tokenizer", None) is None and hasattr(model, "load_onnx_model"):
            model.load_onnx_model()
        return _FastEmbedTokenizer(model.tokenizer)


class LocalSentenceTransformerProvider(EmbeddingProvider):
    """Korean BGE provider; load model only when first used."""

    def __init__(self, model_name: str):
        _limit_cpu_threads()
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def warmup(self) -> None:
        """Preload the embedding model."""
        _ = self.model

    async def embed_documents(self, documents: list[str]) -> list[list[float]]:
        try:
            vectors = await asyncio.to_thread(
                self.model.encode, documents, batch_size=32, normalize_embeddings=True
            )
        except TypeError:
            vectors = await asyncio.to_thread(
                self.model.encode, documents, normalize_embeddings=True
            )
        return vectors.tolist()

    async def embed_query(self, query: str) -> list[float]:
        try:
            vectors = await asyncio.to_thread(
                self.model.encode, [query], batch_size=32, normalize_embeddings=True
            )
        except TypeError:
            vectors = await asyncio.to_thread(
                self.model.encode, [query], normalize_embeddings=True
            )
        return vectors[0].tolist()


    def get_vector_name(self) -> str:
        return f"st-{self.model_name.split('/')[-1].lower()}"

    def get_vector_size(self) -> int:
        return 1024

    def get_tokenizer(self) -> Tokenizer:
        return _SentenceTransformerTokenizer(self.model.tokenizer)


def create_embedding_provider(model_name: str) -> EmbeddingProvider:
    if model_name == "dragonkue/BGE-m3-ko" or model_name in DENSE_MODELS:
        return LocalSentenceTransformerProvider(model_name)
    raise ValueError(f"Unsupported dense model: {model_name}")
