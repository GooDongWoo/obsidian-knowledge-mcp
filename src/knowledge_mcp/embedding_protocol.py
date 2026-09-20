from abc import ABC, abstractmethod
from typing import Protocol


class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...

    def decode(self, tokens: list[int]) -> str: ...


class EmbeddingProvider(ABC):
    """Minimal provider contract used by the local Qdrant store."""

    @abstractmethod
    async def embed_documents(self, documents: list[str]) -> list[list[float]]:
        raise NotImplementedError

    @abstractmethod
    async def embed_query(self, query: str) -> list[float]:
        raise NotImplementedError

    @abstractmethod
    def get_vector_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_vector_size(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_tokenizer(self) -> Tokenizer:
        raise NotImplementedError
