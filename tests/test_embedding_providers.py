import asyncio
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from knowledge_mcp import embeddings


def test_factory_selects_supported_provider(monkeypatch):
    bge = object()
    monkeypatch.setattr(embeddings, "LocalSentenceTransformerProvider", lambda name: bge, raising=False)
    assert embeddings.create_embedding_provider("dragonkue/BGE-m3-ko") is bge
    with pytest.raises(ValueError, match="Unsupported dense model"):
        embeddings.create_embedding_provider("another-model")
    with pytest.raises(ValueError, match="Unsupported dense model"):
        embeddings.create_embedding_provider("intfloat/multilingual-e5-large")


def test_e5_tokenizer_adapter_loads_fastembed_model():
    calls = []
    raw_tokenizer = SimpleNamespace(
        encode=lambda text: SimpleNamespace(ids=[2, 3]),
        decode=lambda ids: "decoded",
    )
    model = SimpleNamespace(tokenizer=None)
    model.load_onnx_model = lambda: (calls.append(True), setattr(model, "tokenizer", raw_tokenizer))
    provider = embeddings.LocalFastEmbedProvider.__new__(embeddings.LocalFastEmbedProvider)
    provider.embedding_model = SimpleNamespace(model=model)
    tokenizer = provider.get_tokenizer()
    assert calls == [True]
    assert tokenizer.encode("text") == [2, 3]
    assert tokenizer.decode([2, 3]) == "decoded"


def test_bge_embeds_query_and_documents_with_normalization(monkeypatch):
    calls = []
    raw_tokenizer = SimpleNamespace(
        encode=lambda text, **kwargs: [11, 12],
        decode=lambda ids, **kwargs: "decoded",
    )

    class FakeSentenceTransformer:
        def __init__(self, name):
            assert name == "dragonkue/BGE-m3-ko"
            self.tokenizer = raw_tokenizer

        def encode(self, texts, *, normalize_embeddings):
            calls.append((texts, normalize_embeddings))
            return np.full((len(texts), 1024), 1 / 32, dtype=np.float32)

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=FakeSentenceTransformer))
    provider = embeddings.LocalSentenceTransformerProvider("dragonkue/BGE-m3-ko")
    assert provider.get_vector_size() == 1024
    assert provider.get_vector_name() != "fast-multilingual-e5-large"
    assert len(asyncio.run(provider.embed_query("query"))) == 1024
    assert len(asyncio.run(provider.embed_documents(["doc1", "doc2"]))) == 2
    assert calls == [(["query"], True), (["doc1", "doc2"], True)]
    tokenizer = provider.get_tokenizer()
    assert tokenizer.encode("text") == [11, 12]
    assert tokenizer.decode([11, 12]) == "decoded"
