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

        def encode(self, texts, *, batch_size, normalize_embeddings):
            assert batch_size == 8
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


@pytest.mark.parametrize("batch,expected", [(None, 8), (3, 3)])
@pytest.mark.anyio
async def test_embedding_batch_reaches_actual_encode(tmp_path, monkeypatch, batch, expected):
    from dataclasses import replace
    from knowledge_mcp.config import Settings
    calls = []
    class Model:
        def __init__(self, name):
            pass
        def encode(self, texts, *, batch_size, normalize_embeddings):
            calls.append((batch_size, normalize_embeddings))
            return np.zeros((len(texts), 1024))
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=Model))
    settings = Settings.from_paths(vault_root=tmp_path / "vault", project_root=tmp_path / "project")
    if batch is not None:
        settings = replace(settings, embedding_batch_size=batch)
    provider = embeddings.create_embedding_provider(settings.dense_model, settings=settings)
    assert provider._model is None
    assert len(await provider.embed_documents(["synthetic"] * 10)) == 10
    assert len(await provider.embed_query("synthetic")) == 1024
    assert calls == [(expected, True), (expected, True)]


@pytest.mark.parametrize("cuda", [True, False])
@pytest.mark.parametrize("strategy", ["kSameAsRequested", "kNextPowerOfTwo"])
def test_fastembed_options_reach_real_inference_session(tmp_path, monkeypatch, cuda, strategy):
    """Keep real TextEmbedding -> OnnxTextEmbedding -> OnnxModel forwarding."""
    from dataclasses import replace
    from fastembed.text.onnx_embedding import OnnxTextEmbedding
    from knowledge_mcp.config import Settings
    captures = []
    def session(path, *, providers, sess_options):
        captures.append((providers, sess_options.intra_op_num_threads))
        return SimpleNamespace(get_providers=lambda: [p if isinstance(p, str) else p[0] for p in providers])
    monkeypatch.setattr(embeddings.ort, "get_available_providers", lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"] if cuda else ["CPUExecutionProvider"])
    monkeypatch.setattr(embeddings, "prepare_cuda_runtime", lambda: None)
    monkeypatch.setattr(embeddings.ort, "InferenceSession", session)
    monkeypatch.setattr(OnnxTextEmbedding, "download_model", lambda *_args, **_kwargs: tmp_path)
    monkeypatch.setattr("fastembed.text.onnx_text_model.load_tokenizer", lambda **_: (object(), {}))
    monkeypatch.setattr(embeddings.os, "cpu_count", lambda: 8)
    settings = replace(Settings.from_paths(vault_root=tmp_path / "vault", project_root=tmp_path / "project"),
                       onnx_gpu_mem_limit=268435456, onnx_intra_op_num_threads=3, onnx_arena_extend_strategy=strategy)
    provider = embeddings.LocalFastEmbedProvider("BAAI/bge-small-en-v1.5", settings=settings)
    expected = [("CUDAExecutionProvider", {"gpu_mem_limit": 268435456, "arena_extend_strategy": strategy}), "CPUExecutionProvider"] if cuda else ["CPUExecutionProvider"]
    assert captures == [(expected, 3)]
    assert provider.get_vector_size() == 384
