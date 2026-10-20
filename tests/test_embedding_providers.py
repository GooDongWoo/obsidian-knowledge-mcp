import asyncio
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from knowledge_mcp import embeddings


@pytest.fixture(autouse=True)
def cpu_torch_device(monkeypatch):
    # Each CUDA-specific test overrides this; never depend on the host GPU.
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)


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
        def __init__(self, name, *, device=None, model_kwargs=None):
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
        def __init__(self, name, *, device=None, model_kwargs=None):
            pass
        def encode(self, texts, *, batch_size, normalize_embeddings):
            calls.append((batch_size, normalize_embeddings))
            return np.full((len(texts), 1024), 1 / 32, dtype=np.float32)
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


@pytest.mark.parametrize("cuda,policy,device,dtype", [
    (True, "auto", "cuda", "bfloat16"),
    (False, "auto", "cpu", "float32"),
    (True, "float32", "cuda", "float32"),
    (False, "float32", "cpu", "float32"),
])
@pytest.mark.parametrize("first_access", ["query", "documents", "tokenizer"])
def test_torch_embedding_policy_applies_to_every_lazy_entry(
    tmp_path, monkeypatch, cuda, policy, device, dtype, first_access
):
    from contextlib import nullcontext
    from dataclasses import replace
    import torch
    from knowledge_mcp.config import Settings

    constructors, encodes, probes = [], [], []
    tokenizer = SimpleNamespace(
        encode=lambda text, **kwargs: [11] if kwargs == {"add_special_tokens": False} else [],
        decode=lambda tokens, **kwargs: "decoded" if kwargs == {"skip_special_tokens": True} else "",
    )
    class Model:
        def __init__(self, name, *, device, model_kwargs):
            constructors.append((name, device, model_kwargs))
            self.tokenizer = tokenizer
        def encode(self, texts, *, batch_size, normalize_embeddings):
            encodes.append((texts, batch_size, normalize_embeddings))
            values = np.zeros((len(texts), 1024), dtype=np.float64)
            values[:, 0] = [1.0 if text == "first" else -1.0 for text in texts]
            return values

    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.cuda, "device", lambda selected: nullcontext())
    def supported(including_emulation=True):
        probes.append(including_emulation)
        return True
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", supported)
    monkeypatch.setattr(embeddings, "configure_torch_resources", lambda *_: None)
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=Model))
    settings = replace(Settings.from_paths(vault_root=tmp_path / "vault", project_root=tmp_path / "project"),
                       inference_precision=policy)
    provider = embeddings.LocalSentenceTransformerProvider(settings.dense_model, settings=settings)
    assert (provider.precision_policy, provider.device, provider.effective_dtype) == (policy, None, None)
    if first_access == "query":
        assert asyncio.run(provider.embed_query("first"))[0] == 1.0
    elif first_access == "documents":
        assert [row[0] for row in asyncio.run(provider.embed_documents(["first", "second"]))] == [1.0, -1.0]
    else:
        adapter = provider.get_tokenizer()
        assert adapter.encode("text") == [11]
        assert adapter.decode([11]) == "decoded"
    vectors = asyncio.run(provider.embed_documents(["first", "second"]))
    assert provider._encode(["first"]).dtype == np.float32
    assert all(type(value) is float for row in vectors for value in row)
    assert (provider.device, provider.effective_dtype) == (device, dtype)
    assert constructors == [(settings.dense_model, device, {"torch_dtype": dtype})]
    assert probes == ([False] if cuda and policy == "auto" else [])
    assert all(batch == 8 and normalize is True for _, batch, normalize in encodes)


def test_embedding_unsupported_bf16_fails_before_model_construction(monkeypatch):
    from contextlib import nullcontext
    import torch
    constructors = []
    class Model:
        def __init__(self, name, *, device=None, model_kwargs=None):
            constructors.append(name)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda including_emulation=True: False)
    monkeypatch.setattr(embeddings, "configure_torch_resources", lambda *_: None)
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=Model))
    provider = embeddings.LocalSentenceTransformerProvider("test-model")
    with pytest.raises(RuntimeError, match="bfloat16 support"):
        provider.warmup()
    assert constructors == []
    assert provider._model is None
    assert provider.device is None and provider.effective_dtype is None


def test_empty_embedding_documents_do_not_load_model(monkeypatch):
    def forbidden(*_):
        pytest.fail("empty documents loaded model")
    monkeypatch.setattr(embeddings, "configure_torch_resources", forbidden)
    provider = embeddings.LocalSentenceTransformerProvider("test-model")
    assert asyncio.run(provider.embed_documents([])) == []
    assert provider._model is None


@pytest.mark.parametrize("bad_vectors", [
    np.zeros((1, 1024)),
    np.full((1, 1024), np.nan),
    np.full((1, 1024), np.inf),
    np.ones((1, 1023)),
    np.ones((2, 1024)),
    np.ones(1024),
])
@pytest.mark.parametrize("method,argument", [("embed_query", "query"), ("embed_documents", ["doc"])])
def test_embedding_rejects_invalid_vectors(bad_vectors, method, argument):
    provider = embeddings.LocalSentenceTransformerProvider("test-model")
    provider._model = SimpleNamespace(encode=lambda *args, **kwargs: bad_vectors)
    with pytest.raises(ValueError, match="embedding"):
        asyncio.run(getattr(provider, method)(argument))


def test_embedding_float32_numpy_boundary_after_small_bf16_tensor():
    import torch
    provider = embeddings.LocalSentenceTransformerProvider("test-model")
    tensor = torch.zeros((1, 1024), dtype=torch.bfloat16)
    tensor[0, 0] = 1
    # SentenceTransformer.encode converts bf16 tensors to float32 before NumPy.
    provider._model = SimpleNamespace(encode=lambda *args, **kwargs: tensor.float().numpy())
    vectors = provider._encode(["doc"])
    assert vectors.dtype == np.float32
    assert vectors.tolist() == [[1.0] + [0.0] * 1023]
