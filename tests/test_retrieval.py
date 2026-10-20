import pytest
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import torch

from knowledge_mcp.search import SearchRequest, SearchResult


@pytest.fixture(autouse=True)
def isolated_cuda_precision(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("torch.cuda.device", lambda _: nullcontext())
    monkeypatch.setattr("torch.cuda.is_bf16_supported", lambda including_emulation=True: True)


@pytest.mark.parametrize("cuda,policy,dtype", [
    (True, "auto", "bfloat16"), (False, "auto", "float32"),
    (True, "float32", "float32"), (False, "float32", "float32"),
])
@pytest.mark.parametrize("constructor", ["model_kwargs", "automodel_args"])
def test_reranker_precision_policy_and_constructor_compatibility(tmp_path, monkeypatch, cuda, policy, dtype, constructor):
    from dataclasses import replace
    from knowledge_mcp.config import Settings
    from knowledge_mcp.reranker import LocalReranker

    calls, probes = [], []
    def supported(including_emulation=True):
        probes.append(including_emulation)
        return True
    class ModernCrossEncoder:
        def __init__(self, name, *, device=None, model_kwargs=None):
            calls.append((device, model_kwargs))
    class LegacyCrossEncoder:
        def __init__(self, name, *, device=None, automodel_args=None):
            calls.append((device, automodel_args))
    model = ModernCrossEncoder if constructor == "model_kwargs" else LegacyCrossEncoder
    monkeypatch.setattr("torch.cuda.is_available", lambda: cuda)
    monkeypatch.setattr("torch.cuda.is_bf16_supported", supported)
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(CrossEncoder=model))
    settings = replace(Settings.from_paths(vault_root=tmp_path / "vault", project_root=tmp_path), inference_precision=policy)
    reranker = LocalReranker(settings=settings)
    assert reranker.status["precision_policy"] == policy
    assert reranker.status["effective_dtype"] is None
    assert reranker.status["device"] is None
    reranker.warmup()
    device = "cuda" if cuda else "cpu"
    assert calls == [(device, {"torch_dtype": dtype})]
    assert probes == ([False] if cuda and policy == "auto" else [])
    assert reranker.status["device"] == device
    assert reranker.status["effective_dtype"] == dtype


def test_reranker_unsupported_bf16_falls_back_before_cuda_construction(monkeypatch):
    from knowledge_mcp.reranker import LocalReranker
    calls = []
    class CrossEncoder:
        def __init__(self, name, *, device=None, model_kwargs=None):
            calls.append((device, model_kwargs))
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.is_bf16_supported", lambda including_emulation=True: False)
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(CrossEncoder=CrossEncoder))
    reranker = LocalReranker()
    reranker.warmup()
    assert calls == [("cpu", {"torch_dtype": "float32"})]
    assert reranker.status["device"] == "cpu"
    assert reranker.status["effective_dtype"] == "float32"
    assert reranker.status["last_fallback"] == "reranker_cuda_init_failed"


@pytest.mark.anyio
async def test_reranker_bf16_tensor_scores_convert_inside_worker_and_keep_ties(monkeypatch):
    import threading
    from knowledge_mcp.reranker import LocalReranker
    owner = threading.get_ident()
    class WorkerTensor(torch.Tensor):
        def float(self, *args, **kwargs):
            assert threading.get_ident() != owner
            return super().float(*args, **kwargs)
        def numpy(self, *args, **kwargs):
            assert threading.get_ident() != owner
            assert self.dtype is torch.float32
            return super().numpy(*args, **kwargs)
    class CrossEncoder:
        def __init__(self, name, *, device=None, model_kwargs=None):
            pass
        def predict(self, pairs, *, batch_size, convert_to_tensor=False):
            assert convert_to_tensor is True
            assert threading.get_ident() != owner
            return torch.tensor([.5, .5], dtype=torch.bfloat16).as_subclass(WorkerTensor)
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(CrossEncoder=CrossEncoder))
    reranker = LocalReranker()
    found = await reranker.rerank("q", [result("a", "one", .9), result("b", "two", .1)])
    assert [(item.point_id, item.score) for item in found] == [("a", .5), ("b", .5)]


@pytest.mark.anyio
async def test_tensor_reranking_scores_only_first_40_candidates(monkeypatch):
    from knowledge_mcp.retrieval import MultiModelStore
    class CrossEncoder:
        def __init__(self, name, *, device, model_kwargs):
            pass
        def predict(self, pairs, *, batch_size, convert_to_tensor):
            assert convert_to_tensor is True
            assert [document for _, document in pairs] == [str(i) for i in range(40)]
            return torch.arange(40, dtype=torch.bfloat16)
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(CrossEncoder=CrossEncoder))
    candidates = [result(str(i), str(i), 1 / (i + 1)) for i in range(41)]
    store = MultiModelStore({"dragonkue/BGE-m3-ko": CandidateStore(candidates)})
    found = await store.hybrid_search(SearchRequest(query="q", limit=3))
    assert [item.point_id for item in found] == ["39", "38", "37"]
    assert all(item.rerank_applied for item in found)


@pytest.mark.anyio
async def test_reranker_conversion_failure_is_invalid_scores(monkeypatch):
    from knowledge_mcp.reranker import LocalReranker, RerankError
    class BrokenTensor:
        def __len__(self):
            return 1
        def __iter__(self):
            return iter([.5])
        def detach(self):
            raise RuntimeError("private conversion payload")
    class CrossEncoder:
        def __init__(self, name, *, device=None, model_kwargs=None):
            pass
        def predict(self, pairs, **kwargs):
            return BrokenTensor()
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(CrossEncoder=CrossEncoder))
    reranker = LocalReranker()
    with pytest.raises(RerankError, match="reranker_invalid_scores"):
        await reranker.rerank("q", [result("a", "one", .9)])


def result(point_id, source, score):
    return SearchResult(
        point_id=point_id, document=point_id, source_path=source,
        document_type="other", file_type="md", security_level="public",
        created_at="2025-01-01T00:00:00+09:00", modified_at="2025-01-01T00:00:00+09:00",
        score=score,
    )


class CandidateStore:
    def __init__(self, candidates):
        self.candidates = candidates
        self.requests = []

    async def hybrid_candidates(self, request):
        self.requests.append(request)
        return self.candidates


class FakeReranker:
    loaded = True
    model_name = "dragonkue/bge-reranker-v2-m3-ko"

    async def rerank(self, query, candidates):
        return sorted(candidates, key=lambda candidate: candidate.point_id, reverse=True)


@pytest.mark.anyio
async def test_model_routes_request_to_selected_store():
    from knowledge_mcp.retrieval import MultiModelStore

    bge = CandidateStore([result("bge", "b", 0.2)])
    store = MultiModelStore({"dragonkue/BGE-m3-ko": bge})
    assert (await store.hybrid_search(SearchRequest(query="q", rerank=False)))[0].point_id == "bge"
    assert len(bge.requests) == 1


@pytest.mark.anyio
async def test_rerank_precedes_source_cap_and_limit():
    from knowledge_mcp.retrieval import MultiModelStore

    candidates = [result(str(i), "one", 1 / (i + 1)) for i in range(3)] + [result("z", "two", 0.1)]
    store = MultiModelStore({"dragonkue/BGE-m3-ko": CandidateStore(candidates)}, FakeReranker())
    found = await store.hybrid_search(SearchRequest(query="q", rerank=True, limit=3))
    assert [item.point_id for item in found] == ["z", "2", "1"]


def test_request_rejects_unknown_model():
    with pytest.raises(ValueError):
        SearchRequest(query="q", embedding_model="unknown")


def test_search_result_score_must_be_finite():
    with pytest.raises(ValueError):
        result("bad", "one", float("inf"))


@pytest.mark.anyio
async def test_reranker_loads_on_first_use_and_replaces_rrf_scores(monkeypatch):
    from knowledge_mcp.reranker import LocalReranker

    calls = []

    class CrossEncoder:
        def __init__(self, name, *, device, model_kwargs):
            calls.append((name, device, model_kwargs))

        def predict(self, pairs, *, batch_size=8, convert_to_tensor):
            assert pairs == [("q", "a"), ("q", "b")]
            assert convert_to_tensor is True
            return torch.tensor([0.2, 0.8])

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(CrossEncoder=CrossEncoder))
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    reranker = LocalReranker()
    assert not reranker.loaded
    ranked = await reranker.rerank("q", [result("a", "one", 0.9), result("b", "two", 0.1)])
    assert [item.point_id for item in ranked] == ["b", "a"]
    assert [item.score for item in ranked] == pytest.approx([0.8, 0.2])
    assert reranker.loaded
    await reranker.rerank("q", [result("a", "one", 0.9), result("b", "two", 0.1)])
    assert calls == [("dragonkue/bge-reranker-v2-m3-ko", "cuda", {"torch_dtype": "bfloat16"})]


@pytest.mark.anyio
async def test_reranker_inference_failure_is_not_hidden(monkeypatch):
    from knowledge_mcp.reranker import LocalReranker

    class CrossEncoder:
        def __init__(self, name, *, device, model_kwargs):
            pass

        def predict(self, pairs, *, batch_size=8, convert_to_tensor):
            raise RuntimeError("inference failed")

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(CrossEncoder=CrossEncoder))
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    with pytest.raises(RuntimeError, match="reranker_inference_failed"):
        await LocalReranker().rerank("q", [result("a", "one", 0.9)])


@pytest.mark.anyio
async def test_reranker_rejects_non_finite_scores(monkeypatch):
    from knowledge_mcp.reranker import LocalReranker

    class CrossEncoder:
        def __init__(self, name, *, device, model_kwargs):
            pass

        def predict(self, pairs, *, batch_size=8, convert_to_tensor):
            return torch.tensor([float("nan")])

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(CrossEncoder=CrossEncoder))
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    with pytest.raises(RuntimeError, match="reranker_invalid_scores"):
        await LocalReranker().rerank("q", [result("a", "one", 0.9)])


@pytest.mark.anyio
async def test_default_reranking_applies_before_source_limit():
    from knowledge_mcp.retrieval import MultiModelStore

    store = MultiModelStore({"dragonkue/BGE-m3-ko": CandidateStore([
        result("a", "one", .9), result("b", "two", .1)])}, FakeReranker())
    found = await store.hybrid_search(SearchRequest(query="q", limit=1))
    assert [item.point_id for item in found] == ["b"]
    assert found[0].rerank_requested is True
    assert found[0].rerank_applied is True
    assert found[0].rerank_error is None


@pytest.mark.anyio
async def test_explicit_false_preserves_rrf_without_model_construction(monkeypatch):
    from knowledge_mcp.retrieval import MultiModelStore

    def forbidden(*args, **kwargs):
        pytest.fail("explicit false loaded a reranker")

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(CrossEncoder=forbidden))
    store = MultiModelStore({"dragonkue/BGE-m3-ko": CandidateStore([result("a", "one", .9)])})
    found = await store.hybrid_search(SearchRequest(query="q", rerank=False))
    assert (found[0].score, found[0].rerank_requested, found[0].rerank_applied, found[0].rerank_error) == (.9, False, False, None)
    assert not store.reranker.loaded


@pytest.mark.anyio
@pytest.mark.parametrize("available,cuda_fails,devices,fallback", [
    (True, False, ["cuda"], None),
    (True, True, ["cuda", "cpu"], "reranker_cuda_init_failed"),
    (False, False, ["cpu"], "reranker_cuda_unavailable"),
])
async def test_device_selection_and_cpu_fallback_still_apply(monkeypatch, available, cuda_fails, devices, fallback):
    from knowledge_mcp.reranker import LocalReranker
    from knowledge_mcp.retrieval import MultiModelStore

    attempts = []
    class CrossEncoder:
        def __init__(self, name, *, device, model_kwargs):
            attempts.append((device, model_kwargs))
            if device == "cuda" and cuda_fails:
                raise RuntimeError("private constructor details")
        def predict(self, pairs, *, batch_size=8, convert_to_tensor):
            return torch.tensor([.2, .8])

    monkeypatch.setattr("torch.cuda.is_available", lambda: available)
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(CrossEncoder=CrossEncoder))
    reranker = LocalReranker()
    store = MultiModelStore({"dragonkue/BGE-m3-ko": CandidateStore([result("a", "one", .9), result("b", "two", .1)])}, reranker)
    found = await store.hybrid_search(SearchRequest(query="q"))
    assert [item.point_id for item in found] == ["b", "a"]
    assert all(item.rerank_applied and item.rerank_error is None for item in found)
    assert attempts == [(device, {"torch_dtype": "bfloat16" if device == "cuda" else "float32"}) for device in devices]
    assert reranker.status == {"model": reranker.model_name, "loaded": True, "device": devices[-1], "last_fallback": fallback,
                              "precision_policy": "auto", "effective_dtype": "bfloat16" if devices[-1] == "cuda" else "float32"}


@pytest.mark.anyio
@pytest.mark.parametrize("failure,code", [
    ("init", "reranker_init_failed"),
    ("inference", "reranker_inference_failed"),
    ("nan", "reranker_invalid_scores"),
    ("inf", "reranker_invalid_scores"),
    ("short", "reranker_invalid_scores"),
    ("text", "reranker_invalid_scores"),
])
async def test_failed_reranker_returns_unchanged_rrf_candidates(monkeypatch, failure, code):
    from knowledge_mcp.retrieval import MultiModelStore

    attempts = []
    class CrossEncoder:
        def __init__(self, name, *, device, model_kwargs):
            attempts.append(device)
            if failure == "init":
                raise RuntimeError("private init payload")
        def predict(self, pairs, *, batch_size=8, convert_to_tensor):
            if failure == "inference":
                raise RuntimeError("private inference payload")
            if failure == "text":
                return ["invalid", .1]
            return torch.tensor({"nan": [float("nan"), .1], "inf": [.1, float("inf")], "short": [.1]}[failure])

    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(CrossEncoder=CrossEncoder))
    candidates = [result("a", "one", .9), result("b", "two", .1)]
    store = MultiModelStore({"dragonkue/BGE-m3-ko": CandidateStore(candidates)})
    found = await store.hybrid_search(SearchRequest(query="q"))
    assert [(item.point_id, item.score) for item in found] == [("a", .9), ("b", .1)]
    assert all(item.rerank_requested and not item.rerank_applied and item.rerank_error == code for item in found)
    assert store.reranker.status["last_fallback"] == code
    assert all(not item.rerank_requested for item in candidates)
    for original, returned in zip(candidates, found):
        assert returned.model_dump(exclude={"rerank_requested", "rerank_applied", "rerank_error"}) == original.model_dump(exclude={"rerank_requested", "rerank_applied", "rerank_error"})
    if failure == "init":
        await store.hybrid_search(SearchRequest(query="again"))
        assert attempts == ["cuda", "cpu"]
        assert store.reranker.status["device"] is None
        assert store.reranker.status["effective_dtype"] is None


@pytest.mark.anyio
async def test_empty_candidates_do_not_apply_or_load_reranker():
    from knowledge_mcp.retrieval import MultiModelStore

    store = MultiModelStore({"dragonkue/BGE-m3-ko": CandidateStore([])})
    assert await store.hybrid_search(SearchRequest(query="q")) == []
    assert store.reranker.status["loaded"] is False


@pytest.mark.anyio
@pytest.mark.parametrize("batch,expected", [(None, 8), (3, 3)])
async def test_reranker_batch_reaches_predict(tmp_path, monkeypatch, batch, expected):
    from dataclasses import replace
    from knowledge_mcp.config import Settings
    from knowledge_mcp.reranker import LocalReranker
    calls = []
    class CrossEncoder:
        def __init__(self, name, *, device, model_kwargs):
            pass
        def predict(self, pairs, *, batch_size, convert_to_tensor):
            calls.append(batch_size)
            return torch.tensor([.2, .8])
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(CrossEncoder=CrossEncoder))
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    settings = Settings.from_paths(vault_root=tmp_path / "vault", project_root=tmp_path / "project")
    if batch is not None:
        settings = replace(settings, reranker_batch_size=batch)
    reranker = LocalReranker(settings=settings)
    assert not reranker.loaded
    ranked = await reranker.rerank("synthetic", [result("a", "one", .9), result("b", "two", .1)])
    assert [item.point_id for item in ranked] == ["b", "a"]
    assert calls == [expected]
