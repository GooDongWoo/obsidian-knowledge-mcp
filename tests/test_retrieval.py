import pytest
import sys
from types import SimpleNamespace

from knowledge_mcp.search import SearchRequest, SearchResult


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
        def __init__(self, name, *, device):
            calls.append((name, device))

        def predict(self, pairs):
            assert pairs == [("q", "a"), ("q", "b")]
            return [0.2, 0.8]

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(CrossEncoder=CrossEncoder))
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    reranker = LocalReranker()
    assert not reranker.loaded
    ranked = await reranker.rerank("q", [result("a", "one", 0.9), result("b", "two", 0.1)])
    assert [item.point_id for item in ranked] == ["b", "a"]
    assert [item.score for item in ranked] == [0.8, 0.2]
    assert reranker.loaded
    await reranker.rerank("q", [result("a", "one", 0.9), result("b", "two", 0.1)])
    assert calls == [("dragonkue/bge-reranker-v2-m3-ko", "cuda")]


@pytest.mark.anyio
async def test_reranker_inference_failure_is_not_hidden(monkeypatch):
    from knowledge_mcp.reranker import LocalReranker

    class CrossEncoder:
        def __init__(self, name, *, device):
            pass

        def predict(self, pairs):
            raise RuntimeError("inference failed")

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(CrossEncoder=CrossEncoder))
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    with pytest.raises(RuntimeError, match="reranker_inference_failed"):
        await LocalReranker().rerank("q", [result("a", "one", 0.9)])


@pytest.mark.anyio
async def test_reranker_rejects_non_finite_scores(monkeypatch):
    from knowledge_mcp.reranker import LocalReranker

    class CrossEncoder:
        def __init__(self, name, *, device):
            pass

        def predict(self, pairs):
            return [float("nan")]

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
        def __init__(self, name, *, device):
            attempts.append(device)
            if device == "cuda" and cuda_fails:
                raise RuntimeError("private constructor details")
        def predict(self, pairs):
            return [.2, .8]

    monkeypatch.setattr("torch.cuda.is_available", lambda: available)
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(CrossEncoder=CrossEncoder))
    reranker = LocalReranker()
    store = MultiModelStore({"dragonkue/BGE-m3-ko": CandidateStore([result("a", "one", .9), result("b", "two", .1)])}, reranker)
    found = await store.hybrid_search(SearchRequest(query="q"))
    assert [item.point_id for item in found] == ["b", "a"]
    assert all(item.rerank_applied and item.rerank_error is None for item in found)
    assert attempts == devices
    assert reranker.status == {"model": reranker.model_name, "loaded": True, "device": devices[-1], "last_fallback": fallback}


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
        def __init__(self, name, *, device):
            attempts.append(device)
            if failure == "init":
                raise RuntimeError("private init payload")
        def predict(self, pairs):
            if failure == "inference":
                raise RuntimeError("private inference payload")
            return {"nan": [float("nan"), .1], "inf": [.1, float("inf")], "short": [.1], "text": ["invalid", .1]}[failure]

    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(CrossEncoder=CrossEncoder))
    candidates = [result("a", "one", .9), result("b", "two", .1)]
    store = MultiModelStore({"dragonkue/BGE-m3-ko": CandidateStore(candidates)})
    found = await store.hybrid_search(SearchRequest(query="q"))
    assert [(item.point_id, item.score) for item in found] == [("a", .9), ("b", .1)]
    assert all(item.rerank_requested and not item.rerank_applied and item.rerank_error == code for item in found)
    assert store.reranker.status["last_fallback"] == code
    assert all(not item.rerank_requested for item in candidates)
    if failure == "init":
        assert attempts == ["cuda", "cpu"]
        assert store.reranker.status["device"] is None


@pytest.mark.anyio
async def test_empty_candidates_do_not_apply_or_load_reranker():
    from knowledge_mcp.retrieval import MultiModelStore

    store = MultiModelStore({"dragonkue/BGE-m3-ko": CandidateStore([])})
    assert await store.hybrid_search(SearchRequest(query="q")) == []
    assert store.reranker.status["loaded"] is False
