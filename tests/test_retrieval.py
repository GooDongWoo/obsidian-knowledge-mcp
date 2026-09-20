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
    assert (await store.hybrid_search(SearchRequest(query="q")))[0].point_id == "bge"
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
    reranker = LocalReranker()
    assert not reranker.loaded
    ranked = await reranker.rerank("q", [result("a", "one", 0.9), result("b", "two", 0.1)])
    assert [item.point_id for item in ranked] == ["b", "a"]
    assert [item.score for item in ranked] == [0.8, 0.2]
    assert reranker.loaded
    await reranker.rerank("q", [result("a", "one", 0.9), result("b", "two", 0.1)])
    assert calls == [("dragonkue/bge-reranker-v2-m3-ko", "cpu")]


@pytest.mark.anyio
async def test_reranker_inference_failure_is_not_hidden(monkeypatch):
    from knowledge_mcp.reranker import LocalReranker

    class CrossEncoder:
        def __init__(self, name, *, device):
            pass

        def predict(self, pairs):
            raise RuntimeError("inference failed")

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(CrossEncoder=CrossEncoder))
    with pytest.raises(RuntimeError, match="inference failed"):
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
    with pytest.raises(ValueError, match="non-finite"):
        await LocalReranker().rerank("q", [result("a", "one", 0.9)])
