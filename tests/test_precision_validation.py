"""Offline checks for the frozen benchmark; never construct real models."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "validate_inference_precision.py"


def harness():
    assert SCRIPT.is_file(), "baseline validator has not been implemented"
    spec = importlib.util.spec_from_file_location("precision_validator", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dataset_freezes_diverse_queries_and_prejudged_relevance():
    module = harness()
    documents, queries = module.load_dataset(module.DEFAULT_DATASET)
    assert len(documents) == 200
    assert len(queries) == 100
    from collections import Counter
    assert Counter(q["category"] for q in queries) == {
        "korean_semantic": 30, "mixed_technical": 20, "exact_identifier": 15,
        "negation_hard_negative": 15, "date_type_private_filter": 10, "long_input": 10,
    }
    ids = {d["id"] for d in documents}
    assert len(ids) == 200
    for query in queries:
        assert set(query["relevance"]) <= ids
        assert set(query["relevance"].values()) == {0, 1, 2}
        assert query["rationale"]
    assert all(len(q["text"]) > 1000 for q in queries if q["category"] == "long_input")
    assert any(q.get("filters", {}).get("include_private") for q in queries)


def test_preparation_records_hashes_without_importing_model_packages(tmp_path):
    module = harness()
    for model in module.FROZEN_MODELS.values():
        snapshot = tmp_path / ("models--" + model["id"].replace("/", "--")) / "snapshots" / model["revision"]
        snapshot.mkdir(parents=True)
    identity = module.prepare_identity(module.DEFAULT_DATASET, tmp_path)
    assert len(identity["dataset_sha256"]) == 64
    assert identity["counts"] == {"documents": 200, "queries": 100}
    assert all(len(model["revision"]) == 40 for model in identity["models"].values())
    assert identity["inference_contract"]["normalize_embeddings"] is True
    assert identity["inference_contract"]["chunking"] == {"size": 400, "overlap": 60}
    before = identity["dataset_sha256"]
    changed = tmp_path / "changed.jsonl"
    changed.write_bytes(module.DEFAULT_DATASET.read_bytes() + b"\n")
    assert module.prepare_identity(changed, tmp_path)["dataset_sha256"] != before


def test_dataset_rejects_unknown_qrels_and_duplicate_ids(tmp_path):
    module = harness()
    rows = [json.loads(line) for line in module.DEFAULT_DATASET.read_text(encoding="utf-8").splitlines()]
    rows[-1]["relevance"]["missing"] = 2
    bad = tmp_path / "bad.jsonl"
    bad.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown document"):
        module.load_dataset(bad)
    rows[-1]["relevance"].pop("missing")
    rows[1]["id"] = rows[0]["id"]
    bad.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        module.load_dataset(bad)


def test_cli_requires_opt_in_before_model_loading(tmp_path):
    harness()
    environment = {key: value for key, value in os.environ.items() if not key.startswith("KNOWLEDGE_")}
    result = subprocess.run([sys.executable, str(SCRIPT), "--phase", "baseline", "--output", str(tmp_path / "no.json")],
                            env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert "KNOWLEDGE_RUN_REAL_MODELS=1" in result.stderr
    assert not (tmp_path / "no.json").exists()


def test_worker_cannot_bypass_inference_opt_in_with_prepare_only(tmp_path, monkeypatch):
    module = harness()
    monkeypatch.setattr(module, "run_worker", lambda args: {"unexpected_inference": True})
    monkeypatch.delenv("KNOWLEDGE_RUN_REAL_MODELS", raising=False)
    with pytest.raises(SystemExit):
        module.main(["--phase", "baseline", "--worker", "--prepare-only", "--component", "embedding",
                     "--precision", "default", "--device", "cpu", "--output", str(tmp_path / "bad.json")])


def test_ordinary_tests_block_parent_knowledge_and_env_loading(tmp_path):
    from knowledge_mcp import config
    assert "KNOWLEDGE_ISOLATION_SENTINEL" not in os.environ
    (tmp_path / ".env").write_text("KNOWLEDGE_ISOLATION_SENTINEL=leaked\n", encoding="utf-8")
    config._load_env_file(tmp_path)
    assert "KNOWLEDGE_ISOLATION_SENTINEL" not in os.environ


def test_baseline_loader_preserves_default_arm_and_forces_local_pinned_fp32(tmp_path, monkeypatch):
    module = harness()
    calls = []
    def constructor(path, **kwargs):
        calls.append((path, kwargs))
        return object()
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(float32="float32-sentinel"))
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(
        SentenceTransformer=constructor, CrossEncoder=constructor))
    for component, model in module.FROZEN_MODELS.items():
        snapshot = tmp_path / ("models--" + model["id"].replace("/", "--")) / "snapshots" / model["revision"]
        snapshot.mkdir(parents=True)
        module.load_frozen_model(component, "default", "cpu", tmp_path)
        module.load_frozen_model(component, "float32", "cuda", tmp_path)
        assert calls[-2] == (str(snapshot.resolve()), {
            "device": "cpu", "revision": model["revision"], "local_files_only": True})
        assert calls[-1] == (str(snapshot.resolve()), {
            "device": "cuda", "revision": model["revision"], "local_files_only": True,
            "model_kwargs": {"torch_dtype": "float32-sentinel"}})


def test_real_model_marker_is_checked_at_collection_before_env_cleanup(tmp_path):
    import shutil
    shutil.copy(ROOT / "tests" / "conftest.py", tmp_path / "conftest.py")
    (tmp_path / "test_opt_in.py").write_text(
        "import pytest\n@pytest.mark.real_models\ndef test_probe():\n    assert False, 'opt-in reached'\n", encoding="utf-8")
    environment = {k: v for k, v in os.environ.items() if not k.startswith("KNOWLEDGE_")}
    environment["PYTHONPATH"] = str(ROOT / "src")
    command = [sys.executable, "-m", "pytest", "-q", "-o", "markers=real_models: real model opt-in", str(tmp_path)]
    result = subprocess.run(command, cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0 and "1 skipped" in result.stdout
    environment["KNOWLEDGE_RUN_REAL_MODELS"] = "1"
    result = subprocess.run(command, cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 1 and "opt-in reached" in result.stdout

def test_document_metrics_deduplicate_chunks_and_use_graded_ndcg():
    module = harness()
    metrics = module.ranking_metrics(["x", "a", "a", "b"], {"a": 2, "b": 1, "c": 0})
    assert metrics["recall20"] == 1.0
    assert metrics["mrr8"] == 0.5
    # DCG=3/log2(3)+1/log2(4); IDCG=3+1/log2(3).
    assert metrics["ndcg8"] == pytest.approx(0.6590018048024133)
    assert module.ranking_metrics([], {"a": 2}) == {"recall20": 0.0, "ndcg8": 0.0, "mrr8": 0.0}


def test_bootstrap_uses_paired_scenarios_and_fixed_seed():
    module = harness()
    assert module.paired_bootstrap([-0.01] * 100) == pytest.approx([-0.01, -0.01])
    mixed = [-0.5, 0.5] * 50
    # Query pairs form 50 correlated scenarios; their means are exactly zero.
    assert module.paired_bootstrap(mixed) == [0.0, 0.0]
    assert module.paired_bootstrap([0.2, 0.2, -0.1, -0.1]) == module.paired_bootstrap([0.2, 0.2, -0.1, -0.1])


def test_comparison_rejects_drift_but_allows_recorded_harness_evolution():
    module = harness()
    identity = {"dataset_sha256": "a", "models": {"x": {"id": "m", "revision": "r", "snapshot": "local"}},
                "packages": {"torch": "1"}, "inference_contract": {"batch_size": 8}, "python": "3", "git_head": "old"}
    newer = dict(identity, git_head="new", harness_sha256="new")
    assert module.require_comparable(identity, newer) is None
    for key in ("dataset_sha256", "models", "packages", "inference_contract"):
        with pytest.raises(ValueError, match=key):
            module.require_comparable(identity, dict(newer, **{key: "drift"}))


def test_owned_endpoint_rejects_production_and_unmatched_process(tmp_path):
    module = harness()
    metadata = {"url": "http://127.0.0.1:36333", "pid": 42, "storage": str(tmp_path), "executable": "qdrant.exe"}
    module.check_owned_facts(metadata["url"], metadata, {"pid": 42, "command": f"qdrant.exe --storage {tmp_path}", "port_pid": 42})
    for url, facts in [("http://127.0.0.1:26333", {"pid": 42, "command": str(tmp_path), "port_pid": 42}),
                       (metadata["url"], {"pid": 43, "command": str(tmp_path), "port_pid": 43}),
                       (metadata["url"], {"pid": 42, "command": "unrelated", "port_pid": 42})]:
        with pytest.raises(ValueError, match="owned"):
            module.check_owned_facts(url, metadata, facts)


def test_materialized_vault_preserves_explicit_parser_metadata(tmp_path):
    module = harness()
    from knowledge_mcp.config import Settings
    from knowledge_mcp.documents import parse_source
    document = {"id": "a", "source_path": "synthetic/a.md", "text": "원문 그대로", "security_level": "private",
                "document_type": "experience", "created_at": "2026-10-01T10:00:00+09:00", "modified_at": "2026-10-20T12:00:00+09:00"}
    module.materialize_vault(tmp_path, [document])
    settings = Settings(tmp_path, tmp_path / ".runtime", Settings.DEFAULT_QDRANT_URL, "test", Settings.DEFAULT_DENSE_MODEL, "test", project_root=tmp_path / "project")
    parsed = parse_source(tmp_path / document["source_path"], settings)
    assert parsed.security_level == "private"
    assert parsed.document_type == "experience"
    assert parsed.created_at == document["created_at"]
    assert parsed.modified_at == document["modified_at"]
    assert parsed.sections[0].text == "원문 그대로"


def test_quality_gates_hold_on_critical_or_bootstrap_failure():
    module = harness()
    queries = [{"id": "q1", "critical": True, "relevance": {"a": 2, "b": 1}}]
    perfect = [["a", "b"]]
    report = module.compare_rankings(queries, perfect, [["c", "b"]])
    assert report["critical_missing_increase"] == 1
    assert report["pass"] is False
    assert module.compare_rankings(queries, perfect, perfect)["pass"] is True


def test_vector_diagnostics_catch_nonfinite_zero_and_wrong_dimension():
    import numpy as np
    module = harness()
    good = [[1.0] + [0.0] * 1023]
    assert module.vector_diagnostics(good)["pass"] is True
    for bad in ([[0.0] * 1024], [[float("nan")] + [0.0] * 1023], [[1.0, 0.0]]):
        assert module.vector_diagnostics(bad)["pass"] is False
    assert module.vector_comparison(good, good) == {"cosine_mean": 1.0, "cosine_min": 1.0, "pass": True}


def test_gate_summary_holds_on_latency_memory_leak_or_fallback():
    module = harness()
    baseline = {"steady_allocated_bytes": 1000, "warm_p95_ms": 100, "throughput": 10}
    candidate = {"steady_allocated_bytes": 700, "warm_p95_ms": 109, "throughput": 9.6,
                 "allocated_samples": [700] * 501, "fallback_count": 0}
    assert module.performance_gates(baseline, candidate)["pass"] is True
    # Exact permitted boundaries must not fail from binary floating subtraction.
    assert module.performance_gates(baseline, dict(candidate, steady_allocated_bytes=750,
                                                  warm_p95_ms=110, throughput=9.5))["pass"] is True
    for changes in ({"warm_p95_ms": 111}, {"throughput": 9.4}, {"steady_allocated_bytes": 800},
                    {"allocated_samples": [i * 1000000 for i in range(501)]}, {"fallback_count": 1}):
        assert module.performance_gates(baseline, dict(candidate, **changes))["pass"] is False


def test_canonical_points_ignore_generation_and_preserve_sparse_and_metadata():
    module = harness()
    payload = {"document": "body", "metadata": {"source_path": "a.md", "chunk_index": 0, "generation": "g1", "security_level": "private"}}
    first = [{"id": "one", "payload": payload, "vector": {"dense": [1.0], "bm25": {"indices": [4, 2], "values": [0.5, 1.0]}}}]
    second = [{"id": "two", "payload": {"document": "body", "metadata": dict(payload["metadata"], generation="g2")},
               "vector": {"dense": [0.9], "bm25": {"indices": [2, 4], "values": [1.0, 0.5]}}}]
    assert module.canonical_points(first) == module.canonical_points(second)
    second[0]["payload"]["metadata"]["security_level"] = "public"
    assert module.canonical_points(first) != module.canonical_points(second)

def test_compare_prepare_only_never_loads_models(tmp_path, monkeypatch):
    module = harness()
    for model in module.FROZEN_MODELS.values():
        (tmp_path / ("models--" + model["id"].replace("/", "--")) / "snapshots" / model["revision"]).mkdir(parents=True)
    monkeypatch.setattr(module, "run_worker", lambda args: pytest.fail("unexpected model load"))
    output = tmp_path / "compare.json"
    module.main(["--phase", "compare", "--prepare-only", "--cache-dir", str(tmp_path), "--output", str(output)])
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "prepared_not_measured"


def test_filter_audit_catches_private_type_and_calendar_violations():
    module = harness()
    query = {"filters": {"document_type": ["reference"], "created_from": "2026-10-01", "created_to": "2026-10-20"}}
    good = {"security_level": "public", "document_type": "reference", "created_at": "2026-10-20T23:59:59+09:00"}
    assert module.filter_violations(query, [good]) == 0
    assert module.filter_violations(query, [dict(good, security_level="private"), dict(good, document_type="diary"), dict(good, created_at="2026-10-21T00:00:00+09:00")]) == 3

def test_score_diagnostics_align_same_candidates_and_report_ties():
    module = harness()
    baseline = [[{"source_path": "a", "document": "A", "score": 0.5}, {"source_path": "b", "document": "B", "score": 0.5}]]
    candidate = [[{"source_path": "b", "document": "B", "score": 0.6}, {"source_path": "a", "document": "A", "score": 0.4}]]
    result = module.score_diagnostics(baseline, candidate)
    assert result["score_abs_error_mean"] == pytest.approx(0.1)
    assert result["baseline_adjacent_exact_ties"] == 1
    assert result["candidate_adjacent_exact_ties"] == 0

def test_failed_worker_writes_hold_artifact_instead_of_success(tmp_path, monkeypatch):
    module = harness()
    args = SimpleNamespace(dataset=module.DEFAULT_DATASET, cache_dir=tmp_path, output=tmp_path / "summary.json")
    def fail(*a, **kw):
        raise subprocess.CalledProcessError(9, a[0])
    monkeypatch.setattr(module.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        module.execute_child(args, {}, "compare", tmp_path / "worker.json", [])
    result = json.loads(args.output.read_text(encoding="utf-8"))
    assert result["status"] == "HOLD"
    assert result["failed_returncode"] == 9

def test_kernel_probe_records_cuda_events_and_runs_representative_call(monkeypatch):
    module = harness()
    calls = []
    class Profile:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def events(self):
            return [SimpleNamespace(device_type="DeviceType.CPU", name="aten::linear"),
                    SimpleNamespace(device_type="DeviceType.CUDA", name="bf16_gemm")]
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(profiler=SimpleNamespace(
        profile=lambda **kwargs: Profile(), ProfilerActivity=SimpleNamespace(CPU="cpu", CUDA="cuda"))))
    result = module.kernel_probe(lambda: calls.append("inferred"))
    assert calls == ["inferred"]
    assert result == {"status": "observed", "cuda_kernels": ["bf16_gemm"], "attention_ops": []}

def test_synthetic_vault_has_own_empty_git_history_before_parser(tmp_path):
    module = harness()
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    vault = tmp_path / "runtime" / "vault"
    vault.mkdir(parents=True)
    module.materialize_vault(vault, [])
    actual = subprocess.check_output(["git", "-C", str(vault), "rev-parse", "--show-toplevel"], text=True).strip()
    assert Path(actual).resolve() == vault.resolve()
    history = subprocess.run(["git", "-C", str(vault), "log", "--oneline"], capture_output=True, text=True)
    assert not history.stdout.strip()
