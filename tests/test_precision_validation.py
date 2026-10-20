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


def phase_evidence():
    """Small protocol fixture; no vectors/models needed to test conjunction."""
    import copy
    identity = {"dataset_sha256": "frozen", "models": {"embedding": {"id": "m", "revision": "r"}},
                "packages": {"torch": "installed"}, "inference_contract": {"batch_size": 8}, "python": "3"}
    runtime = {"device": "cuda", "threads": 4}
    roles = [(component, precision, "cuda") for component in ("embedding", "reranker")
             for precision in ("default", "float32")]
    roles += [("production_both", "float32", "cuda"), ("production_both", "auto", "cuda"),
              ("production_both", "auto", "cpu")]
    runs = [{"component": component, "requested_precision": precision, "requested_device": device,
             "identity": copy.deepcopy(identity), "runtime": dict(runtime, device=device)}
            for component, precision, device in roles]
    compare = {"phase": "compare", "identity": copy.deepcopy(identity), "runs": runs,
               "comparison": {"old_default_vs_explicit_fp32": {"embedding": {"pass": True}, "reranker": {"pass": True}},
                              "old_explicit_vs_production_override": {"embedding": {"pass": True}},
                              "production_fp32_vs_auto_vectors": {"pass": True}, "production_dtype_pass": True,
                              "cpu_functional_dtype_pass": True, "performance": {"pass": True}}}
    qdrant = {"phase": "qdrant", "identity": copy.deepcopy(identity),
              "runs": [{"identity": copy.deepcopy(identity), "runtime": dict(runtime), "complete": True,
                        "vector_diagnostics": {"pass": True}, "filter_violations": 0, "fallback_count": 0} for _ in range(2)],
              "cpu_read": {"identity": copy.deepcopy(identity), "runtime": dict(runtime, device="cpu")},
              "comparison": {"status": "PASS", "cpu_read_pass": True, "dtype_pass": True,
                             "payload_bm25_semantics_equal": True, "schema_equal": True,
                             "chunk_vector_comparison": {"pass": True}, "performance": {"pass": True},
                             "quality": {key: {"pass": True} for key in ("candidates", "rerank_false", "same_candidate", "end_to_end")}}}
    return compare, qdrant


def test_joined_gate_requires_compare_and_every_worker():
    module = harness()
    compare, qdrant = phase_evidence()
    assert module.join_phase_gates(None, qdrant)["status"] == "HOLD"
    assert module.join_phase_gates(compare, None)["status"] == "HOLD"
    assert module.join_phase_gates(compare, qdrant)["status"] == "PASS"
    compare["runs"].pop()
    assert module.join_phase_gates(compare, qdrant)["status"] == "HOLD"
    compare, qdrant = phase_evidence()
    qdrant["runs"].pop()
    assert module.join_phase_gates(compare, qdrant)["status"] == "HOLD"


@pytest.mark.parametrize("key", ["dataset_sha256", "models", "packages", "inference_contract", "python"])
def test_joined_gate_holds_on_cross_phase_or_worker_identity_drift(key):
    module = harness()
    compare, qdrant = phase_evidence()
    compare["identity"][key] = "different"
    assert module.join_phase_gates(compare, qdrant)["status"] == "HOLD"
    compare, qdrant = phase_evidence()
    compare["runs"][0]["identity"][key] = "different"
    assert module.join_phase_gates(compare, qdrant)["status"] == "HOLD"
    compare, qdrant = phase_evidence()
    qdrant["cpu_read"]["identity"][key] = "different"
    assert module.join_phase_gates(compare, qdrant)["status"] == "HOLD"


@pytest.mark.parametrize("path", [
    ("old_default_vs_explicit_fp32", "embedding", "pass"),
    ("old_default_vs_explicit_fp32", "reranker", "pass"),
    ("old_explicit_vs_production_override", "embedding", "pass"),
    ("production_fp32_vs_auto_vectors", "pass"), ("production_dtype_pass",),
    ("cpu_functional_dtype_pass",), ("performance", "pass"),
])
def test_joined_gate_holds_on_each_false_or_missing_compare_gate(path):
    module = harness()
    compare, qdrant = phase_evidence()
    target = compare["comparison"]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = False
    assert module.join_phase_gates(compare, qdrant)["status"] == "HOLD"
    del target[path[-1]]
    assert module.join_phase_gates(compare, qdrant)["status"] == "HOLD"


@pytest.mark.parametrize("path", [("dtype_pass",), ("payload_bm25_semantics_equal",), ("schema_equal",),
                                 ("chunk_vector_comparison", "pass"), ("performance", "pass"),
                                 *[("quality", key, "pass") for key in ("candidates", "rerank_false", "same_candidate", "end_to_end")]])
def test_joined_gate_rejects_false_index_subgate_despite_cached_pass(path):
    module = harness()
    compare, qdrant = phase_evidence()
    target = qdrant["comparison"]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = False
    assert module.join_phase_gates(compare, qdrant)["status"] == "HOLD"
    del target[path[-1]]
    assert module.join_phase_gates(compare, qdrant)["status"] == "HOLD"


def test_joined_gate_holds_on_index_cpu_or_actual_runtime_failure():
    module = harness()
    for fault in ("index", "cpu", "runtime", "complete"):
        compare, qdrant = phase_evidence()
        if fault == "index":
            qdrant["comparison"]["status"] = "HOLD"
        elif fault == "cpu":
            qdrant["comparison"]["cpu_read_pass"] = False
        elif fault == "runtime":
            qdrant["runs"][1]["runtime"]["threads"] = 8
        else:
            qdrant["runs"][0]["complete"] = False
        assert module.join_phase_gates(compare, qdrant)["status"] == "HOLD"
    for device in ("cuda", "cpu"):
        compare, qdrant = phase_evidence()
        run = next(row for row in compare["runs"] if row["requested_device"] == device)
        run["runtime"]["threads"] = 8
        assert module.join_phase_gates(compare, qdrant)["status"] == "HOLD"


def test_finalize_cli_is_offline_and_binds_exact_artifacts(tmp_path, monkeypatch):
    from hashlib import sha256
    module = harness()
    compare, qdrant = phase_evidence()
    compare_path, qdrant_path = tmp_path / "compare.json", tmp_path / "qdrant.json"
    compare_path.write_text(json.dumps(compare), encoding="utf-8")
    qdrant_path.write_text(json.dumps(qdrant), encoding="utf-8")
    monkeypatch.setattr(module, "prepare_identity", lambda *args: pytest.fail("offline finalize must not inspect model cache"))
    monkeypatch.setattr(module, "run_worker", lambda *args: pytest.fail("offline finalize must not load models"))
    monkeypatch.delenv("KNOWLEDGE_RUN_REAL_MODELS", raising=False)
    output = tmp_path / "joined.json"
    module.main(["--phase", "finalize", "--compare-artifact", str(compare_path), "--qdrant-artifact", str(qdrant_path), "--output", str(output)])
    joined = json.loads(output.read_text(encoding="utf-8"))
    assert joined["status"] == "PASS"
    assert joined["input_artifacts"]["compare"]["sha256"] == sha256(compare_path.read_bytes()).hexdigest()
    assert joined["input_artifacts"]["qdrant"]["identity"] == qdrant["identity"]
    module.main(["--phase", "finalize", "--output", str(output)])
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "HOLD"


def test_qdrant_cli_never_promotes_index_only_pass_to_final_pass(tmp_path, monkeypatch):
    module = harness()
    compare, qdrant = phase_evidence()
    cpu = qdrant["cpu_read"]
    for component in ("embedding", "reranker"):
        cpu[component] = {"parameter_dtypes": {"torch.float32": 1}, "parameter_devices": {"cpu": 1}}
        cpu[component + "_forward"] = [{"output_dtype": "torch.float32", "device": "cpu"}]
    workers = iter([*qdrant["runs"], cpu])
    monkeypatch.setattr(module, "prepare_identity", lambda *args: compare["identity"])
    monkeypatch.setattr(module, "verify_owned_qdrant", lambda *args: {})
    monkeypatch.setattr(module, "execute_child", lambda *args: next(workers))
    monkeypatch.setattr(module, "qdrant_comparison", lambda *args: {"status": "PASS"})
    monkeypatch.setenv("KNOWLEDGE_RUN_REAL_MODELS", "1")
    output = tmp_path / "index-only.json"
    module.main(["--phase", "qdrant", "--qdrant-url", "http://127.0.0.1:36333", "--output", str(output)])
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["comparison"]["status"] == "PASS"
    assert result["status"] == "HOLD_pending_join"
