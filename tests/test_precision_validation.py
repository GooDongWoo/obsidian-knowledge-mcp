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
