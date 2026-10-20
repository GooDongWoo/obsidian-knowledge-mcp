"""Frozen, offline baseline loader, independent of production precision policy.

Preparation never imports torch or sentence_transformers. Real inference requires
KNOWLEDGE_RUN_REAL_MODELS=1 and an exclusive measurement window. Each model,
device and precision combination runs in a fresh sequential child process.
Compare and owned-Qdrant execution are intentionally reserved for Task 5.
"""

import argparse
from collections import Counter
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "tests" / "fixtures" / "inference_precision.jsonl"
FROZEN_MODELS = {
    "embedding": {"id": "dragonkue/BGE-m3-ko", "revision": "7074d66aa46562342193ca4feb3d89bf9dad71b4"},
    "reranker": {"id": "dragonkue/bge-reranker-v2-m3-ko", "revision": "2aca5884ecac490192af9ebd86836d9073d826cd"},
}
CATEGORY_COUNTS = {"korean_semantic": 30, "mixed_technical": 20, "exact_identifier": 15,
                   "negation_hard_negative": 15, "date_type_private_filter": 10, "long_input": 10}


def load_dataset(path):
    records = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [row["id"] for row in records]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate dataset id")
    documents = [row for row in records if row["kind"] == "document"]
    queries = [row for row in records if row["kind"] == "query"]
    if len(documents) != 200 or len(queries) != 100 or len(records) != 300:
        raise ValueError("dataset requires 200 documents and 100 queries")
    if Counter(q["category"] for q in queries) != CATEGORY_COUNTS:
        raise ValueError("dataset category split differs from frozen specification")
    doc_ids = {d["id"] for d in documents}
    for query in queries:
        if not set(query["relevance"]) <= doc_ids:
            raise ValueError("qrels reference unknown document")
        if set(query["relevance"].values()) != {0, 1, 2}:
            raise ValueError("each query requires explicit 0/1/2 judgments")
    return documents, queries


def cached_snapshot(model, cache_dir):
    snapshot = Path(cache_dir) / ("models--" + model["id"].replace("/", "--")) / "snapshots" / model["revision"]
    if not snapshot.is_dir():
        raise FileNotFoundError(f"frozen model revision missing from local cache: {snapshot}")
    return snapshot.resolve()


def prepare_identity(dataset, cache_dir):
    documents, queries = load_dataset(dataset)
    models = {key: dict(model, snapshot=str(cached_snapshot(model, cache_dir)))
              for key, model in FROZEN_MODELS.items()}
    packages = {}
    for name in ("torch", "sentence-transformers", "transformers", "qdrant-client", "fastembed-gpu", "onnxruntime-gpu"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {
        "dataset_sha256": sha256(Path(dataset).read_bytes()).hexdigest(),
        "dataset_path": str(Path(dataset).resolve()), "counts": {"documents": len(documents), "queries": len(queries)},
        "categories": CATEGORY_COUNTS, "models": models, "packages": packages,
        "python": sys.version, "executable": sys.executable,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short", "--untracked-files=all"], cwd=ROOT, text=True),
        "harness_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "inference_contract": {"batch_size": 8, "cpu_threads": 4, "normalize_embeddings": True,
                               "chunking": {"size": 400, "overlap": 60},
                               "baseline_input_unit": "whole synthetic documents (no chunking in DB-free baseline)",
                               "prompt_pooling_truncation": "unchanged model defaults from pinned revision",
                               "cuda_memory_fraction": None},
    }


def load_frozen_model(component, precision, device, cache_dir):
    # Keep this baseline constructor independent of production loaders, including
    # future bf16 defaults. The default arm deliberately supplies no dtype kwarg.
    import torch
    from sentence_transformers import CrossEncoder, SentenceTransformer

    frozen = FROZEN_MODELS[component]
    snapshot = cached_snapshot(frozen, cache_dir)
    kwargs = {"device": device, "revision": frozen["revision"], "local_files_only": True}
    if precision == "float32":
        kwargs["model_kwargs"] = {"torch_dtype": torch.float32}
    constructor = SentenceTransformer if component == "embedding" else CrossEncoder
    return constructor(str(snapshot), **kwargs)


def run_worker(args):
    import torch

    documents, queries = load_dataset(args.dataset)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no silent baseline fallback")
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model = load_frozen_model(args.component, args.precision, args.device, args.cache_dir)
    parameters = model.parameters() if args.component == "embedding" else model.model.parameters()
    dtypes, devices = Counter(), Counter()
    for parameter in parameters:
        dtypes[str(parameter.dtype)] += parameter.numel()
        devices[str(parameter.device)] += parameter.numel()
    if args.device == "cuda":
        torch.cuda.synchronize()
    load_seconds = time.perf_counter() - started
    started = time.perf_counter()
    if args.component == "embedding":
        values = model.encode([d["text"] for d in documents] + [q["text"] for q in queries],
                              batch_size=8, normalize_embeddings=True).tolist()
        outputs = {"document_ids": [d["id"] for d in documents], "query_ids": [q["id"] for q in queries],
                   "document_vectors": values[:len(documents)], "query_vectors": values[len(documents):]}
    else:
        by_id = {d["id"]: d for d in documents}
        # Fixed judged pairs, never chosen from a candidate ranking. Task 5 adds
        # retrieval-candidate evaluation; these preserve baseline scoring identity.
        pair_ids = [[q["id"], did] for q in queries for did in q["relevance"]]
        by_query = {q["id"]: q for q in queries}
        pairs = [(by_query[qid]["text"], by_id[did]["text"]) for qid, did in pair_ids]
        outputs = {"pair_ids": pair_ids, "scores": model.predict(pairs, batch_size=8).tolist()}
    if args.device == "cuda":
        torch.cuda.synchronize()
    return {"component": args.component, "requested_precision": args.precision, "requested_device": args.device,
            "parameter_dtypes": dict(dtypes), "parameter_devices": dict(devices),
            "bf16_supported": torch.cuda.is_bf16_supported() if args.device == "cuda" else None,
            "load_seconds": load_seconds, "inference_seconds": time.perf_counter() - started,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else None,
            "outputs": outputs}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("baseline", "compare", "qdrant"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--cache-dir", type=Path, default=Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub")))
    parser.add_argument("--qdrant-url", help="owned test endpoint; execution added in Task 5")
    parser.add_argument("--prepare-only", action="store_true", help="freeze identity only; no model imports or inference")
    parser.add_argument("--devices", nargs="+", choices=("cpu", "cuda"), default=["cpu", "cuda"])
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--component", choices=tuple(FROZEN_MODELS), help=argparse.SUPPRESS)
    parser.add_argument("--precision", choices=("default", "float32"), help=argparse.SUPPRESS)
    parser.add_argument("--device", choices=("cpu", "cuda"), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.phase != "baseline":
        parser.error("compare and owned-Qdrant execution are deferred to Task 5")
    if (args.worker or not args.prepare_only) and os.environ.get("KNOWLEDGE_RUN_REAL_MODELS") != "1":
        parser.error("real inference requires KNOWLEDGE_RUN_REAL_MODELS=1 and an exclusive measurement window")
    if args.worker:
        if not all((args.component, args.precision, args.device)):
            parser.error("worker requires component, precision and device")
        result = run_worker(args)
    else:
        result = {"schema_version": 1, "phase": args.phase,
                  "status": "prepared_not_measured" if args.prepare_only else "measured",
                  "identity": prepare_identity(args.dataset, args.cache_dir), "runs": []}
        if not args.prepare_only:
            environment = {k: v for k, v in os.environ.items() if not k.startswith("KNOWLEDGE_")}
            environment.update(KNOWLEDGE_RUN_REAL_MODELS="1", PYTHONPATH=str(ROOT / "src"), HF_HUB_OFFLINE="1",
                               TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
            for device in dict.fromkeys(args.devices):
                for component in FROZEN_MODELS:
                    for precision in ("default", "float32"):
                        child_output = args.output.parent / f"{args.output.stem}-{component}-{device}-{precision}.json"
                        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--phase", "baseline", "--worker",
                                        "--output", str(child_output), "--dataset", str(args.dataset.resolve()),
                                        "--cache-dir", str(args.cache_dir.resolve()), "--component", component,
                                        "--precision", precision, "--device", device],
                                       cwd=ROOT, env=environment, check=True)
                        result["runs"].append(json.loads(child_output.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
