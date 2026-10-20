"""Frozen, offline baseline loader, independent of production precision policy.

Preparation never imports torch or sentence_transformers. Real inference requires
KNOWLEDGE_RUN_REAL_MODELS=1 and an exclusive measurement window. Each model,
device and precision combination runs in a fresh sequential child process.
Compare and owned-Qdrant execution use only frozen synthetic inputs and an
explicitly owned server. A failed or unmeasured gate remains HOLD.
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
import math
from urllib.parse import urlparse


def ranking_metrics(ranking, relevance):
    """Document judgments: collapse chunk duplicates before assigning ranks.

    Unjudged documents conventionally receive zero gain, not a new judgment.
    """
    ranking = list(dict.fromkeys(ranking))
    relevant = {key for key, gain in relevance.items() if gain > 0}
    gains = [relevance.get(key, 0) for key in ranking[:8]]
    dcg = sum((2 ** gain - 1) / math.log2(rank + 2) for rank, gain in enumerate(gains))
    ideal = sum((2 ** gain - 1) / math.log2(rank + 2)
                for rank, gain in enumerate(sorted(relevance.values(), reverse=True)[:8]))
    return {"recall20": len(set(ranking[:20]) & relevant) / len(relevant) if relevant else 0.0,
            "ndcg8": dcg / ideal if ideal else 0.0,
            "mrr8": next((1 / (rank + 1) for rank, gain in enumerate(gains) if gain > 0), 0.0)}


def paired_bootstrap(differences):
    import numpy as np
    # Frozen consecutive query pairs share one of 50 synthetic scenarios.
    clusters = np.array([sum(differences[i:i + 2]) / len(differences[i:i + 2])
                         for i in range(0, len(differences), 2)])
    rng = np.random.default_rng(20261020)
    means = clusters[rng.integers(0, len(clusters), (2000, len(clusters)))].mean(axis=1)
    return np.quantile(means, [0.025, 0.975]).tolist()


def compare_rankings(queries, baseline, candidate):
    import numpy as np
    if len(queries) != len(baseline) or len(queries) != len(candidate):
        raise ValueError("ranking coverage differs")
    old = [ranking_metrics(r, q["relevance"]) for q, r in zip(queries, baseline)]
    new = [ranking_metrics(r, q["relevance"]) for q, r in zip(queries, candidate)]
    report = {key: {"baseline": float(np.mean([m[key] for m in old])),
                    "candidate": float(np.mean([m[key] for m in new])),
                    "delta": float(np.mean([b[key] - a[key] for a, b in zip(old, new)])),
                    "paired_scenario_bootstrap95": paired_bootstrap([b[key] - a[key] for a, b in zip(old, new)])}
              for key in ("recall20", "ndcg8", "mrr8")}
    critical = lambda rows: sum(bool(q.get("critical")) and
                               not any(q["relevance"].get(d, 0) == 2 for d in list(dict.fromkeys(r))[:8])
                               for q, r in zip(queries, rows))
    report["critical_missing_baseline"] = critical(baseline)
    report["critical_missing_candidate"] = critical(candidate)
    report["critical_missing_increase"] = critical(candidate) - critical(baseline)
    report["top8_overlap_mean"] = float(np.mean([len(set(a[:8]) & set(b[:8])) / max(1, len(set(a[:8])))
                                                 for a, b in zip(baseline, candidate)]))
    report["top1_change_rate"] = float(np.mean([a[:1] != b[:1] for a, b in zip(baseline, candidate)]))
    report["pass"] = all(report[k]["delta"] >= -0.01 - 1e-12 and
                         report[k]["paired_scenario_bootstrap95"][0] >= -0.02 - 1e-12
                         for k in ("recall20", "ndcg8", "mrr8")) and report["critical_missing_increase"] <= 0
    return report


def require_comparable(first, second):
    for key in ("dataset_sha256", "models", "packages", "inference_contract", "python"):
        if first[key] != second[key]:
            raise ValueError(f"comparison identity mismatch: {key}")


def check_owned_facts(url, metadata, facts):
    parsed = urlparse(url)
    if (url != "http://127.0.0.1:36333" or url != metadata["url"] or parsed.port != 36333
            or facts["pid"] != metadata["pid"] or facts["port_pid"] != metadata["pid"]
            or str(Path(metadata["storage"])).casefold() not in facts["command"].casefold()
            or Path(metadata["executable"]).name.casefold() not in facts["command"].casefold()):
        raise ValueError("owned Qdrant endpoint/process/storage verification failed")


def verify_owned_qdrant(url, provenance):
    metadata = json.loads(Path(provenance).read_text(encoding="utf-8-sig"))
    # PID and port are integers from local metadata, never shell text.
    pid = int(metadata["pid"])
    command = (f"$p=Get-CimInstance Win32_Process -Filter 'ProcessId={pid}'; "
               "$c=Get-NetTCPConnection -LocalPort 36333 -State Listen; "
               "@{pid=$p.ProcessId;command=$p.CommandLine;port_pid=($c | Select-Object -First 1).OwningProcess} | ConvertTo-Json -Compress")
    facts = json.loads(subprocess.check_output(["powershell", "-NoProfile", "-Command", command], text=True))
    check_owned_facts(url, metadata, facts)
    from urllib.request import urlopen
    with urlopen(url, timeout=10) as response:
        server = json.load(response)
    if server.get("version") != "1.19.1":
        raise ValueError("owned Qdrant version must be 1.19.1")
    return {"metadata": metadata, "process": facts, "server": server}


def materialize_vault(root, documents):
    root.mkdir(parents=True, exist_ok=True)
    # Empty synthetic history prevents parse_source's real --follow query from
    # walking the unrelated enclosing application's Git history.
    if not (root / ".git").exists():
        subprocess.run(["git", "-c", "init.defaultBranch=precision-validation", "init", "--quiet", str(root)], check=True)
    for document in documents:
        path = root / document["source_path"]
        if not path.resolve().is_relative_to(root.resolve()) or path.suffix != ".md":
            raise ValueError("benchmark source must be Markdown inside owned Vault")
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {key: document[key] for key in ("document_type", "created_at", "modified_at")}
        metadata["security"] = document["security_level"]
        frontmatter = "\n".join(f"{key}: {json.dumps(value, ensure_ascii=False)}" for key, value in metadata.items())
        path.write_text(f"---\n{frontmatter}\n---\n{document['text']}\n", encoding="utf-8")


def vector_diagnostics(vectors):
    import numpy as np
    values = np.asarray(vectors, dtype=np.float64)
    shape_ok = values.ndim == 2 and values.shape[1] == 1024
    finite = bool(np.isfinite(values).all())
    zero_count = int(np.sum(~np.any(values != 0, axis=1))) if values.ndim == 2 else len(values)
    norm_error = float(np.max(np.abs(np.linalg.norm(values, axis=1) - 1))) if shape_ok and finite else None
    return {"shape": list(values.shape), "finite": finite, "zero_vectors": zero_count,
            "max_norm_error": norm_error,
            "pass": shape_ok and finite and zero_count == 0 and norm_error <= 0.01}


def vector_comparison(first, second):
    import numpy as np
    a, b = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("vector coverage differs")
    similarities = np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    mean, minimum = float(np.mean(similarities)), float(np.min(similarities))
    return {"cosine_mean": mean, "cosine_min": minimum, "pass": mean >= 0.999 and minimum >= 0.99}


def performance_gates(first, second):
    memory_reduction = 1 - second["steady_allocated_bytes"] / first["steady_allocated_bytes"]
    latency_change = second["warm_p95_ms"] / first["warm_p95_ms"] - 1
    throughput_change = second["throughput"] / first["throughput"] - 1
    samples = second["allocated_samples"]
    growth = max(samples) - samples[0]
    monotonic = all(b >= a for a, b in zip(samples, samples[1:])) and samples[-1] > samples[0]
    leak_ok = growth <= max(64 * 1024 ** 2, samples[0] * 0.05) and not monotonic
    return {"steady_allocated_reduction": memory_reduction, "warm_p95_change": latency_change,
            "throughput_change": throughput_change, "max_allocated_growth_bytes": growth,
            "monotonic_growth": monotonic, "leak_pass": leak_ok,
            "pass": memory_reduction >= 0.25 - 1e-12 and latency_change <= 0.1 + 1e-12
                    and throughput_change >= -0.05 - 1e-12
                    and leak_ok and second["fallback_count"] == 0}


def canonical_points(points):
    """Align full reindexes by source/chunk/body, not dtype-specific point IDs."""
    rows = []
    for point in points:
        metadata = {k: v for k, v in point["payload"]["metadata"].items() if k != "generation"}
        sparse = point["vector"]["bm25"]
        rows.append({"metadata": metadata, "body_sha256": sha256(point["payload"]["document"].encode()).hexdigest(),
                     "bm25": sorted(zip(sparse["indices"], sparse["values"]))})
    return sorted(rows, key=lambda row: (row["metadata"]["source_path"], row["metadata"]["chunk_index"], row["body_sha256"]))


def score_diagnostics(first, second):
    import numpy as np
    errors = []
    for a, b in zip(first, second):
        key = lambda row: (row["source_path"], row["document"])
        old, new = {key(row): row["score"] for row in a}, {key(row): row["score"] for row in b}
        if old.keys() != new.keys():
            raise ValueError("same-candidate scoring coverage differs")
        errors.extend(abs(old[key] - new[key]) for key in old)
    result = {"score_abs_error_mean": float(np.mean(errors)), "score_abs_error_max": float(max(errors))}
    for label, rows in (("baseline", first), ("candidate", second)):
        gaps = [abs(a["score"] - b["score"]) for row in rows for a, b in zip(row, row[1:])]
        result[label + "_adjacent_exact_ties"] = sum(gap == 0 for gap in gaps)
        result[label + "_adjacent_gaps_le_0_001"] = sum(gap <= 0.001 for gap in gaps)
        result[label + "_top1_top2_margin"] = [row[0]["score"] - row[1]["score"] for row in rows if len(row) > 1]
    return result


def kernel_probe(call):
    import torch
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA]) as profile:
        call()
    kernels = sorted({event.name for event in profile.events() if str(event.device_type) == "DeviceType.CUDA"})
    attention = sorted({event.name for event in profile.events() if "attention" in event.name.lower()})
    return {"status": "observed" if kernels else "CUDA_kernel_events_unavailable",
            "cuda_kernels": kernels, "attention_ops": attention}


def synchronize(device):
    if device == "cuda":
        import torch
        torch.cuda.synchronize()


def memory_snapshot(device):
    import torch
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (key, ctypes.c_size_t) for key in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        result = {"rss_bytes": counters.WorkingSetSize, "peak_rss_bytes": counters.PeakWorkingSetSize}
    else:
        import resource
        result = {"peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
    if device == "cuda":
        synchronize(device)
        result.update(allocated_bytes=torch.cuda.memory_allocated(), reserved_bytes=torch.cuda.memory_reserved(),
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
        try:
            result["nvidia_smi"] = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"], text=True).strip()
            result["process_vram"] = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"], text=True).strip()
        except (OSError, subprocess.CalledProcessError) as error:
            result["nvidia_smi_error"] = type(error).__name__
    return result


def model_evidence(model, component):
    target = model if component == "embedding" else model.model
    dtypes, devices = Counter(), Counter()
    for parameter in target.parameters():
        dtypes[str(parameter.dtype)] += parameter.numel()
        devices[str(parameter.device)] += parameter.numel()
    config = target[0].auto_model.config if component == "embedding" else target.config
    return {"parameter_dtypes": dict(dtypes), "parameter_devices": dict(devices),
            "attention_implementation": getattr(config, "_attn_implementation", None)}


def attach_forward_probe(model, component):
    """Observe actual linear activations without changing inference or autocast."""
    import torch
    target = model if component == "embedding" else model.model
    evidence = []
    def hook(module, inputs, output):
        if not evidence:
            evidence.append({"input_dtype": str(inputs[0].dtype), "output_dtype": str(output.dtype),
                             "device": str(output.device), "module": type(module).__name__})
    layer = next(module for module in target.modules() if isinstance(module, torch.nn.Linear))
    return evidence, layer.register_forward_hook(hook)


def production_models(args, settings):
    from unittest.mock import patch
    import torch
    from knowledge_mcp.embeddings import LocalSentenceTransformerProvider
    from knowledge_mcp.reranker import LocalReranker
    # Production constructors and inference paths, with immutable cached snapshots
    # in place of remote aliases. Keep vector-name identity equal to production.
    provider = LocalSentenceTransformerProvider(FROZEN_MODELS["embedding"]["id"], settings=settings)
    provider.model_name = str(cached_snapshot(FROZEN_MODELS["embedding"], args.cache_dir))
    reranker = LocalReranker(settings=settings)
    reranker.model_name = str(cached_snapshot(FROZEN_MODELS["reranker"], args.cache_dir))
    with patch.object(torch.cuda, "is_available", return_value=args.device == "cuda"):
        provider.warmup()
        embedding_memory = memory_snapshot(args.device)
        reranker.warmup()
    provider.model_name = FROZEN_MODELS["embedding"]["id"]
    reranker.model_name = FROZEN_MODELS["reranker"]["id"]
    return provider, reranker, embedding_memory


def runtime_evidence(device):
    import torch
    return {"device": device, "cuda_version": torch.version.cuda, "torch_version": torch.__version__,
            "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
            "bf16_supported": torch.cuda.is_bf16_supported() if device == "cuda" else None,
            "threads": torch.get_num_threads(), "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "tf32_cudnn": torch.backends.cudnn.allow_tf32,
            "allocator_config": os.environ.get("PYTORCH_CUDA_ALLOC_CONF")}


def filter_violations(query, rows):
    from datetime import datetime
    from knowledge_mcp.documents import SEOUL
    filters = query.get("filters", {})
    count = 0
    for row in rows:
        invalid = row["security_level"] == "private" and not filters.get("include_private", False)
        for field in ("document_type", "file_type"):
            invalid |= field in filters and row[field] not in filters[field]
        for field in ("created", "modified"):
            if field + "_from" not in filters and field + "_to" not in filters:
                continue
            day = datetime.fromisoformat(row[field + "_at"]).astimezone(SEOUL).date().isoformat()
            invalid |= field + "_from" in filters and day < filters[field + "_from"]
            invalid |= field + "_to" in filters and day > filters[field + "_to"]
        count += bool(invalid)
    return count


def benchmark_settings(args, vault, collection):
    from knowledge_mcp.config import Settings
    return Settings(vault, args.run_root / "state", Settings.DEFAULT_QDRANT_URL, collection,
                    Settings.DEFAULT_DENSE_MODEL, "precision-validation", project_root=args.run_root / "project",
                    inference_precision=args.precision)


def score_candidates(reranker, query, candidates):
    import asyncio
    return asyncio.run(reranker.rerank(query, candidates))


def production_worker(args):
    import asyncio
    import numpy as np
    import torch
    from knowledge_mcp.search import SearchResult
    documents, queries = load_dataset(args.dataset)
    if args.smoke:
        documents, queries = [documents[0], documents[-1]], [queries[0], queries[-1]]
    settings = benchmark_settings(args, args.run_root / "vault", "unused")
    provider, reranker, single_memory = production_models(args, settings)
    evidence, embedding_hook = attach_forward_probe(provider.model, "embedding")
    reranker_evidence, reranker_hook = attach_forward_probe(reranker._model, "reranker")
    started = time.perf_counter()
    vectors = asyncio.run(provider.embed_documents([d["text"] for d in documents] + [q["text"] for q in queries]))
    synchronize(args.device)
    corpus_seconds = time.perf_counter() - started
    by_id = {d["id"]: d for d in documents}
    pair_ids = [(q["id"], did) for q in queries for did in q["relevance"] if did in by_id]
    by_query = {q["id"]: q for q in queries}
    def candidates(ids):
        return [SearchResult(point_id=did, document=by_id[did]["text"], score=0,
                             **{key: by_id[did].get(key, []) for key in
                                ("source_path", "document_type", "file_type", "security_level", "created_at", "modified_at")})
                for did in ids]
    scores = []
    for query in queries:
        rows = candidates([did for did in query["relevance"] if did in by_id])
        ranked = score_candidates(reranker, query["text"], rows)
        by_score = {r.point_id: r.score for r in ranked}
        scores.extend(by_score[r.point_id] for r in rows)
    # Batch-one boundary uses production contracts without trimming long input.
    original = provider.batch_size
    provider.batch_size = 1
    batch1 = asyncio.run(provider.embed_documents([queries[-1]["text"]]))
    provider.batch_size = original
    original_rerank_batch = reranker.batch_size
    reranker.batch_size = 1
    batch1_scores = score_candidates(reranker, queries[-1]["text"], candidates([documents[-1]["id"]]))
    reranker.batch_size = original_rerank_batch
    for _ in range(5):
        asyncio.run(provider.embed_query(queries[0]["text"]))
        score_candidates(reranker, queries[0]["text"], candidates([documents[0]["id"]]))
    synchronize(args.device)
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start_memory = memory_snapshot(args.device)
    timings, allocated = [], [start_memory.get("allocated_bytes", 0)]
    embedding_timings, reranker_timings = [], []
    # CPU is explicitly bounded functional coverage; CUDA gets all 100*5.
    for repeat in range(1 if args.smoke else 5):
        for query in queries:
            synchronize(args.device)
            started = time.perf_counter()
            asyncio.run(provider.embed_query(query["text"]))
            synchronize(args.device)
            midpoint = time.perf_counter()
            score_candidates(reranker, query["text"], candidates([did for did in query["relevance"] if did in by_id]))
            synchronize(args.device)
            timings.append((time.perf_counter() - started) * 1000)
            embedding_timings.append((midpoint - started) * 1000)
            reranker_timings.append((time.perf_counter() - midpoint) * 1000)
            allocated.append(torch.cuda.memory_allocated() if args.device == "cuda" else 0)
    result = {"component": "production_both", "requested_precision": args.precision, "requested_device": args.device,
              "runtime": runtime_evidence(args.device), "embedding": model_evidence(provider.model, "embedding"),
              "reranker": model_evidence(reranker._model, "reranker"), "embedding_forward": evidence,
              "reranker_forward": reranker_evidence, "embedding_only_memory": single_memory,
              "steady_memory": start_memory, "end_memory": memory_snapshot(args.device),
              "steady_allocated_bytes": start_memory.get("allocated_bytes", 0),
              "warm_p95_ms": float(np.percentile(timings, 95)), "warm_timings_ms": timings,
              "embedding_timings_ms": embedding_timings, "reranker_timings_ms": reranker_timings,
              "throughput": len(vectors) / corpus_seconds, "corpus_seconds": corpus_seconds,
              "allocated_samples": allocated, "fallback_count": int(reranker.status["last_fallback"] is not None and args.device == "cuda"),
              "batch1_diagnostics": vector_diagnostics(batch1), "smoke": args.smoke,
              "batch1_reranker_score": batch1_scores[0].score,
              "input_lengths": {"embedding_max_seq_length": provider.model.max_seq_length,
                  "reranker_max_length": reranker._model.max_seq_length,
                  "max_document_tokens": max(len(provider.model.tokenizer.encode(d["text"])) for d in documents),
                  "max_query_tokens": max(len(provider.model.tokenizer.encode(q["text"])) for q in queries),
                  "full_context_boundary_tested": False},
              "outputs": {"document_ids": [d["id"] for d in documents], "query_ids": [q["id"] for q in queries],
                          "document_vectors": vectors[:len(documents)], "query_vectors": vectors[len(documents):],
                          "pair_ids": pair_ids, "scores": scores}}
    embedding_hook.remove()
    reranker_hook.remove()
    return result


async def qdrant_worker(args):
    from dataclasses import asdict
    from unittest.mock import patch
    import numpy as np
    import torch
    from qdrant_client import AsyncQdrantClient
    from knowledge_mcp.config import Settings
    from knowledge_mcp.documents import parse_source, chunk_document
    from knowledge_mcp.indexer import KnowledgeIndexer
    from knowledge_mcp.qdrant_store import KnowledgeStore
    from knowledge_mcp.retrieval import MultiModelStore
    from knowledge_mcp.search import SearchRequest, SearchResult
    ownership = verify_owned_qdrant(args.qdrant_url, args.ownership)
    documents, queries = load_dataset(args.dataset)
    vault = args.run_root / "vault"
    materialize_vault(vault, documents)
    collection = "precision_" + args.run_root.name.replace("-", "_") + "_" + args.precision
    settings = benchmark_settings(args, vault, collection)
    provider, reranker, single_memory = production_models(args, settings)
    embedding_evidence, embedding_hook = attach_forward_probe(provider.model, "embedding")
    reranker_evidence, reranker_hook = attach_forward_probe(reranker._model, "reranker")
    # Same injection used by integration tests, keeping the production URL guard.
    with patch("knowledge_mcp.qdrant_store.AsyncQdrantClient",
               lambda **kwargs: AsyncQdrantClient(url=args.qdrant_url, cloud_inference=True, timeout=120)):
        store = KnowledgeStore(settings, provider)
    try:
        if args.smoke:
            reference = json.loads(args.reference.read_text(encoding="utf-8"))
            if collection != reference["collection"] or not await store.client.collection_exists(collection):
                raise ValueError("CPU read requires the measured full BF16 collection")
            multi = MultiModelStore({settings.dense_model: store}, reranker)
            rows = []
            for query in [queries[0], next(q for q in queries if q.get("filters"))]:
                ranked = await multi.hybrid_search(SearchRequest(query=query["text"], **query.get("filters", {})))
                values = [r.model_dump(mode="json") for r in ranked]
                if not values or filter_violations(query, values) or any(not r.rerank_applied for r in ranked):
                    raise ValueError("CPU query into full BF16 index failed")
                rows.append(values)
            return {"collection": collection, "coverage": "2 actual CPU embedding+reranker queries into full BF16 index (ordinary + filtered)",
                    "rows": rows, "embedding": model_evidence(provider.model, "embedding"),
                    "reranker": model_evidence(reranker._model, "reranker"), "embedding_forward": embedding_evidence,
                    "reranker_forward": reranker_evidence, "runtime": runtime_evidence("cpu")}
        tokenizer = provider.get_tokenizer()
        expected_chunks = [chunk for document in documents
                           for chunk in chunk_document(parse_source(vault / document["source_path"], settings), tokenizer)]
        indexer = KnowledgeIndexer(settings, store, tokenizer, embedding_precision=provider.effective_dtype)
        synchronize(args.device)
        started = time.perf_counter()
        summary = await indexer.sync()
        synchronize(args.device)
        index_seconds = time.perf_counter() - started
        unchanged = await indexer.sync()
        manifest = indexer.manifest.completed_files()
        points, offset = [], None
        while True:
            page, offset = await store.client.scroll(collection, limit=256, offset=offset, with_vectors=True, with_payload=True)
            points.extend(point.model_dump(mode="json") for point in page)
            if offset is None:
                break
        points.sort(key=lambda p: (p["payload"]["metadata"]["source_path"], p["payload"]["metadata"]["chunk_index"]))
        expected = sorted((chunk.source_path, chunk.chunk_index, sha256(chunk.document.encode()).hexdigest()) for chunk in expected_chunks)
        actual = sorted((p["payload"]["metadata"]["source_path"], p["payload"]["metadata"]["chunk_index"],
                         sha256(p["payload"]["document"].encode()).hexdigest()) for p in points)
        complete = (summary.failed == summary.skipped == 0 and summary.added == len(documents)
                    and len(manifest) == len(documents) and len(points) == len(expected_chunks)
                    and expected == actual and unchanged.unchanged == len(documents) and unchanged.failed == 0
                    and all(len(entry.point_ids) == entry.point_count for entry in manifest.values())
                    and sum(entry.point_count for entry in manifest.values()) == len(points))
        multi = MultiModelStore({settings.dense_model: store}, reranker)
        requests = [SearchRequest(query=q["text"], **q.get("filters", {}), limit=8) for q in queries]
        rows, candidates, rrf_rows = [], [], []
        violations = fallback = 0
        for query, request in zip(queries, requests):
            candidate = await store.hybrid_candidates(request)
            candidates.append([r.model_dump(mode="json") for r in candidate])
            rrf = await multi.hybrid_search(request.model_copy(update={"rerank": False, "limit": 20}))
            ranked = await multi.hybrid_search(request)
            rows.append([r.model_dump(mode="json") for r in ranked])
            rrf_rows.append([r.model_dump(mode="json") for r in rrf])
            violations += filter_violations(query, [r.model_dump() for r in candidate + ranked + rrf])
            fallback += sum(not r.rerank_applied or r.rerank_error is not None for r in ranked)
        reference = json.loads(args.reference.read_text(encoding="utf-8")) if args.reference else None
        shared_candidates = reference["candidates"] if reference else candidates
        same_candidate_rows = []
        for query, frozen in zip(queries, shared_candidates):
            ranked = await reranker.rerank(query["text"], [SearchResult.model_validate(r) for r in frozen])
            same_candidate_rows.append([r.model_dump(mode="json") for r in ranked])
        # Profile one representative pair outside warm timing/peak windows.
        kernel_evidence = {
            "embedding": kernel_probe(lambda: provider._encode([queries[0]["text"]])),
            "reranker": kernel_probe(lambda: reranker._score(queries[0]["text"],
                                      [SearchResult.model_validate(row) for row in candidates[0][:1]]))}
        for _ in range(5):
            await multi.hybrid_search(requests[0])
        synchronize(args.device)
        torch.cuda.reset_peak_memory_stats()
        start_memory = memory_snapshot(args.device)
        timings, allocated = [], [torch.cuda.memory_allocated()]
        for _ in range(5):
            for request in requests:
                synchronize(args.device)
                started = time.perf_counter()
                ranked = await multi.hybrid_search(request)
                synchronize(args.device)
                timings.append((time.perf_counter() - started) * 1000)
                allocated.append(torch.cuda.memory_allocated())
                fallback += sum(not r.rerank_applied or r.rerank_error is not None for r in ranked)
        info = await store.client.get_collection(collection)
        result = {"collection": collection, "ownership": ownership, "index_summary": asdict(summary),
                  "same_dtype_sync_summary": asdict(unchanged), "expected_sources": len(documents),
                  "manifest_sources": len(manifest), "expected_points": len(expected_chunks), "actual_points": len(points),
                  "complete": complete, "index_seconds": index_seconds, "throughput": len(points) / index_seconds,
                  "schema": {"config": info.config.model_dump(mode="json"), "payload_schema": {k: v.model_dump(mode="json") for k, v in info.payload_schema.items()}},
                  "points": points, "candidates": candidates, "rerank_false": rrf_rows, "end_to_end": rows,
                  "same_candidate": same_candidate_rows, "filter_violations": violations, "fallback_count": fallback,
                  "embedding": model_evidence(provider.model, "embedding"), "reranker": model_evidence(reranker._model, "reranker"),
                  "embedding_forward": embedding_evidence, "reranker_forward": reranker_evidence,
                  "runtime": runtime_evidence(args.device), "steady_memory": start_memory,
                  "kernel_evidence": kernel_evidence,
                  "embedding_only_memory": single_memory, "end_memory": memory_snapshot(args.device),
                  "steady_allocated_bytes": start_memory["allocated_bytes"], "allocated_samples": allocated,
                  "warm_p95_ms": float(np.percentile(timings, 95)), "warm_timings_ms": timings,
                  "vector_diagnostics": vector_diagnostics([p["vector"]["dense"] for p in points])}
        embedding_hook.remove()
        reranker_hook.remove()
        return result
    finally:
        await store.client.close()


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
    forwards, hook = attach_forward_probe(model, args.component)
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
    inference_seconds = time.perf_counter() - started
    def query_call(query):
        if args.component == "embedding":
            return model.encode([query["text"]], batch_size=8, normalize_embeddings=True)
        return model.predict([(query["text"], by_id[did]["text"]) for did in query["relevance"]], batch_size=8)
    for _ in range(5):
        query_call(queries[0])
    timings = []
    for _ in range(5):
        for query in queries:
            synchronize(args.device)
            started = time.perf_counter()
            query_call(query)
            synchronize(args.device)
            timings.append((time.perf_counter() - started) * 1000)
    hook.remove()
    return {"component": args.component, "requested_precision": args.precision, "requested_device": args.device,
            "parameter_dtypes": dict(dtypes), "parameter_devices": dict(devices),
            "bf16_supported": torch.cuda.is_bf16_supported() if args.device == "cuda" else None,
            "load_seconds": load_seconds, "inference_seconds": inference_seconds,
            "runtime": runtime_evidence(args.device), "forward": forwards,
            "memory": memory_snapshot(args.device), "warm_timings_ms": timings,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else None,
            "outputs": outputs}


def dtype_gate(run, dtype, device):
    expected = "torch." + dtype
    return all(set(run[component]["parameter_dtypes"]) == {expected}
               and all(key.startswith(device) for key in run[component]["parameter_devices"])
               and run[component + "_forward"]
               and all(probe["output_dtype"] == expected and probe["device"].startswith(device)
                       for probe in run[component + "_forward"])
               for component in ("embedding", "reranker"))


def compare_outputs(runs, queries):
    import numpy as np
    baseline = [run for run in runs if run.get("component") != "production_both"]
    production = [run for run in runs if run.get("component") == "production_both"]
    parity = {}
    for component in FROZEN_MODELS:
        pair = [r for r in baseline if r["component"] == component]
        key = "document_vectors" if component == "embedding" else "scores"
        error = float(np.max(np.abs(np.asarray(pair[0]["outputs"][key]) - np.asarray(pair[1]["outputs"][key]))))
        query_error = 0.0
        if component == "embedding":
            delta = np.asarray(pair[0]["outputs"]["query_vectors"]) - np.asarray(pair[1]["outputs"]["query_vectors"])
            query_error = float(np.max(np.abs(delta)))
        parity[component] = {"max_abs_error": error, "query_max_abs_error": query_error,
                             "pass": error <= 1e-6 and query_error <= 1e-6}
    first, second, cpu = production
    vector = vector_comparison(first["outputs"]["document_vectors"] + first["outputs"]["query_vectors"],
                               second["outputs"]["document_vectors"] + second["outputs"]["query_vectors"])
    old_embedding = next(r for r in baseline if r["component"] == "embedding" and r["requested_precision"] == "float32")
    old_reranker = next(r for r in baseline if r["component"] == "reranker" and r["requested_precision"] == "float32")
    bridge = {"embedding": vector_comparison(old_embedding["outputs"]["document_vectors"] + old_embedding["outputs"]["query_vectors"],
                                            first["outputs"]["document_vectors"] + first["outputs"]["query_vectors"]),
              "reranker_max_abs_error": float(np.max(np.abs(np.array(old_reranker["outputs"]["scores"]) - first["outputs"]["scores"])))}
    for component, old in (("embedding", old_embedding), ("reranker", old_reranker)):
        previous_p95 = float(np.percentile(old["warm_timings_ms"], 95))
        override_p95 = float(np.percentile(first[component + "_timings_ms"], 95))
        bridge[component + "_warm_p95_ms"] = {"old_explicit": previous_p95, "production_override": override_p95,
                                               "change": override_p95 / previous_p95 - 1}
    score_error = np.abs(np.array(first["outputs"]["scores"]) - second["outputs"]["scores"])
    return {"old_default_vs_explicit_fp32": parity, "old_explicit_vs_production_override": bridge,
            "production_fp32_vs_auto_vectors": vector,
            "reranker_score_abs_error": {"mean": float(score_error.mean()), "max": float(score_error.max())},
            "production_dtype_pass": dtype_gate(first, "float32", "cuda") and dtype_gate(second, "bfloat16", "cuda"),
            "cpu_functional_dtype_pass": dtype_gate(cpu, "float32", "cpu") and cpu["batch1_diagnostics"]["pass"],
            "performance": performance_gates(first, second)}


def qdrant_comparison(first, second, documents, queries):
    require_comparable(first["identity"], second["identity"])
    if first["runtime"] != second["runtime"]:
        raise ValueError("comparison identity mismatch: actual execution contract")
    by_path = {document["source_path"]: document["id"] for document in documents}
    def rankings(run, key):
        return [list(dict.fromkeys(by_path[row["source_path"]] for row in rows)) for rows in run[key]]
    quality = {key: compare_rankings(queries, rankings(first, key), rankings(second, key))
               for key in ("candidates", "rerank_false", "same_candidate", "end_to_end")}
    # Final limit8 is evaluated at8; candidate recall@20 is separately measured
    # before source cap/limit and never inferred from the returned8.
    for run in (first, second):
        run["quality_absolute"] = {key: compare_rankings(queries, rankings(run, key), rankings(run, key))
                                   for key in ("candidates", "end_to_end")}
    vectors = vector_comparison([p["vector"]["dense"] for p in first["points"]],
                               [p["vector"]["dense"] for p in second["points"]])
    semantics = canonical_points(first["points"]) == canonical_points(second["points"])
    # Collection metadata includes the same Vault path for the two fresh arms.
    schema_equal = first["schema"] == second["schema"]
    performance = performance_gates(first, second)
    dtype_ok = dtype_gate(first, "float32", "cuda") and dtype_gate(second, "bfloat16", "cuda")
    passed = (first["complete"] and second["complete"] and semantics and schema_equal and vectors["pass"]
              and first["vector_diagnostics"]["pass"] and second["vector_diagnostics"]["pass"]
              and dtype_ok and not first["filter_violations"] and not second["filter_violations"]
              and not first["fallback_count"] and not second["fallback_count"]
              and all(value["pass"] for value in quality.values()) and performance["pass"])
    return {"status": "PASS" if passed else "HOLD", "quality": quality,
            "same_candidate_scores": score_diagnostics(first["same_candidate"], second["same_candidate"]),
            "chunk_vector_comparison": vectors, "payload_bm25_semantics_equal": semantics,
            "schema_equal": schema_equal, "dtype_pass": dtype_ok, "performance": performance}


def execute_child(args, environment, phase, output, extra):
    command = [sys.executable, str(Path(__file__).resolve()), "--phase", phase, "--worker",
               "--output", str(output), "--dataset", str(args.dataset.resolve()),
               "--cache-dir", str(args.cache_dir.resolve()), *extra]
    print(json.dumps({"event": "worker_start", "command": command}), flush=True)
    # No generic timeout or arbitrary Python cleanup: the controller owns this
    # sequential benchmark tree and may stop only that tree if needed.
    try:
        subprocess.run(command, cwd=ROOT, env=environment, check=True)
    except subprocess.CalledProcessError as error:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"status": "HOLD", "failed_command": command,
                                           "failed_returncode": error.returncode,
                                           "worker_output": str(output)}, indent=2) + "\n", encoding="utf-8")
        raise
    print(json.dumps({"event": "worker_complete", "output": str(output)}), flush=True)
    result = json.loads(output.read_text(encoding="utf-8"))
    require_comparable(prepare_identity(args.dataset, args.cache_dir), result["identity"])
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("baseline", "compare", "qdrant"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--cache-dir", type=Path, default=Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub")))
    parser.add_argument("--qdrant-url", help="owned test endpoint only: http://127.0.0.1:36333")
    parser.add_argument("--ownership", type=Path, default=ROOT / ".superpowers/sdd/2026-10-20-bf16-inference-migration/qdrant-owned.json")
    parser.add_argument("--run-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--reference", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--smoke", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--prepare-only", action="store_true", help="freeze identity only; no model imports or inference")
    parser.add_argument("--devices", nargs="+", choices=("cpu", "cuda"), default=["cpu", "cuda"],
                        help="baseline devices only; compare/qdrant require CUDA plus bounded CPU smoke")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--component", choices=tuple(FROZEN_MODELS), help=argparse.SUPPRESS)
    parser.add_argument("--precision", choices=("default", "float32", "auto"), help=argparse.SUPPRESS)
    parser.add_argument("--device", choices=("cpu", "cuda"), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if (args.worker or not args.prepare_only) and os.environ.get("KNOWLEDGE_RUN_REAL_MODELS") != "1":
        parser.error("real inference requires KNOWLEDGE_RUN_REAL_MODELS=1 and an exclusive measurement window")
    if args.worker:
        if not all((args.precision, args.device)) or (args.phase == "baseline" and not args.component):
            parser.error("worker requires component, precision and device")
        from knowledge_mcp import config, documents
        # Real workers are as hermetic as the fake tests: no production .env.
        config._load_env_file = documents._load_env_file = lambda *a, **kw: None
        if args.phase == "baseline":
            result = run_worker(args)
        elif args.phase == "compare":
            result = production_worker(args)
        else:
            import asyncio
            result = asyncio.run(qdrant_worker(args))
        result["identity"] = prepare_identity(args.dataset, args.cache_dir)
    else:
        result = {"schema_version": 2, "phase": args.phase,
                  "status": "prepared_not_measured" if args.prepare_only else "measured",
                  "identity": prepare_identity(args.dataset, args.cache_dir), "runs": []}
        if not args.prepare_only:
            environment = {k: v for k, v in os.environ.items() if not k.startswith("KNOWLEDGE_")}
            environment.update(KNOWLEDGE_RUN_REAL_MODELS="1", PYTHONPATH=str(ROOT / "src"), HF_HUB_OFFLINE="1",
                               TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            if args.phase != "baseline":
                from uuid import uuid4
                args.run_root = args.output.parent.resolve() / ("run_" + uuid4().hex[:12])
                args.run_root.mkdir()
                result["run_root"] = str(args.run_root)
                result["evaluation_limitations"] = ["synthetic corpus only; production corpus quality unverified",
                    "196 unjudged documents per query conventionally receive zero gain",
                    "100 queries represent 50 correlated scenarios; paired cluster bootstrap2000 seed20261020",
                    "Windows installed environment only; Linux and ST3/4 combinations unexecuted"]
                if args.phase == "compare":
                    for component in FROZEN_MODELS:
                        for precision in ("default", "float32"):
                            output = args.run_root / f"old-{component}-{precision}.json"
                            result["runs"].append(execute_child(args, environment, "baseline", output,
                                ["--component", component, "--precision", precision, "--device", "cuda"]))
                    for precision, device, smoke in (("float32", "cuda", False), ("auto", "cuda", False), ("auto", "cpu", True)):
                        output = args.run_root / f"production-{device}-{precision}.json"
                        extra = ["--precision", precision, "--device", device, "--run-root", str(args.run_root)]
                        if smoke:
                            extra.append("--smoke")
                        result["runs"].append(execute_child(args, environment, "compare", output, extra))
                    result["comparison"] = compare_outputs(result["runs"], load_dataset(args.dataset)[1])
                    result["status"] = "HOLD_pending_full_qdrant_gate"
                else:
                    result["ownership"] = verify_owned_qdrant(args.qdrant_url, args.ownership)
                    for precision in ("float32", "auto"):
                        output = args.run_root / f"qdrant-{precision}.json"
                        extra = ["--precision", precision, "--device", "cuda", "--run-root", str(args.run_root),
                                 "--qdrant-url", args.qdrant_url, "--ownership", str(args.ownership)]
                        if precision == "auto":
                            extra.extend(["--reference", str(args.run_root / "qdrant-float32.json")])
                        result["runs"].append(execute_child(args, environment, "qdrant", output, extra))
                    result["comparison"] = qdrant_comparison(*result["runs"], *load_dataset(args.dataset))
                    cpu = execute_child(args, environment, "qdrant", args.run_root / "qdrant-cpu-read.json",
                        ["--precision", "auto", "--device", "cpu", "--smoke", "--run-root", str(args.run_root),
                         "--qdrant-url", args.qdrant_url, "--ownership", str(args.ownership),
                         "--reference", str(args.run_root / "qdrant-auto.json")])
                    result["cpu_read"] = cpu
                    result["comparison"]["cpu_read_pass"] = dtype_gate(cpu, "float32", "cpu")
                    if not result["comparison"]["cpu_read_pass"]:
                        result["comparison"]["status"] = "HOLD"
                    result["status"] = result["comparison"]["status"]
                args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                return
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
