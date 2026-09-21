"""Launcher: ensure Qdrant, synchronize the Vault, then run stdio MCP or SSE daemon."""

import argparse
import asyncio
from dataclasses import asdict, replace
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import urlopen

from .config import Settings
from .daemon import (
    get_daemon_pid,
    is_daemon_running,
    run_daemon,
    start_daemon_process,
    stop_daemon_process,
)
from .embeddings import create_embedding_provider
from .indexer import KnowledgeIndexer
from .qdrant_store import KnowledgeStore
from .retrieval import MultiModelStore
from .server import create_application
from .state import Manifest, OperationLog, index_lock


MODEL_COLLECTIONS = {
    "dragonkue/BGE-m3-ko": "obsidian_knowledge_bge_m3_ko_v1",
}

DEFAULT_DAEMON_HOST = os.environ.get("KNOWLEDGE_DAEMON_HOST", "127.0.0.1")
DEFAULT_DAEMON_PORT = int(os.environ.get("KNOWLEDGE_DAEMON_PORT", "8765"))


def model_settings(settings: Settings) -> dict[str, Settings]:
    """Keep each model's vectors and manifest in a dedicated collection."""
    collections = MODEL_COLLECTIONS if settings.collection_name == MODEL_COLLECTIONS[Settings.DEFAULT_DENSE_MODEL] else {
        Settings.DEFAULT_DENSE_MODEL: settings.collection_name,
    }
    return {
        model: replace(settings, dense_model=model, collection_name=collection)
        for model, collection in collections.items()
    }


def _dependencies(settings: Settings):
    log = OperationLog(settings.runtime_dir)
    stores = {}
    indexers = {}
    for model, selected in model_settings(settings).items():
        provider = create_embedding_provider(model)
        store = KnowledgeStore(selected, provider)
        stores[model] = store
        indexers[model] = KnowledgeIndexer(
            selected, store, provider.get_tokenizer(),
            manifest=Manifest(settings.runtime_dir, selected.collection_name), operation_log=log,
        )
    return MultiModelStore(stores), log, indexers


def ensure_qdrant(settings: Settings, *, timeout_seconds: int = 60) -> None:
    project_dir = settings.project_root
    compose = project_dir / "docker-compose.yml"
    settings.qdrant_storage_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["KNOWLEDGE_QDRANT_STORAGE"] = str(settings.qdrant_storage_dir.resolve()).replace("\\", "/")
    creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    subprocess.run(
        ["docker", "compose", "-f", str(compose), "up", "-d"],
        cwd=project_dir,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
        creationflags=creationflags,
    )
    deadline = time.monotonic() + timeout_seconds
    last_error = "not ready"
    while time.monotonic() < deadline:
        try:
            with urlopen(f"{settings.qdrant_url}/healthz", timeout=3) as response:
                if response.status < 400:
                    return
        except Exception as exc:
            last_error = str(exc)
        time.sleep(1)
    raise RuntimeError(f"Qdrant did not become healthy: {last_error}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="knowledge-mcp")
    subparsers = parser.add_subparsers(dest="command")

    # serve command
    serve_parser = subparsers.add_parser("serve", help="Run MCP stdio proxy or standalone server")
    serve_parser.add_argument("--client", choices=("codex", "claude-code", "antigravity"), default="codex")
    serve_parser.add_argument("--standalone", action="store_true", help="Run legacy in-process server without daemon")
    serve_parser.add_argument("--host", default=DEFAULT_DAEMON_HOST, help="Daemon host")
    serve_parser.add_argument("--port", type=int, default=DEFAULT_DAEMON_PORT, help="Daemon port")

    # daemon command
    daemon_parser = subparsers.add_parser("daemon", help="Manage background knowledge-mcp daemon")
    daemon_sub = daemon_parser.add_subparsers(dest="daemon_action")
    for action in ("start", "stop", "status", "run"):
        action_parser = daemon_sub.add_parser(action)
        action_parser.add_argument("--client", choices=("codex", "claude-code", "antigravity"), default="codex")
        action_parser.add_argument("--host", default=DEFAULT_DAEMON_HOST, help="Daemon host")
        action_parser.add_argument("--port", type=int, default=DEFAULT_DAEMON_PORT, help="Daemon port")

    # index and rebuild commands
    for command in ("index", "rebuild"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--client", choices=("codex", "claude-code", "antigravity"), default="codex")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    command = args.command or "serve"
    client_name = getattr(args, "client", "codex")
    settings = Settings.from_env(client_name)

    host = getattr(args, "host", DEFAULT_DAEMON_HOST)
    port = getattr(args, "port", DEFAULT_DAEMON_PORT)

    try:
        if command == "daemon":
            action = getattr(args, "daemon_action", None) or "status"
            if action == "start":
                if is_daemon_running(port=port, host=host):
                    pid = get_daemon_pid(settings.runtime_dir)
                    print(f"Daemon is already running on http://{host}:{port}/sse (PID: {pid})")
                    return 0
                pid = start_daemon_process(settings, port=port, host=host)
                print(f"Daemon started on http://{host}:{port}/sse (PID: {pid})")
                return 0
            elif action == "stop":
                stopped = stop_daemon_process(settings, port=port, host=host)
                if stopped:
                    print("Daemon stopped.")
                else:
                    print("Daemon was not running or failed to stop.")
                return 0
            elif action == "status":
                running = is_daemon_running(port=port, host=host)
                pid = get_daemon_pid(settings.runtime_dir)
                if running:
                    print(f"Daemon is running on http://{host}:{port}/sse (PID: {pid})")
                else:
                    print("Daemon is stopped.")
                return 0 if running else 1
            elif action == "run":
                run_daemon(settings, host=host, port=port)
                return 0
            else:
                print(f"Unknown daemon action: {action}", file=sys.stderr)
                return 1

        if command == "serve":
            if getattr(args, "standalone", False):
                # Legacy in-process server
                ensure_qdrant(settings)
                store, operation_log, indexers = _dependencies(settings)

                async def sync_all_standalone() -> bool:
                    summaries = {model: await indexer.sync() for model, indexer in indexers.items()}
                    for model, summary in summaries.items():
                        print(f"{model}: {asdict(summary)}", file=sys.stderr)
                    return all(summary.status == "completed" for summary in summaries.values())

                async def run_standalone() -> None:
                    await sync_all_standalone()
                    if hasattr(store, "reranker") and hasattr(store.reranker, "warmup"):
                        await asyncio.to_thread(store.reranker.warmup)
                    application = create_application(settings, store, operation_log)
                    application.mcp.run(transport="stdio")

                asyncio.run(run_standalone())
                return 0
            else:
                # Default mode: Ensure daemon is running and proxy stdio
                if not is_daemon_running(port=port, host=host):
                    start_daemon_process(settings, port=port, host=host)
                import anyio
                from .proxy import run_stdio_proxy

                anyio.run(run_stdio_proxy, f"http://{host}:{port}/sse", settings)
                return 0

        # index and rebuild
        ensure_qdrant(settings)
        store, operation_log, indexers = _dependencies(settings)

        async def sync_all(*, rebuild: bool = False) -> bool:
            if rebuild:
                # One writer lock covers collection removal and both fresh indexes.
                with index_lock(settings.runtime_dir):
                    for selected in store.stores.values():
                        if await selected.client.collection_exists(selected.settings.collection_name):
                            await selected.client.delete_collection(selected.settings.collection_name)
                    for indexer in indexers.values():
                        indexer.manifest.clear()
                    summaries = {
                        model: await indexer.sync(assume_locked=True)
                        for model, indexer in indexers.items()
                    }
            else:
                summaries = {model: await indexer.sync() for model, indexer in indexers.items()}
            for model, summary in summaries.items():
                print(f"{model}: {asdict(summary)}", file=sys.stderr)
            return all(summary.status == "completed" for summary in summaries.values())

        if command in {"index", "rebuild"}:
            return 0 if asyncio.run(sync_all(rebuild=command == "rebuild")) else 1

        return 0
    except Exception as exc:
        print(f"knowledge-mcp: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
