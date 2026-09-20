"""Daemon server lifecycle and SSE application runner."""

import asyncio
import atexit
from dataclasses import asdict
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any
import urllib.request

from .config import Settings
from .server import create_application
from .state import index_lock


def get_daemon_pid(runtime_dir: Path) -> int | None:
    """Read the daemon PID from daemon.pid if it exists."""
    pid_path = runtime_dir / "daemon.pid"
    if pid_path.is_file():
        try:
            return int(pid_path.read_text(encoding="utf-8").strip())
        except (ValueError, OSError):
            return None
    return None


def is_daemon_running(port: int = 8765, host: str = "127.0.0.1") -> bool:
    """Check if the daemon is responding on /health or /sse."""
    for path in ("/health", "/sse"):
        try:
            req = urllib.request.Request(
                f"http://{host}:{port}{path}",
                headers={"Accept": "text/event-stream, application/json, */*"},
            )
            with urllib.request.urlopen(req, timeout=1.0) as response:
                if response.status == 200:
                    return True
        except Exception:
            continue
    return False


def start_daemon_process(
    settings: Settings | None = None,
    port: int = 8765,
    host: str = "127.0.0.1",
    timeout: float = 60.0,
) -> int:
    """Start the daemon in a background process and wait for readiness."""
    if is_daemon_running(port=port, host=host):
        pid = get_daemon_pid(settings.runtime_dir) if settings else None
        return pid or 0

    if settings is None:
        settings = Settings.from_env("codex")

    settings.runtime_dir.mkdir(parents=True, exist_ok=True)
    log_path = settings.runtime_dir / "daemon.log"

    python_exe = sys.executable
    if sys.platform == "win32":
        pythonw = Path(sys.executable).parent / "pythonw.exe"
        if pythonw.is_file():
            python_exe = str(pythonw)

    cmd = [
        python_exe,
        "-m",
        "knowledge_mcp.cli",
        "daemon",
        "run",
        "--host",
        host,
        "--port",
        str(port),
        "--client",
        settings.client_name,
    ]

    env = os.environ.copy()
    env["KNOWLEDGE_VAULT_ROOT"] = str(settings.vault_root)
    env["KNOWLEDGE_PROJECT_ROOT"] = str(settings.project_root)
    env["KNOWLEDGE_COLLECTION"] = settings.collection_name
    env["KNOWLEDGE_DENSE_MODEL"] = settings.dense_model

    creationflags = 0
    startupinfo = None
    if sys.platform == "win32":
        creationflags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE

    with open(log_path, "a", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            cmd,
            cwd=str(settings.project_root),
            stdout=log_file,
            stderr=log_file,
            stdin=subprocess.DEVNULL,
            creationflags=creationflags,
            startupinfo=startupinfo,
            env=env,
            close_fds=True,
        )

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if is_daemon_running(port=port, host=host):
            return proc.pid
        if proc.poll() is not None:
            raise RuntimeError(
                f"Daemon process exited prematurely with code {proc.returncode}. Check log at {log_path}"
            )
        time.sleep(0.5)

    raise TimeoutError(f"Daemon did not become healthy within {timeout} seconds on http://{host}:{port}/sse")


def stop_daemon_process(
    settings: Settings | None = None,
    port: int = 8765,
    host: str = "127.0.0.1",
    timeout: float = 5.0,
) -> bool:
    """Stop the background daemon process if running."""
    if settings is None:
        settings = Settings.from_env("codex")

    pid = get_daemon_pid(settings.runtime_dir)
    if pid is not None:
        try:
            if sys.platform == "win32":
                subprocess.run(
                    ["taskkill", "/F", "/PID", str(pid)],
                    check=False,
                    capture_output=True,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
            else:
                os.kill(pid, 15)  # SIGTERM
        except OSError:
            pass

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not is_daemon_running(port=port, host=host):
            pid_path = settings.runtime_dir / "daemon.pid"
            if pid_path.is_file():
                try:
                    pid_path.unlink()
                except OSError:
                    pass
            return True
        time.sleep(0.2)

    return not is_daemon_running(port=port, host=host)


def run_daemon(settings: Settings, host: str = "127.0.0.1", port: int = 8765) -> None:
    """Run the daemon server: ensure Qdrant, sync once, and start SSE FastMCP."""
    from .cli import _dependencies, ensure_qdrant

    pid_path = settings.runtime_dir / "daemon.pid"
    settings.runtime_dir.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(str(os.getpid()), encoding="utf-8")

    def cleanup_pid() -> None:
        if pid_path.is_file():
            try:
                pid_path.unlink()
            except OSError:
                pass

    atexit.register(cleanup_pid)

    try:
        ensure_qdrant(settings)
        store, operation_log, indexers = _dependencies(settings)

        async def sync_all(*, rebuild: bool = False) -> dict[str, Any]:
            if rebuild:
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
            return {model: asdict(summary) for model, summary in summaries.items()}

        # Indexing runs only ONCE upon daemon startup
        asyncio.run(sync_all())

        if hasattr(store, "reranker") and hasattr(store.reranker, "warmup"):
            asyncio.run(asyncio.to_thread(store.reranker.warmup))

        application = create_application(settings, store, operation_log)
        mcp = application.mcp

        @mcp.tool(
            name="knowledge-index-sync",
            description="Trigger synchronization of the local Vault index.",
        )
        async def knowledge_index_sync(rebuild: bool = False) -> dict[str, Any]:
            return await sync_all(rebuild=rebuild)

        @mcp.custom_route("/health", methods=["GET"])
        async def health(request):
            from starlette.responses import JSONResponse

            return JSONResponse({"status": "ok", "pid": os.getpid()})

        mcp.run(transport="sse", host=host, port=port)
    finally:
        cleanup_pid()
