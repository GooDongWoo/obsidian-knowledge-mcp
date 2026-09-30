"""Daemon lifecycle and Streamable HTTP application runner."""

import asyncio
import atexit
import math
from dataclasses import asdict
from contextlib import contextmanager
import io
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any
import urllib.request

from .config import Settings
from .server import create_application
from .state import index_lock, positive_env_int


_EVENT_CODES = {"runtime_message", "runtime_stdout", "runtime_stderr", "daemon_starting", "daemon_failed"}


class _EventFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        code = getattr(record, "event_code", "runtime_message")
        if not isinstance(code, str) or code not in _EVENT_CODES:
            code = "runtime_message"
        # Never format the message, arguments, logger name or traceback: any
        # of them may include a query, returned document or filesystem path.
        return f"{self.formatTime(record)} {logging.getLevelName(record.levelno)} {code}"


class _EventStream(io.TextIOBase):
    def __init__(self, handler: logging.Handler, code: str, console=None) -> None:
        self.handler = handler
        self.code = code
        self.console = console

    @property
    def encoding(self) -> str:
        return "utf-8"

    def write(self, text: str) -> int:
        if text.strip():
            record = logging.LogRecord("daemon", logging.INFO, "", 0, "", (), None)
            record.event_code = self.code
            self.handler.handle(record)
        if self.console is not None:
            self.console.write(text)
        return len(text)

    def flush(self) -> None:
        self.handler.flush()
        if self.console is not None:
            self.console.flush()


@contextmanager
def daemon_logging(runtime_dir: Path, *, console: bool = False):
    """Bound logs in the child process, including writes from third parties."""
    runtime_dir.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        runtime_dir / "daemon.log", encoding="utf-8",
        maxBytes=positive_env_int("KNOWLEDGE_DAEMON_LOG_MAX_BYTES", 1048576),
        backupCount=positive_env_int("KNOWLEDGE_DAEMON_LOG_BACKUP_COUNT", 3),
    )
    handler.setFormatter(_EventFormatter())
    root = logging.getLogger()
    loggers = [root] + [logger for logger in logging.root.manager.loggerDict.values()
                        if isinstance(logger, logging.Logger)]
    previous = [(logger, logger.handlers[:], logger.propagate) for logger in loggers]
    stdout, stderr = sys.stdout, sys.stderr
    for logger in loggers:
        logger.handlers = []
        logger.propagate = True
    root.addHandler(handler)
    if console:
        root.addHandler(logging.StreamHandler(stderr))
    sys.stdout = _EventStream(handler, "runtime_stdout", stdout if console else None)
    sys.stderr = _EventStream(handler, "runtime_stderr", stderr if console else None)
    try:
        yield
    finally:
        sys.stdout, sys.stderr = stdout, stderr
        for logger, handlers, propagate in previous:
            logger.handlers = handlers
            logger.propagate = propagate
        # Libraries may configure new loggers while running. Detach handlers
        # bound to our temporary streams before returning to an embedding caller.
        previous_loggers = {logger for logger, _, _ in previous}
        for logger in logging.root.manager.loggerDict.values():
            if isinstance(logger, logging.Logger) and logger not in previous_loggers:
                logger.handlers = [item for item in logger.handlers
                                   if not isinstance(getattr(item, "stream", None), _EventStream)]
        handler.close()


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
    """Check readiness without a protocol ping or a persistent event stream."""
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/health", timeout=1.0) as response:
            return response.status == 200
    except Exception:
        return False


def start_daemon_process(
    settings: Settings | None = None,
    port: int = 8765,
    host: str = "127.0.0.1",
    timeout: float | None = None,
) -> int:
    """Start the daemon in a background process and wait for readiness."""
    if is_daemon_running(port=port, host=host):
        pid = get_daemon_pid(settings.runtime_dir) if settings else None
        return pid or 0

    if settings is None:
        settings = Settings.from_env("codex")
    if timeout is None:
        try:
            timeout = float(os.environ.get("KNOWLEDGE_DAEMON_START_TIMEOUT", "900"))
        except ValueError:
            timeout = 900.0
        if not math.isfinite(timeout) or timeout <= 0:
            timeout = 900.0

    # Each stdio client owns a separate proxy. Serialize startup across those
    # processes, then recheck readiness before allocating another model copy.
    with index_lock(settings.runtime_dir, lock_name="daemon-start.lock"):
        if is_daemon_running(port=port, host=host):
            return get_daemon_pid(settings.runtime_dir) or 0
        return _start_daemon_locked(settings, port, host, timeout)


def _start_daemon_locked(settings: Settings, port: int, host: str, timeout: float) -> int:

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
    env["KNOWLEDGE_DAEMON_BACKGROUND"] = "1"

    creationflags = 0
    startupinfo = None
    if sys.platform == "win32":
        creationflags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE

    # The child installs rotating safe streams. An inherited raw file handle
    # would bypass both privacy filtering and rotation for its entire lifetime.
    proc = subprocess.Popen(
        cmd,
        cwd=str(settings.project_root),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
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

    # Keep the startup lock until the child is gone. A slow model load must not
    # survive this deadline and race the next client's startup. Windows venv
    # launchers have a child interpreter, so terminate the owned process tree.
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            check=True, capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW,
        )
    else:
        proc.terminate()
    proc.wait(timeout=5)
    raise TimeoutError(f"Daemon did not become healthy within {timeout} seconds on http://{host}:{port}/mcp")


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
    with daemon_logging(settings.runtime_dir, console=os.environ.get("KNOWLEDGE_DAEMON_BACKGROUND") != "1"):
        try:
            logging.getLogger(__name__).warning("", extra={"event_code": "daemon_starting"})
            _run_daemon(settings, host, port)
        except BaseException:
            logging.getLogger(__name__).error("", extra={"event_code": "daemon_failed"})
            raise


def _run_daemon(settings: Settings, host: str = "127.0.0.1", port: int = 8765) -> None:
    """Ensure Qdrant, sync once, and serve the shared models over HTTP."""
    from .cli import _dependencies, ensure_qdrant

    pid_path = settings.runtime_dir / "daemon.pid"
    settings.runtime_dir.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(str(os.getpid()), encoding="utf-8")

    def cleanup_pid() -> None:
        if get_daemon_pid(settings.runtime_dir) == os.getpid():
            try:
                pid_path.unlink()
            except OSError:
                pass

    atexit.register(cleanup_pid)

    try:
        ensure_qdrant(settings)
        store, operation_log, indexers = _dependencies(settings)

        sync_lock = asyncio.Lock()

        async def sync_all(*, rebuild: bool = False) -> dict[str, Any]:
            # Wait asynchronously before entering the Windows file lock. Two
            # simultaneous tools/call requests must not block the event loop
            # while the first writer awaits Qdrant.
            async with sync_lock:
                return await sync_all_locked(rebuild=rebuild)

        async def sync_all_locked(*, rebuild: bool = False) -> dict[str, Any]:
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

        mcp.run(
            transport="http", host=host, port=port, path="/mcp",
            host_origin_protection=True, allowed_hosts=[host, "127.0.0.1", "localhost", "[::1]"],
        )
    finally:
        cleanup_pid()
