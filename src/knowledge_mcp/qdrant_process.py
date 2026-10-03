"""Health-first local Qdrant launch, with explicit native storage provenance."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from urllib.parse import urlsplit
from urllib.request import urlopen

from .config import Settings
from .state import index_lock

OWNER_MARKER = ".knowledge-native-owner.json"
_OWNER = {"owner": "knowledge-mcp-native-qdrant", "format": 1}


def prepare_native_storage(storage: Path) -> None:
    """Initialize only empty directories; a marker permits subsequent restarts."""
    storage.mkdir(parents=True, exist_ok=True)
    marker = storage / OWNER_MARKER
    if marker.exists():
        try:
            if json.loads(marker.read_text(encoding="utf-8")) == _OWNER:
                return
        except (ValueError, OSError):
            pass
        raise RuntimeError("Invalid native Qdrant storage ownership marker; restore a verified snapshot into new storage")
    if any(storage.iterdir()):
        raise RuntimeError("Unknown nonempty Qdrant storage: use new empty native storage and restore a snapshot; never adopt Docker data")
    with marker.open("x", encoding="utf-8") as stream:
        json.dump(_OWNER, stream)
        stream.flush()
        os.fsync(stream.fileno())


def _healthy(url: str) -> bool:
    try:
        with urlopen(f"{url}/healthz", timeout=3) as response:
            return response.status == 200
    except OSError:
        return False


def _start_native(settings: Settings):
    executable = settings.qdrant_executable or shutil.which("qdrant")
    if not executable or not Path(executable).is_absolute() or not Path(executable).is_file():
        raise RuntimeError("Set KNOWLEDGE_QDRANT_EXECUTABLE to the absolute Qdrant 1.19.1 executable path, or put qdrant on PATH")
    endpoint = urlsplit(settings.qdrant_url)
    if endpoint.scheme != "http" or endpoint.hostname not in ("localhost", "127.0.0.1"):
        raise ValueError("Native Qdrant automatic start requires a local HTTP endpoint")
    port = endpoint.port or 6333
    storage = settings.qdrant_storage_dir.resolve()
    prepare_native_storage(storage)
    config = storage / ".knowledge-native.yaml"
    config.write_text("{}\n", encoding="utf-8")
    # Exclude inherited Qdrant options that could override isolation/binding.
    environment = {key: value for key, value in os.environ.items() if not key.startswith("QDRANT__")}
    environment.update({
        "QDRANT__SERVICE__HOST": "127.0.0.1",
        "QDRANT__SERVICE__HTTP_PORT": str(port),
        "QDRANT__SERVICE__GRPC_PORT": str(port + 1),
        "QDRANT__STORAGE__STORAGE_PATH": str(storage),
        "QDRANT__STORAGE__SNAPSHOTS_PATH": str(storage / "snapshots"),
        "QDRANT__STORAGE__TEMP_PATH": str(storage / "tmp"),
        "QDRANT__TELEMETRY_DISABLED": "true",
        "QDRANT__LOG_LEVEL": "WARN",
    })
    log_path = storage / "native.log"
    # Keep the previous launch log; service installations use WinSW rotation.
    if log_path.exists():
        log_path.replace(storage / "native.previous.log")
    options = {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {"start_new_session": True}
    with log_path.open("ab") as log:
        process = subprocess.Popen(
            [str(Path(executable).resolve()), "--config-path", str(config)],
            cwd=storage, env=environment, stdin=subprocess.DEVNULL,
            stdout=log, stderr=log, **options,
        )
    return process, log_path


def ensure_qdrant(settings: Settings, *, timeout_seconds: int = 60) -> None:
    """Reuse healthy services, otherwise start the explicitly selected backend."""
    if _healthy(settings.qdrant_url):
        return
    with index_lock(settings.runtime_dir, lock_name="qdrant-start.lock"):
        if _healthy(settings.qdrant_url):
            return
        process, log_path = None, None
        if settings.qdrant_backend == "docker":
            storage = settings.docker_qdrant_storage_dir
            storage.mkdir(parents=True, exist_ok=True)
            environment = os.environ.copy()
            environment["KNOWLEDGE_QDRANT_STORAGE"] = storage.as_posix()
            subprocess.run(
                ["docker", "compose", "-f", str(settings.project_root / "docker-compose.yml"), "up", "-d"],
                cwd=settings.project_root, check=True, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, env=environment,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            )
        else:
            process, log_path = _start_native(settings)
        deadline = time.monotonic() + timeout_seconds
        ready = False
        try:
            while time.monotonic() < deadline:
                if process is not None and process.poll() is not None:
                    break
                if _healthy(settings.qdrant_url):
                    ready = True
                    return
                time.sleep(0.1)
            raise RuntimeError(f"Qdrant did not become healthy; inspect log {log_path or 'docker compose logs qdrant'}. Use a short absolute native storage path on Windows.")
        finally:
            # Only reap a child this call created, and only after failed startup.
            if process is not None and not ready:
                if process.poll() is None:
                    process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
