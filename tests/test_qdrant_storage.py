from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json

import pytest

from knowledge_mcp.config import Settings
from knowledge_mcp.cli import ensure_qdrant


@pytest.fixture
def settings(tmp_path):
    return replace(Settings.from_paths(vault_root=tmp_path / "vault", project_root=tmp_path / "project"),
                   qdrant_native_storage=tmp_path / "native", qdrant_executable=str(tmp_path / "qdrant.exe"))


def test_native_default_is_short_separate_and_project_specific(tmp_path):
    first = Settings.from_paths(vault_root=tmp_path / "v", project_root=tmp_path / "a")
    second = Settings.from_paths(vault_root=tmp_path / "v", project_root=tmp_path / "b")
    assert first.qdrant_storage_dir.is_absolute()
    assert first.qdrant_storage_dir.parent == Path.home() / ".knowledge-qdrant"
    assert first.qdrant_storage_dir != second.qdrant_storage_dir
    assert first.docker_qdrant_storage_dir == tmp_path / "a" / ".knowledge" / "qdrant"


def test_existing_healthy_server_needs_no_binary_or_storage(settings, monkeypatch):
    from knowledge_mcp import qdrant_process as native
    monkeypatch.setattr(native, "_healthy", lambda url: True)
    monkeypatch.setattr(native.subprocess, "Popen", lambda *a, **k: pytest.fail("must reuse server"))
    ensure_qdrant(settings)
    assert not settings.qdrant_storage_dir.exists()


def test_unknown_nonempty_storage_is_never_adopted(settings):
    from knowledge_mcp.qdrant_process import prepare_native_storage
    settings.qdrant_storage_dir.mkdir()
    old = settings.qdrant_storage_dir / "collections"
    old.mkdir()
    with pytest.raises(RuntimeError, match="snapshot"):
        prepare_native_storage(settings.qdrant_storage_dir)
    assert list(settings.qdrant_storage_dir.iterdir()) == [old]


def test_empty_storage_gets_durable_marker_and_restarts(settings):
    from knowledge_mcp.qdrant_process import prepare_native_storage, OWNER_MARKER
    prepare_native_storage(settings.qdrant_storage_dir)
    marker = settings.qdrant_storage_dir / OWNER_MARKER
    assert json.loads(marker.read_text())["format"] == 1
    (settings.qdrant_storage_dir / "collections").mkdir()
    prepare_native_storage(settings.qdrant_storage_dir)
    marker.write_text("{}")
    with pytest.raises(RuntimeError, match="ownership"):
        prepare_native_storage(settings.qdrant_storage_dir)


def test_native_launcher_absolute_paths_loopback_hidden_and_health_wait(settings, monkeypatch):
    from knowledge_mcp import qdrant_process as native
    Path(settings.qdrant_executable).touch()
    health = iter([False, False, True])
    monkeypatch.setattr(native, "_healthy", lambda url: next(health))
    calls = []
    process = SimpleNamespace(poll=lambda: None)
    monkeypatch.setattr(native.subprocess, "Popen", lambda *a, **k: calls.append((a, k)) or process)
    ensure_qdrant(settings)
    args, kwargs = calls[0]
    assert args[0][0] == str(Path(settings.qdrant_executable).resolve())
    env = kwargs["env"]
    assert env["QDRANT__SERVICE__HOST"] == "127.0.0.1"
    assert env["QDRANT__SERVICE__HTTP_PORT"] == "6333"
    assert env["QDRANT__SERVICE__GRPC_PORT"] == "6334"
    assert env["QDRANT__STORAGE__STORAGE_PATH"] == str(settings.qdrant_storage_dir)
    assert Path(env["QDRANT__STORAGE__SNAPSHOTS_PATH"]).is_absolute()
    assert env["QDRANT__TELEMETRY_DISABLED"] == "true"
    assert kwargs["stdin"] == native.subprocess.DEVNULL


def test_explicit_docker_rollback_uses_old_storage(settings, monkeypatch):
    from knowledge_mcp import qdrant_process as native
    health = iter([False, False, True])
    monkeypatch.setattr(native, "_healthy", lambda url: next(health))
    calls = []
    monkeypatch.setattr(native.subprocess, "run", lambda *a, **k: calls.append((a, k)))
    ensure_qdrant(replace(settings, qdrant_backend="docker"))
    args, kwargs = calls[0]
    assert args[0] == ["docker", "compose", "-f", str(settings.project_root / "docker-compose.yml"), "up", "-d"]
    assert kwargs["env"]["KNOWLEDGE_QDRANT_STORAGE"] == settings.docker_qdrant_storage_dir.as_posix()
    assert not settings.qdrant_storage_dir.exists()


def test_failed_owned_child_is_reaped_and_error_is_actionable(settings, monkeypatch):
    from knowledge_mcp import qdrant_process as native
    Path(settings.qdrant_executable).touch()
    monkeypatch.setattr(native, "_healthy", lambda url: False)
    calls = []
    process = SimpleNamespace(poll=lambda: None, terminate=lambda: calls.append("terminate"),
                              wait=lambda **kw: calls.append("wait"))
    monkeypatch.setattr(native.subprocess, "Popen", lambda *a, **k: process)
    with pytest.raises(RuntimeError, match="Qdrant.*log"):
        ensure_qdrant(settings, timeout_seconds=0)
    assert calls == ["terminate", "wait"]


def test_settings_read_native_and_docker_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("KNOWLEDGE_VAULT_ROOT", str(tmp_path / "v"))
    monkeypatch.setenv("KNOWLEDGE_PROJECT_ROOT", str(tmp_path / "p"))
    monkeypatch.setenv("KNOWLEDGE_QDRANT_NATIVE_STORAGE", str(tmp_path / "native"))
    monkeypatch.setenv("KNOWLEDGE_QDRANT_STORAGE", str(tmp_path / "old"))
    monkeypatch.setenv("KNOWLEDGE_QDRANT_EXECUTABLE", str(tmp_path / "qdrant.exe"))
    monkeypatch.setenv("KNOWLEDGE_QDRANT_BACKEND", "native")
    config = Settings.from_env("test")
    assert config.qdrant_storage_dir == tmp_path / "native"
    assert config.docker_qdrant_storage_dir == tmp_path / "old"
    assert config.qdrant_executable == str(tmp_path / "qdrant.exe")
    monkeypatch.setenv("KNOWLEDGE_QDRANT_BACKEND", "typo")
    with pytest.raises(ValueError, match="backend"):
        Settings.from_env("test")


def test_compose_defaults_to_separate_docker_storage():
    text = (Path(__file__).parents[1] / "docker-compose.yml").read_text()
    assert "${KNOWLEDGE_QDRANT_STORAGE:-./.knowledge/qdrant}" in text


def test_native_finds_executable_on_path(settings, monkeypatch):
    from knowledge_mcp import qdrant_process as native
    executable = Path(settings.qdrant_executable)
    executable.touch()
    health = iter([False, False, True])
    monkeypatch.setattr(native, "_healthy", lambda url: next(health))
    monkeypatch.setattr(native.shutil, "which", lambda name: str(executable))
    calls = []
    monkeypatch.setattr(native.subprocess, "Popen", lambda *a, **k: calls.append(a) or SimpleNamespace(poll=lambda: None))
    ensure_qdrant(replace(settings, qdrant_executable=None))
    assert calls[0][0][0] == str(executable)


def test_missing_binary_does_not_create_native_storage(settings, monkeypatch):
    from knowledge_mcp import qdrant_process as native
    monkeypatch.setattr(native, "_healthy", lambda url: False)
    with pytest.raises(RuntimeError, match="KNOWLEDGE_QDRANT_EXECUTABLE"):
        ensure_qdrant(settings)
    assert not settings.qdrant_storage_dir.exists()


def test_native_storage_rejects_relative_override(settings):
    with pytest.raises(ValueError, match="absolute"):
        replace(settings, qdrant_native_storage=Path("relative"))


def test_custom_legacy_docker_storage_does_not_become_native(tmp_path, monkeypatch):
    from knowledge_mcp import qdrant_process as native
    monkeypatch.setenv("KNOWLEDGE_VAULT_ROOT", str(tmp_path / "vault"))
    monkeypatch.setenv("KNOWLEDGE_PROJECT_ROOT", str(tmp_path / "project"))
    monkeypatch.setenv("KNOWLEDGE_QDRANT_STORAGE", str(tmp_path / "custom-docker"))
    monkeypatch.delenv("KNOWLEDGE_QDRANT_NATIVE_STORAGE", raising=False)
    settings = Settings.from_env("test")
    assert settings.docker_qdrant_storage_dir == tmp_path / "custom-docker"
    assert settings.qdrant_storage_dir != settings.docker_qdrant_storage_dir
    health = iter([False, False, True])
    monkeypatch.setattr(native, "_healthy", lambda url: next(health))
    calls = []
    monkeypatch.setattr(native.subprocess, "run", lambda *a, **k: calls.append(k))
    ensure_qdrant(replace(settings, qdrant_backend="docker"))
    assert calls[0]["env"]["KNOWLEDGE_QDRANT_STORAGE"] == (tmp_path / "custom-docker").as_posix()
