from pathlib import Path
import pytest

from knowledge_mcp.config import Settings
from knowledge_mcp.documents import discover_sources


def test_cli_loads_fastmcp_settings_before_import(tmp_path):
    import os
    import subprocess
    import sys

    (tmp_path / ".env").write_text(
        "FASTMCP_CHECK_FOR_UPDATES=off\nFASTMCP_TELEMETRY_MODE=off\n", encoding="utf-8"
    )
    environment = os.environ.copy()
    for name in ("FASTMCP_CHECK_FOR_UPDATES", "FASTMCP_TELEMETRY_MODE", "KNOWLEDGE_PROJECT_ROOT"):
        environment.pop(name, None)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    subprocess.run(
        [sys.executable, "-c", "import knowledge_mcp.cli, fastmcp; "
         "assert fastmcp.settings.check_for_updates == 'off'; "
         "assert fastmcp.settings.telemetry_mode == 'off'"],
        cwd=tmp_path, env=environment, check=True, timeout=30,
    )


def test_env_file_loading_in_settings(tmp_path, monkeypatch):
    vault = tmp_path / "my_vault"
    vault.mkdir()
    project = tmp_path / "my_project"
    project.mkdir()

    env_file = project / ".env"
    env_file.write_text(
        f'KNOWLEDGE_VAULT_ROOT="{vault.as_posix()}"\n'
        f'KNOWLEDGE_PROJECT_ROOT="{project.as_posix()}"\n'
        'KNOWLEDGE_QDRANT_URL="http://127.0.0.1:6333"\n',
        encoding="utf-8",
    )

    monkeypatch.delenv("KNOWLEDGE_VAULT_ROOT", raising=False)
    monkeypatch.delenv("KNOWLEDGE_PROJECT_ROOT", raising=False)
    monkeypatch.chdir(project)

    settings = Settings.from_env("test-client")
    assert settings.vault_root == vault.resolve()
    assert settings.project_root == project.resolve()
    assert settings.qdrant_url == "http://127.0.0.1:6333"


def test_missing_vault_root_raises_error(tmp_path, monkeypatch):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    monkeypatch.chdir(empty_dir)
    monkeypatch.delenv("KNOWLEDGE_VAULT_ROOT", raising=False)
    monkeypatch.setattr("knowledge_mcp.config._load_env_file", lambda *_: None)

    with pytest.raises(ValueError, match="KNOWLEDGE_VAULT_ROOT must be configured"):
        Settings.from_env("test-client")


def test_extra_ignores_from_env(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    project = tmp_path / "project"
    vault.mkdir()
    project.mkdir()

    (vault / "normal.md").write_text("public content", encoding="utf-8")
    (vault / "Private").mkdir()
    (vault / "Private" / "secret.md").write_text("private content", encoding="utf-8")
    (vault / "Personal").mkdir()
    (vault / "Personal" / "diary.md").write_text("diary content", encoding="utf-8")

    monkeypatch.setenv("KNOWLEDGE_EXTRA_IGNORES", "Private/**;Personal/**")

    settings = Settings.from_paths(vault_root=vault, project_root=project)
    discovered = [p.name for p in discover_sources(settings)]
    assert discovered == ["normal.md"]


def test_knowledgeignore_local(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    project = tmp_path / "project"
    vault.mkdir()
    project.mkdir()

    (project / ".knowledgeignore").write_text("*.tmp\n", encoding="utf-8")
    (project / ".knowledgeignore.local").write_text("hidden.md\n", encoding="utf-8")

    (vault / "visible.md").write_text("visible", encoding="utf-8")
    (vault / "draft.tmp").write_text("draft", encoding="utf-8")
    (vault / "hidden.md").write_text("hidden", encoding="utf-8")

    monkeypatch.delenv("KNOWLEDGE_EXTRA_IGNORES", raising=False)
    settings = Settings.from_paths(vault_root=vault, project_root=project)
    discovered = [p.name for p in discover_sources(settings)]
    assert discovered == ["visible.md"]
