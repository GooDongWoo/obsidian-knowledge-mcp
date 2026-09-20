from pathlib import Path
from dataclasses import FrozenInstanceError
import pytest

from knowledge_mcp.config import Settings


def test_settings_resolve_vault_from_project(tmp_path, monkeypatch):
    project = tmp_path / "00_System" / "knowledge-mcp"
    project.mkdir(parents=True)
    monkeypatch.setenv("KNOWLEDGE_VAULT_ROOT", str(tmp_path))
    settings = Settings.from_env("codex")
    assert settings.vault_root == tmp_path.resolve()
    assert settings.qdrant_url == "http://127.0.0.1:6333"
    assert settings.collection_name == "obsidian_knowledge_bge_m3_ko_v1"
    assert settings.runtime_dir == tmp_path.resolve() / ".knowledge"
    assert settings.dense_model == "dragonkue/BGE-m3-ko"
    assert settings.client_name == "codex"
    with pytest.raises(FrozenInstanceError):
        settings.client_name = "other"


def test_settings_reject_unsafe_runtime_overrides(monkeypatch):
    monkeypatch.setenv("KNOWLEDGE_QDRANT_URL", "https://remote.example")
    with pytest.raises(ValueError, match="KNOWLEDGE_QDRANT_URL"):
        Settings.from_env("codex")

    monkeypatch.delenv("KNOWLEDGE_QDRANT_URL")
    monkeypatch.setenv("KNOWLEDGE_DENSE_MODEL", "another-model")
    with pytest.raises(ValueError, match="KNOWLEDGE_DENSE_MODEL"):
        Settings.from_env("codex")


def test_settings_accept_bge_model(monkeypatch):
    monkeypatch.setenv("KNOWLEDGE_DENSE_MODEL", "dragonkue/BGE-m3-ko")
    assert Settings.from_env("codex").dense_model == "dragonkue/BGE-m3-ko"
