from pathlib import Path

from knowledge_mcp.config import Settings


def test_qdrant_storage_is_under_project_runtime(tmp_path):
    settings = Settings.from_paths(vault_root=tmp_path / "vault", project_root=tmp_path / "project")
    assert settings.qdrant_storage_dir == tmp_path / "project" / ".knowledge" / "qdrant"


def test_compose_defaults_to_project_storage_bind_mount():
    compose = Path(__file__).parents[1] / "docker-compose.yml"
    text = compose.read_text(encoding="utf-8")
    assert "${KNOWLEDGE_QDRANT_STORAGE:-./.knowledge/qdrant}" in text
    assert "/qdrant/storage" in text
