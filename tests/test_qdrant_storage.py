from knowledge_mcp.config import Settings


def test_qdrant_storage_is_under_vault_runtime(tmp_path):
    settings = Settings.from_paths(vault_root=tmp_path / "vault", project_root=tmp_path / "project")
    assert settings.qdrant_storage_dir == settings.runtime_dir / "qdrant"


def test_compose_uses_vault_storage_bind_mount(tmp_path):
    settings = Settings.from_paths(vault_root=tmp_path / "vault", project_root=tmp_path / "project")
    settings.project_root.mkdir(parents=True)
    compose = settings.project_root / "docker-compose.yml"
    compose.write_text(
        "volumes:\n  - source: ${KNOWLEDGE_QDRANT_STORAGE}\n    target: /qdrant/storage\n",
        encoding="utf-8",
    )
    text = compose.read_text(encoding="utf-8")
    assert "KNOWLEDGE_QDRANT_STORAGE" in text
    assert "/qdrant/storage" in text
