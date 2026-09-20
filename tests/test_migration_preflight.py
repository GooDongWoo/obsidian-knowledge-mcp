def test_project_and_vault_roots_are_distinct(tmp_path):
    from knowledge_mcp.config import Settings

    vault = tmp_path / "vault"
    project = tmp_path / "project"
    settings = Settings.from_paths(vault_root=vault, project_root=project)
    assert settings.vault_root != settings.project_root
    assert settings.runtime_dir == vault / ".knowledge"


def test_external_project_config_and_vault_runtime(tmp_path, monkeypatch):
    from knowledge_mcp.config import Settings

    vault = tmp_path / "vault"
    project = tmp_path / "project"
    monkeypatch.setenv("KNOWLEDGE_VAULT_ROOT", str(vault))
    monkeypatch.setenv("KNOWLEDGE_PROJECT_ROOT", str(project))
    settings = Settings.from_env("codex")
    assert settings.project_root == project.resolve()
    assert settings.runtime_dir == (vault / ".knowledge").resolve()


def test_discovery_reads_ignore_from_project_but_matches_vault_paths(tmp_path):
    from knowledge_mcp.config import Settings
    from knowledge_mcp.documents import discover_sources

    vault = tmp_path / "vault"
    project = tmp_path / "project"
    vault.mkdir()
    project.mkdir()
    (project / ".knowledgeignore").write_text("ignored.md\n", encoding="utf-8")
    (vault / "ignored.md").write_text("skip", encoding="utf-8")
    (vault / "kept.md").write_text("keep", encoding="utf-8")
    settings = Settings.from_paths(vault_root=vault, project_root=project)
    assert [path.name for path in discover_sources(settings)] == ["kept.md"]
