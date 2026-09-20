from pathlib import Path
from contextlib import contextmanager
from types import SimpleNamespace

from knowledge_mcp.cli import ensure_qdrant


def test_cli_accepts_full_rebuild():
    from knowledge_mcp.cli import _parser

    assert _parser().parse_args(["rebuild", "--client", "codex"]).command == "rebuild"


def test_model_collections_returns_single_bge_model(tmp_path):
    from knowledge_mcp.cli import model_settings
    from knowledge_mcp.config import Settings

    base = Settings.from_paths(vault_root=tmp_path / "vault", project_root=tmp_path / "project")
    models = model_settings(base)
    assert set(models) == {"dragonkue/BGE-m3-ko"}
    assert models["dragonkue/BGE-m3-ko"].collection_name == "obsidian_knowledge_bge_m3_ko_v1"


def test_model_collections_respect_custom_collection(tmp_path):
    from dataclasses import replace
    from knowledge_mcp.cli import model_settings
    from knowledge_mcp.config import Settings

    base = Settings.from_paths(vault_root=tmp_path / "vault", project_root=tmp_path / "project")
    models = model_settings(replace(base, collection_name="custom_knowledge"))
    assert models["dragonkue/BGE-m3-ko"].collection_name == "custom_knowledge"


def test_rebuild_removes_collection_before_reindexing(tmp_path, monkeypatch):
    from knowledge_mcp import cli
    from knowledge_mcp.indexer import IndexRunSummary
    from knowledge_mcp.config import Settings

    settings = Settings.from_paths(vault_root=tmp_path / "vault", project_root=tmp_path / "project")
    calls = []

    @contextmanager
    def locked(_):
        calls.append("lock")
        yield
        calls.append("unlock")

    class Client:
        async def collection_exists(self, name):
            return True

        async def delete_collection(self, name):
            calls.append(f"delete:{name}")

    class Indexer:
        def __init__(self, name):
            self.name = name
            self.manifest = SimpleNamespace(clear=lambda: calls.append(f"clear:{name}"))

        async def sync(self, *, assume_locked=False):
            assert assume_locked
            calls.append(f"sync:{self.name}")
            return IndexRunSummary(added=1)

    models = cli.model_settings(settings)
    stores = {model: SimpleNamespace(settings=selected, client=Client()) for model, selected in models.items()}
    indexers = {model: Indexer(model) for model in models}
    monkeypatch.setattr(cli.Settings, "from_env", lambda _: settings)
    monkeypatch.setattr(cli, "ensure_qdrant", lambda _: None)
    monkeypatch.setattr(cli, "index_lock", locked)
    monkeypatch.setattr(cli, "_dependencies", lambda _: (SimpleNamespace(stores=stores), None, indexers))

    assert cli.main(["rebuild"]) == 0
    assert calls == [
        "lock",
        "delete:obsidian_knowledge_bge_m3_ko_v1",
        "clear:dragonkue/BGE-m3-ko",
        "sync:dragonkue/BGE-m3-ko",
        "unlock",
    ]


def test_serve_starts_with_partial_file_failures(tmp_path, monkeypatch):
    from knowledge_mcp import cli
    from knowledge_mcp.indexer import IndexRunSummary
    from knowledge_mcp.config import Settings

    settings = Settings.from_paths(vault_root=tmp_path / "vault", project_root=tmp_path / "project")
    calls = []

    class Indexer:
        async def sync(self):
            return IndexRunSummary(failed=1, error_codes=["parse_failed"])

    monkeypatch.setattr(cli.Settings, "from_env", lambda _: settings)
    monkeypatch.setattr(cli, "ensure_qdrant", lambda _: None)
    monkeypatch.setattr(cli, "_dependencies", lambda _: (None, None, {"bge": Indexer()}))
    monkeypatch.setattr(
        cli, "create_application",
        lambda *_: SimpleNamespace(mcp=SimpleNamespace(run=lambda **kwargs: calls.append(kwargs))),
    )

    assert cli.main(["serve", "--standalone"]) == 0
    assert calls == [{"transport": "stdio"}]


def test_serve_proxy_starts_daemon_and_proxies(tmp_path, monkeypatch):
    from knowledge_mcp import cli
    from knowledge_mcp.config import Settings

    settings = Settings.from_paths(vault_root=tmp_path / "vault", project_root=tmp_path / "project")
    calls = []

    monkeypatch.setattr(cli.Settings, "from_env", lambda _: settings)
    monkeypatch.setattr(cli, "is_daemon_running", lambda **kwargs: False)
    monkeypatch.setattr(cli, "start_daemon_process", lambda *args, **kwargs: calls.append("start_daemon"))
    monkeypatch.setattr("anyio.run", lambda fn, *args: calls.append(("proxy", args)))

    assert cli.main(["serve"]) == 0
    assert "start_daemon" in calls
    assert any(c[0] == "proxy" for c in calls if isinstance(c, tuple))
from knowledge_mcp.config import Settings


def test_launcher_uses_project_compose_and_vault_runtime(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    project = tmp_path / "project"
    project.mkdir()
    (project / "docker-compose.yml").write_text("services: {}\n", encoding="utf-8")
    settings = Settings.from_paths(vault_root=vault, project_root=project)
    calls = []

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))

    class Response:
        status = 200
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return None

    monkeypatch.setattr("knowledge_mcp.cli.subprocess.run", fake_run)
    monkeypatch.setattr("knowledge_mcp.cli.urlopen", lambda *args, **kwargs: Response())
    ensure_qdrant(settings, timeout_seconds=1)

    assert str(project / "docker-compose.yml") in calls[0][0][0]
    assert calls[0][1]["env"]["KNOWLEDGE_QDRANT_STORAGE"] == str(settings.qdrant_storage_dir.resolve()).replace("\\", "/")
    assert settings.qdrant_storage_dir.is_dir()


def test_readme_contains_external_project_and_vault_paths():
    readme = Path(__file__).parents[1] / "README.md"
    text = readme.read_text(encoding="utf-8")
    assert "KNOWLEDGE_PROJECT_ROOT" in text
    assert "KNOWLEDGE_VAULT_ROOT" in text
    assert ".knowledge" in text
