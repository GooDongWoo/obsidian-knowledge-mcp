from pathlib import Path
from contextlib import contextmanager
from types import SimpleNamespace



def test_cli_accepts_full_rebuild():
    from knowledge_mcp.cli import _parser

    assert _parser().parse_args(["rebuild", "--client", "codex"]).command == "rebuild"


def test_cli_force_full_hash_reaches_indexer(tmp_path, monkeypatch):
    from knowledge_mcp import cli
    from knowledge_mcp.config import Settings
    from knowledge_mcp.indexer import IndexRunSummary

    settings = Settings.from_paths(vault_root=tmp_path / "vault", project_root=tmp_path / "project")
    calls = []

    class Indexer:
        async def sync(self, **kwargs):
            calls.append(kwargs)
            return IndexRunSummary()

    monkeypatch.setattr(cli.Settings, "from_env", lambda _: settings)
    monkeypatch.setattr(cli, "ensure_qdrant", lambda _: None)
    monkeypatch.setattr(cli, "_dependencies", lambda _: (None, None, {"test": Indexer()}))
    assert cli.main(["index", "--force-full-hash"]) == 0
    assert calls == [{"force_full_hash": True}]


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

        async def sync(self, *, assume_locked=False, force_full_hash=False):
            assert assume_locked
            assert not force_full_hash
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


def test_serve_rejects_standalone_flag():
    import pytest
    from knowledge_mcp.cli import _parser

    with pytest.raises(SystemExit):
        _parser().parse_args(["serve", "--standalone"])


def test_daemon_application_starts_with_partial_file_failures(tmp_path, monkeypatch):
    import asyncio
    from fastmcp import Client
    from knowledge_mcp import cli
    from knowledge_mcp.daemon import create_daemon_application
    from knowledge_mcp.indexer import IndexRunSummary
    from knowledge_mcp.config import Settings
    from knowledge_mcp.state import OperationLog
    from test_server import FakeStore

    settings = Settings.from_paths(vault_root=tmp_path / "vault", project_root=tmp_path / "project")

    class Indexer:
        async def sync(self, **kwargs):
            return IndexRunSummary(failed=1, error_codes=["parse_failed"])

    monkeypatch.setattr(cli, "ensure_qdrant", lambda _: None)
    monkeypatch.setattr(cli, "_dependencies", lambda _: (
        FakeStore(), OperationLog(settings.runtime_dir), {"bge": Indexer()},
    ))

    app = create_daemon_application(settings)

    async def exercise_lifespan():
        async with Client(app.mcp) as client:
            async with asyncio.timeout(2):
                while app.runtime_status["state"] in {"starting", "indexing"}:
                    await asyncio.sleep(.01)
            status = (await client.call_tool("knowledge-index-status", {})).data
            assert status["state"] == "error"
            assert status["last_error"] == "parse_failed"
            assert status["last_index"]["bge"]["failed"] == 1
            assert {tool.name for tool in await client.list_tools()} == {
                "qdrant-find", "knowledge-index-status", "knowledge-index-sync"
            }

    asyncio.run(exercise_lifespan())


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
    proxy_call = next(c for c in calls if isinstance(c, tuple) and c[0] == "proxy")
    assert proxy_call[1][0] == "http://127.0.0.1:8765/mcp"


def test_cli_import_keeps_ml_out_of_client_process():
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "-c", "import sys, knowledge_mcp.cli; "
         "assert not {'torch', 'onnxruntime', 'fastembed', 'sentence_transformers'} & sys.modules.keys()"],
        check=True, timeout=30,
    )


def test_readme_contains_external_project_and_vault_paths():
    readme = Path(__file__).parents[1] / "README.md"
    text = readme.read_text(encoding="utf-8")
    assert "KNOWLEDGE_PROJECT_ROOT" in text
    assert "KNOWLEDGE_VAULT_ROOT" in text
    assert ".knowledge" in text
