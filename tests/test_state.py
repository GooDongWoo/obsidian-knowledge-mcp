import json
import msvcrt
import sqlite3
from types import SimpleNamespace

import pytest

import knowledge_mcp.state as state
from knowledge_mcp.state import Manifest, OperationLog, index_lock


@pytest.fixture
def operation_log(tmp_path):
    return OperationLog(tmp_path)


def private_result():
    return {
        "source_path": "private.md",
        "point_id": "point-private",
        "score": 0.91,
        "document": "private body",
        "security_level": "private",
    }


def test_manifest_detects_add_change_delete(tmp_path):
    manifest = Manifest(tmp_path)
    manifest.mark_complete("a.md", "hash-a", "gen-a", 1)

    plan = manifest.diff({"a.md": "hash-b", "b.md": "hash-c"})

    assert plan.added == ["b.md"]
    assert plan.changed == ["a.md"]
    assert plan.deleted == []

    assert manifest.diff({}).deleted == ["a.md"]


def test_manifest_persists_point_ids_without_document_body(tmp_path):
    manifest = Manifest(tmp_path)

    manifest.mark_complete("private.md", "hash-a", "gen-a", 2, ["point-1", "point-2"])

    stored = manifest.database_text_for_test()
    assert "private.md" in stored
    assert "point-1" in stored
    assert "private body" not in stored

    entry = manifest.completed_files()["private.md"]
    assert entry.point_ids == ("point-1", "point-2")


def test_manifest_is_scoped_by_collection(tmp_path):
    first = Manifest(tmp_path, "first")
    second = Manifest(tmp_path, "second")
    first.mark_complete("shared.md", "hash-first", "gen-first", 1)
    second.mark_complete("shared.md", "hash-second", "gen-second", 2)

    assert first.completed_files()["shared.md"].content_hash == "hash-first"
    assert second.completed_files()["shared.md"].content_hash == "hash-second"
    assert first.diff({"shared.md": "hash-first"}).changed == []
    assert second.diff({"shared.md": "hash-first"}).changed == ["shared.md"]

    first.remove("shared.md")
    assert first.completed_files() == {}
    assert "shared.md" in second.completed_files()


def test_manifest_ignores_legacy_unscoped_rows(tmp_path):
    OperationLog(tmp_path)
    with sqlite3.connect(tmp_path / "state.sqlite3") as connection:
        connection.execute(
            "INSERT INTO files VALUES (?, ?, ?, ?, ?, ?)",
            ("old.md", "old-hash", "old-gen", 1, "[]", "now"),
        )

    manifest = Manifest(tmp_path)

    assert manifest.completed_files() == {}
    assert manifest.diff({"old.md": "old-hash"}).added == ["old.md"]


def test_manifest_clear_removes_only_selected_collection_and_preserves_queries(tmp_path):
    first = Manifest(tmp_path, "first")
    second = Manifest(tmp_path, "second")
    first.mark_complete("shared.md", "hash-first", "gen-first", 1)
    second.mark_complete("shared.md", "hash-second", "gen-second", 1)
    log = OperationLog(tmp_path)
    log.record_query(query="keep this query", results=[])

    first.clear()

    assert first.completed_files() == {}
    assert "shared.md" in second.completed_files()
    with sqlite3.connect(tmp_path / "state.sqlite3") as connection:
        assert connection.execute("SELECT query FROM queries").fetchall() == [("keep this query",)]


def test_default_manifest_clear_discards_legacy_files_without_erasing_logs(tmp_path):
    manifest = Manifest(tmp_path)
    log = OperationLog(tmp_path)
    with sqlite3.connect(tmp_path / "state.sqlite3") as connection:
        connection.execute(
            "INSERT INTO files VALUES (?, ?, ?, ?, ?, ?)",
            ("old.md", "old-hash", "old-gen", 1, "[]", "now"),
        )
    log.record_query(query="retained query", results=[])
    log.record_index(generation="old-gen", added=1, changed=0, deleted=0)

    manifest.clear()

    with sqlite3.connect(tmp_path / "state.sqlite3") as connection:
        assert connection.execute("SELECT COUNT(*) FROM files").fetchone()[0] == 0
        assert connection.execute("SELECT query FROM queries").fetchone()[0] == "retained query"
        assert connection.execute("SELECT COUNT(*) FROM index_runs").fetchone()[0] == 1


def test_private_query_log_has_no_excerpt(operation_log):
    operation_log.record_query(query="secret", results=[private_result()])

    stored = operation_log.database_text_for_test()
    assert "private body" not in stored
    assert "point-private" in stored
    assert "private.md" in stored


def test_query_log_persists_filters_without_result_body(tmp_path):
    operation_log = OperationLog(tmp_path)
    filters = {"document_type": ["brainstorming"], "security_level": "private"}

    operation_log.record_query(query="secret", filters=filters, results=[private_result()])

    with sqlite3.connect(tmp_path / "state.sqlite3") as connection:
        stored_filters = connection.execute("SELECT filters FROM queries").fetchone()[0]
    assert json.loads(stored_filters) == filters
    assert "private body" not in operation_log.database_text_for_test()


def test_query_filter_migration_preserves_existing_queries(tmp_path):
    with sqlite3.connect(tmp_path / "state.sqlite3") as connection:
        connection.execute("CREATE TABLE queries (id INTEGER PRIMARY KEY, queried_at TEXT NOT NULL, query TEXT NOT NULL)")
        connection.execute("INSERT INTO queries(queried_at, query) VALUES ('now', 'old query')")

    OperationLog(tmp_path)

    with sqlite3.connect(tmp_path / "state.sqlite3") as connection:
        rows = connection.execute("SELECT query, filters FROM queries ORDER BY id").fetchall()
    assert rows == [("old query", "{}")]


def test_index_log_reports_latest_run(operation_log):
    operation_log.record_index(generation="gen-a", added=1, changed=2, deleted=3)

    assert operation_log.index_status() == {
        "status": "completed",
        "generation": "gen-a",
        "added": 1,
        "changed": 2,
        "deleted": 3,
        "error_code": None,
    }


def test_index_log_reports_latest_run_for_each_collection(operation_log):
    operation_log.record_index(
        collection_name="e5", generation="gen-e5", added=1, changed=0, deleted=0,
        status="partial", error_code="parse_failed",
    )
    operation_log.record_index(
        collection_name="bge", generation="gen-bge", added=2, changed=0, deleted=0,
    )

    assert operation_log.index_status("e5") == {
        "status": "partial",
        "generation": "gen-e5",
        "added": 1,
        "changed": 0,
        "deleted": 0,
        "error_code": "parse_failed",
    }
    assert operation_log.index_status("bge")["status"] == "completed"


def test_index_log_stores_stable_error_code_not_private_error_text(operation_log):
    operation_log.record_index(
        generation="gen-a",
        added=0,
        changed=0,
        deleted=0,
        status="failed",
        error_code="qdrant_replace_failed",
    )

    assert operation_log.index_status()["error_code"] == "qdrant_replace_failed"
    with pytest.raises(TypeError):
        operation_log.record_index(
            generation="gen-a", added=0, changed=0, deleted=0, error="private body"
        )
    assert "private body" not in operation_log.database_text_for_test()


def test_error_migration_scrubs_legacy_private_text(tmp_path):
    with sqlite3.connect(tmp_path / "state.sqlite3") as connection:
        connection.execute(
            """
            CREATE TABLE index_runs (
                id INTEGER PRIMARY KEY, started_at TEXT NOT NULL, status TEXT NOT NULL,
                generation TEXT, added INTEGER NOT NULL, changed INTEGER NOT NULL,
                deleted INTEGER NOT NULL, error TEXT
            )
            """
        )
        connection.execute(
            """
            INSERT INTO index_runs(started_at, status, generation, added, changed, deleted, error)
            VALUES ('now', 'failed', NULL, 0, 0, 0, 'private body')
            """
        )

    operation_log = OperationLog(tmp_path)

    assert "private body" not in operation_log.database_text_for_test()
    assert operation_log.index_status()["error_code"] is None


def test_index_lock_releases_after_context_exit(tmp_path):
    with index_lock(tmp_path):
        assert (tmp_path / "index.lock").is_file()

    with index_lock(tmp_path):
        pass


def test_index_lock_releases_when_indexer_fails(tmp_path, monkeypatch):
    modes = []

    def capture_lock(_descriptor, mode, _length):
        modes.append(mode)

    monkeypatch.setattr("knowledge_mcp.state.msvcrt.locking", capture_lock)
    with pytest.raises(RuntimeError, match="failed index"):
        with index_lock(tmp_path):
            raise RuntimeError("failed index")
    assert modes[-1] == msvcrt.LK_UNLCK


def test_index_lock_retries_more_than_msvcrt_blocking_limit(tmp_path, monkeypatch):
    attempts = 0

    def contend(_descriptor, mode, _length):
        nonlocal attempts
        if mode == msvcrt.LK_NBLCK:
            attempts += 1
            if attempts <= 11:
                raise OSError("lock held")

    monkeypatch.setattr("knowledge_mcp.state.msvcrt.locking", contend)
    monkeypatch.setattr(state, "time", SimpleNamespace(sleep=lambda _seconds: None), raising=False)

    with index_lock(tmp_path):
        pass

    assert attempts == 12
