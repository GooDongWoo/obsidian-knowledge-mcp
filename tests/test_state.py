import msvcrt
import sqlite3
from contextlib import contextmanager
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


def test_manifest_persists_optional_source_cache_and_migrates_old_schema(tmp_path):
    with sqlite3.connect(tmp_path / "state.sqlite3") as connection:
        connection.execute("CREATE TABLE collection_files (collection_name TEXT, path TEXT, content_hash TEXT, generation TEXT, point_count INTEGER, point_ids TEXT, completed_at TEXT, PRIMARY KEY(collection_name, path))")
        connection.execute("INSERT INTO collection_files VALUES ('test', 'old.md', 'hash', 'gen', 0, '[]', 'now')")
    manifest = Manifest(tmp_path, "test")
    old = manifest.completed_files()["old.md"]
    assert old.mtime_ns is None and old.size is None
    manifest.mark_complete("old.md", "hash", "gen", 0, mtime_ns=123, size=42,
                           metadata_signature="meta", indexing_version="config")
    entry = Manifest(tmp_path, "test").completed_files()["old.md"]
    assert (entry.mtime_ns, entry.size, entry.metadata_signature, entry.indexing_version) == (123, 42, "meta", "config")


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
        assert connection.execute("SELECT result_count FROM queries").fetchall() == [(0,)]


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
        assert connection.execute("SELECT result_count FROM queries").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM index_runs").fetchone()[0] == 1


def test_private_query_log_has_no_excerpt(operation_log):
    operation_log.record_query(query="secret", results=[private_result()])

    stored = operation_log.database_text_for_test()
    assert "private body" not in stored
    assert "secret" not in stored
    assert "point-private" not in stored
    assert "private.md" not in stored


def test_query_log_discards_filters_client_and_result_details(tmp_path):
    operation_log = OperationLog(tmp_path)
    filters = {"document_type": ["brainstorming"], "security_level": "private"}

    operation_log.record_query(query="secret", filters=filters, results=[private_result()],
                               client_name="private-client", elapsed_ms=12.5,
                               rerank_requested=True, rerank_applied=None)

    with sqlite3.connect(tmp_path / "state.sqlite3") as connection:
        metrics = connection.execute(
            "SELECT elapsed_ms, result_count, rerank_requested, rerank_applied, error_code FROM queries"
        ).fetchall()
    assert metrics == [(12.5, 1, 1, None, None)]
    stored = operation_log.database_text_for_test()
    for sensitive in ("secret", "brainstorming", "private-client", "private.md", "private body"):
        assert sensitive not in stored


def test_query_migration_backs_up_once_scrubs_pages_and_preserves_index_state(tmp_path):
    with sqlite3.connect(tmp_path / "state.sqlite3") as connection:
        connection.execute("CREATE TABLE queries (id INTEGER PRIMARY KEY, queried_at TEXT NOT NULL, query TEXT NOT NULL)")
        connection.execute("INSERT INTO queries(queried_at, query) VALUES ('2026-09-30T00:00:00+00:00', 'old query')")
        connection.execute("CREATE TABLE query_results (query_id INTEGER, source_path TEXT)")
        connection.execute("INSERT INTO query_results VALUES (1, 'sensitive-result.md')")
        connection.execute("""CREATE TABLE index_runs (id INTEGER PRIMARY KEY, started_at TEXT,
                           status TEXT, generation TEXT, added INTEGER, changed INTEGER, deleted INTEGER)""")
        connection.execute("INSERT INTO index_runs VALUES (1, 'now', 'completed', 'preserved', 1, 2, 3)")
        connection.execute("""CREATE TABLE collection_files (collection_name TEXT, path TEXT,
                           content_hash TEXT, generation TEXT, point_count INTEGER, point_ids TEXT,
                           completed_at TEXT, PRIMARY KEY(collection_name, path))""")
        connection.execute("INSERT INTO collection_files VALUES (?, 'manifest.md', 'hash', 'preserved', 1, '[]', 'now')",
                           (state.DEFAULT_COLLECTION_NAME,))

    with pytest.warns(RuntimeWarning, match="state.pre-privacy.sqlite3"):
        log = OperationLog(tmp_path)
    backup = tmp_path / "state.pre-privacy.sqlite3"
    original_backup = backup.read_bytes()
    restarted = OperationLog(tmp_path)

    with sqlite3.connect(tmp_path / "state.sqlite3") as connection:
        rows = connection.execute("SELECT result_count, rerank_requested, rerank_applied FROM queries").fetchall()
    assert rows == [(1, None, None)]
    assert "old query" not in restarted.database_text_for_test()
    assert "sensitive-result.md" not in restarted.database_text_for_test()
    assert "old query" in original_backup.decode("utf-8", errors="ignore")
    assert backup.read_bytes() == original_backup
    assert restarted.index_status()["generation"] == "preserved"
    assert Manifest(tmp_path).completed_files()["manifest.md"].generation == "preserved"


@pytest.mark.parametrize("value", ["invalid", "0", "-1"])
def test_invalid_retention_limits_use_bounded_defaults(tmp_path, monkeypatch, value):
    monkeypatch.setenv("KNOWLEDGE_QUERY_RETENTION_DAYS", value)
    monkeypatch.setenv("KNOWLEDGE_QUERY_MAX_ROWS", value)
    monkeypatch.setattr(state, "_now", lambda: "2026-10-01T00:00:00+00:00")
    log = OperationLog(tmp_path)
    with sqlite3.connect(tmp_path / "state.sqlite3") as connection:
        connection.execute("INSERT INTO queries(queried_at, result_count) VALUES ('2026-08-31T00:00:00+00:00', 0)")
    log.record_query(results=[], error_code="search_failed")
    with sqlite3.connect(tmp_path / "state.sqlite3") as connection:
        assert connection.execute("SELECT error_code FROM queries").fetchall() == [("search_failed",)]


def test_query_retention_keeps_exact_30_day_boundary_during_operation(tmp_path, monkeypatch):
    monkeypatch.setattr(state, "_now", lambda: "2026-10-01T00:00:00+00:00")
    log = OperationLog(tmp_path)
    with sqlite3.connect(tmp_path / "state.sqlite3") as connection:
        connection.executemany(
            "INSERT INTO queries(queried_at, result_count) VALUES (?, 0)",
            [("2026-08-31T23:59:59+00:00",), ("2026-09-01T00:00:00+00:00",)],
        )
    log.record_query(query="discard", results=[])
    with sqlite3.connect(tmp_path / "state.sqlite3") as connection:
        assert connection.execute("SELECT queried_at FROM queries ORDER BY id").fetchall() == [
            ("2026-09-01T00:00:00+00:00",), ("2026-10-01T00:00:00+00:00",)]


def test_query_row_cap_reclaims_database_pages_and_preserves_latest_index(tmp_path, monkeypatch):
    monkeypatch.setenv("KNOWLEDGE_QUERY_MAX_ROWS", "3")
    log = OperationLog(tmp_path)
    log.record_index(generation="keep", added=1, changed=0, deleted=0)
    with sqlite3.connect(tmp_path / "state.sqlite3") as connection:
        connection.executemany("INSERT INTO queries(queried_at, result_count) VALUES (?, 0)",
                               [("2026-10-01T00:00:00+00:00",)] * 20000)
    before = (tmp_path / "state.sqlite3").stat().st_size
    log.record_query(results=[], elapsed_ms=42)
    with sqlite3.connect(tmp_path / "state.sqlite3") as connection:
        assert connection.execute("SELECT COUNT(*) FROM queries").fetchone()[0] == 3
        assert connection.execute("SELECT elapsed_ms FROM queries ORDER BY id DESC LIMIT 1").fetchone()[0] == 42
        assert connection.execute("PRAGMA freelist_count").fetchone()[0] == 0
    assert (tmp_path / "state.sqlite3").stat().st_size < before / 2
    assert OperationLog(tmp_path).index_status()["generation"] == "keep"


def test_index_history_retention_preserves_latest_inactive_collection_status(tmp_path, monkeypatch):
    monkeypatch.setattr(state, "_now", lambda: "2026-10-01T00:00:00+00:00")
    log = OperationLog(tmp_path)
    with sqlite3.connect(tmp_path / "state.sqlite3") as connection:
        connection.executemany(
            "INSERT INTO index_runs(started_at, collection_name, status, generation, added, changed, deleted) "
            "VALUES (?, ?, 'completed', ?, 1, 0, 0)",
            [("2026-08-01T00:00:00+00:00", None, "old-legacy"),
             ("2026-08-02T00:00:00+00:00", None, "latest-legacy"),
             ("2026-08-01T00:00:00+00:00", "inactive", "latest-inactive"),
             ("2026-08-31T23:59:59+00:00", "active", "expired"),
             ("2026-09-01T00:00:00+00:00", "active", "boundary"),
             ("2026-10-01T00:00:00+00:00", "active", "latest-active")],
        )
    assert log.index_status("inactive")["generation"] == "latest-inactive"
    with sqlite3.connect(tmp_path / "state.sqlite3") as connection:
        assert connection.execute("SELECT generation FROM index_runs ORDER BY id").fetchall() == [
            ("latest-legacy",), ("latest-inactive",), ("boundary",), ("latest-active",)]
    assert OperationLog(tmp_path).index_status("active")["generation"] == "latest-active"


def test_index_history_row_cap_exempts_latest_status_per_collection(tmp_path, monkeypatch):
    monkeypatch.setenv("KNOWLEDGE_INDEX_MAX_ROWS", "2")
    log = OperationLog(tmp_path)
    for number in range(10):
        log.record_index(collection_name="active", generation=str(number), added=0, changed=0, deleted=0)
    with sqlite3.connect(tmp_path / "state.sqlite3") as connection:
        assert connection.execute("SELECT generation FROM index_runs ORDER BY id").fetchall() == [
            ("7",), ("8",), ("9",)]
    assert log.index_status("active")["generation"] == "9"


def test_query_retention_configuration_and_error_code_validation(tmp_path, monkeypatch):
    monkeypatch.setenv("KNOWLEDGE_QUERY_RETENTION_DAYS", "1")
    monkeypatch.setattr(state, "_now", lambda: "2026-10-01T00:00:00+00:00")
    log = OperationLog(tmp_path)
    with sqlite3.connect(tmp_path / "state.sqlite3") as connection:
        connection.execute("INSERT INTO queries(queried_at, result_count) VALUES ('2026-09-29T00:00:00+00:00', 0)")
    log.record_query(results=[], error_code="search_failed")
    with sqlite3.connect(tmp_path / "state.sqlite3") as connection:
        assert connection.execute("SELECT error_code FROM queries").fetchall() == [("search_failed",)]
    with pytest.raises(ValueError):
        log.record_query(results=[], error_code="private exception text")


def test_privacy_migration_failure_rolls_back_schema_for_retry(tmp_path, monkeypatch):
    with sqlite3.connect(tmp_path / "state.sqlite3") as connection:
        connection.execute("CREATE TABLE queries(id INTEGER PRIMARY KEY, queried_at TEXT, query TEXT)")
        connection.execute("INSERT INTO queries VALUES (1, '2026-09-30T00:00:00+00:00', 'legacy-secret')")
    original = state._add_column_if_missing

    def fail_after_query_migration(*args):
        raise RuntimeError("interrupted migration")

    monkeypatch.setattr(state, "_add_column_if_missing", fail_after_query_migration)
    with pytest.warns(RuntimeWarning), pytest.raises(RuntimeError, match="interrupted"):
        OperationLog(tmp_path)
    with sqlite3.connect(tmp_path / "state.sqlite3") as connection:
        assert connection.execute("SELECT query FROM queries").fetchall() == [("legacy-secret",)]
    monkeypatch.setattr(state, "_add_column_if_missing", original)
    with pytest.warns(RuntimeWarning):
        log = OperationLog(tmp_path)
    assert "legacy-secret" not in log.database_text_for_test()


def test_privacy_migration_retries_compaction_after_interrupted_vacuum(tmp_path, monkeypatch):
    with sqlite3.connect(tmp_path / "state.sqlite3") as connection:
        connection.execute("CREATE TABLE queries(id INTEGER PRIMARY KEY, queried_at TEXT, query TEXT)")
        connection.execute("INSERT INTO queries VALUES (1, '2026-09-30T00:00:00+00:00', ?)",
                           ("synthetic-secret" * 10000,))
    original_connection = state._connection

    class InterruptedConnection:
        def __init__(self, connection):
            self.connection = connection

        def execute(self, sql, *args):
            if sql == "VACUUM":
                raise RuntimeError("interrupted vacuum")
            return self.connection.execute(sql, *args)

        def __getattr__(self, name):
            return getattr(self.connection, name)

    @contextmanager
    def fail_vacuum(runtime_dir):
        with original_connection(runtime_dir) as connection:
            yield InterruptedConnection(connection)

    monkeypatch.setattr(state, "_connection", fail_vacuum)
    with pytest.warns(RuntimeWarning), pytest.raises(RuntimeError, match="interrupted vacuum"):
        OperationLog(tmp_path)
    before = (tmp_path / "state.sqlite3").stat().st_size
    monkeypatch.setattr(state, "_connection", original_connection)
    with pytest.warns(RuntimeWarning):
        OperationLog(tmp_path)
    assert (tmp_path / "state.sqlite3").stat().st_size < before / 2
    with sqlite3.connect(tmp_path / "state.sqlite3") as connection:
        assert connection.execute("PRAGMA freelist_count").fetchone()[0] == 0


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

    with pytest.warns(RuntimeWarning, match="state.pre-privacy.sqlite3"):
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
