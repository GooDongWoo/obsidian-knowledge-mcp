"""Durable, body-free local state for incremental indexing and operations."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, closing, contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import msvcrt
import os
from pathlib import Path
import re
import sqlite3
import time
from typing import Iterator, Mapping, Sequence, Sized
import warnings


DATABASE_NAME = "state.sqlite3"
DEFAULT_COLLECTION_NAME = "obsidian_knowledge_v1"
ERROR_CODE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
PRIVACY_BACKUP_NAME = "state.pre-privacy.sqlite3"


@dataclass(frozen=True, slots=True)
class SyncPlan:
    added: list[str]
    changed: list[str]
    deleted: list[str]


@dataclass(frozen=True, slots=True)
class ManifestEntry:
    content_hash: str
    generation: str
    point_count: int
    point_ids: tuple[str, ...] = ()
    mtime_ns: int | None = None
    size: int | None = None
    metadata_signature: str | None = None
    indexing_version: str | None = None


class Manifest:
    """Tracks only successfully replaced Qdrant generations.

    Call ``mark_complete`` after, never before, Qdrant has replaced a file's
    generation. This makes a failed replacement eligible for the next diff.
    """

    def __init__(self, runtime_dir: Path, collection_name: str = DEFAULT_COLLECTION_NAME) -> None:
        self.runtime_dir = Path(runtime_dir)
        self.collection_name = collection_name
        _initialize(self.runtime_dir)

    def diff(self, current_files: Mapping[str, str]) -> SyncPlan:
        with _connection(self.runtime_dir) as connection:
            indexed = {
                row["path"]: row["content_hash"]
                for row in connection.execute(
                    "SELECT path, content_hash FROM collection_files WHERE collection_name = ?",
                    (self.collection_name,),
                )
            }
        current = dict(current_files)
        return SyncPlan(
            added=sorted(set(current) - set(indexed)),
            changed=sorted(path for path in set(current) & set(indexed) if current[path] != indexed[path]),
            deleted=sorted(set(indexed) - set(current)),
        )

    def mark_complete(
        self,
        path: str,
        content_hash: str,
        generation: str,
        point_count: int,
        point_ids: Sequence[str] = (),
        *, mtime_ns: int | None = None, size: int | None = None,
        metadata_signature: str | None = None, indexing_version: str | None = None,
    ) -> None:
        with _connection(self.runtime_dir) as connection:
            connection.execute(
                """
                INSERT INTO collection_files(collection_name, path, content_hash, generation, point_count, point_ids, completed_at,
                                             mtime_ns, size, metadata_signature, indexing_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(collection_name, path) DO UPDATE SET
                    content_hash = excluded.content_hash,
                    generation = excluded.generation,
                    point_count = excluded.point_count,
                    point_ids = excluded.point_ids,
                    completed_at = excluded.completed_at,
                    mtime_ns = excluded.mtime_ns,
                    size = excluded.size,
                    metadata_signature = excluded.metadata_signature,
                    indexing_version = excluded.indexing_version
                """,
                (self.collection_name, path, content_hash, generation, point_count, json.dumps(list(point_ids)), _now(),
                 mtime_ns, size, metadata_signature, indexing_version),
            )

    def remove(self, path: str) -> None:
        """Forget a path after its Qdrant points have been deleted."""
        with _connection(self.runtime_dir) as connection:
            connection.execute(
                "DELETE FROM collection_files WHERE collection_name = ? AND path = ?",
                (self.collection_name, path),
            )

    def clear(self) -> None:
        """Forget this collection's files, including default collection legacy state."""
        with _connection(self.runtime_dir) as connection:
            connection.execute(
                "DELETE FROM collection_files WHERE collection_name = ?", (self.collection_name,)
            )
            if self.collection_name == DEFAULT_COLLECTION_NAME:
                connection.execute("DELETE FROM files")

    def completed_files(self) -> dict[str, ManifestEntry]:
        """Return the committed input hashes and expected Qdrant generations."""
        with _connection(self.runtime_dir) as connection:
            return {
                row["path"]: ManifestEntry(
                    row["content_hash"], row["generation"], row["point_count"],
                    tuple(json.loads(row["point_ids"] or "[]")),
                    row["mtime_ns"], row["size"], row["metadata_signature"], row["indexing_version"],
                )
                for row in connection.execute(
                    "SELECT path, content_hash, generation, point_count, point_ids, mtime_ns, size, metadata_signature, indexing_version "
                    "FROM collection_files WHERE collection_name = ?",
                    (self.collection_name,),
                )
            }

    def database_text_for_test(self) -> str:
        return _database_text(self.runtime_dir)


class OperationLog:
    """Stores bounded query metrics and durable index status, without query payloads."""

    def __init__(self, runtime_dir: Path) -> None:
        self.runtime_dir = Path(runtime_dir)
        _initialize(self.runtime_dir)

    def record_index(
        self,
        *,
        collection_name: str | None = None,
        generation: str | None,
        added: int,
        changed: int,
        deleted: int,
        status: str = "completed",
        error_code: str | None = None,
    ) -> None:
        if error_code is not None and not ERROR_CODE.fullmatch(error_code):
            raise ValueError("error_code must be a stable lowercase identifier")
        with _connection(self.runtime_dir) as connection:
            connection.execute(
                """
                INSERT INTO index_runs(
                    started_at, collection_name, status, generation, added, changed, deleted, error_code
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (_now(), collection_name, status, generation, added, changed, deleted, error_code),
            )
            _prune_operations(connection)

    def record_query(
        self,
        *,
        query: str | None = None,
        filters: Mapping[str, object] | None = None,
        results: Sized = (),
        client_name: str | None = None,
        elapsed_ms: float | None = None,
        rerank_requested: bool | None = None,
        rerank_applied: bool | None = None,
        error_code: str | None = None,
    ) -> None:
        # Keep the legacy keyword API, but never serialize its payload fields.
        if error_code is not None and not ERROR_CODE.fullmatch(error_code):
            raise ValueError("error_code must be a stable lowercase identifier")
        with _connection(self.runtime_dir) as connection:
            connection.execute(
                """
                INSERT INTO queries(queried_at, elapsed_ms, result_count, rerank_requested, rerank_applied, error_code)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (_now(), elapsed_ms, len(results), rerank_requested, rerank_applied, error_code),
            )
            _prune_operations(connection)

    def index_status(self, collection_name: str | None = None) -> dict[str, object]:
        with _connection(self.runtime_dir) as connection:
            _prune_operations(connection)
            if collection_name is None:
                row = connection.execute(
                    """
                    SELECT status, generation, added, changed, deleted, error_code
                    FROM index_runs ORDER BY id DESC LIMIT 1
                    """
                ).fetchone()
            else:
                row = connection.execute(
                    """
                    SELECT status, generation, added, changed, deleted, error_code
                    FROM index_runs WHERE collection_name = ? ORDER BY id DESC LIMIT 1
                    """,
                    (collection_name,),
                ).fetchone()
        return dict(row) if row else {}

    def database_text_for_test(self) -> str:
        return _database_text(self.runtime_dir)


@contextmanager
def index_lock(runtime_dir: Path, *, lock_name: str = "index.lock") -> Iterator[None]:
    """Serialize indexers with a Windows advisory lock, releasing it always."""
    directory = Path(runtime_dir)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / lock_name).open("a+b") as lock_file:
        lock_file.seek(0)
        while True:
            try:
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                break
            except OSError:
                time.sleep(0.1)
        try:
            # Windows permits locking beyond EOF. Initialize only after owning
            # the byte: reading it first fails when another process holds it.
            if lock_file.seek(0, 2) == 0:
                lock_file.write(b"0")
                lock_file.flush()
            yield
        finally:
            lock_file.seek(0)
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)


@asynccontextmanager
async def async_index_lock(runtime_dir: Path):
    """Wait cooperatively; cancellation never leaves a worker acquiring a lock."""
    directory = Path(runtime_dir)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "index.lock").open("a+b") as lock_file:
        lock_file.seek(0)
        while True:
            try:
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                break
            except OSError:
                await asyncio.sleep(0.1)
        try:
            if lock_file.seek(0, 2) == 0:
                lock_file.write(b"0")
                lock_file.flush()
            yield
        finally:
            lock_file.seek(0)
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)


async def run_blocking(function, /, *args, **kwargs):
    """Finish an in-flight thread operation before releasing its owner/lock."""
    task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        # A thread cannot be cancelled. In particular, a manifest commit must
        # finish before another writer can enter or shutdown closes clients.
        try:
            await finish_thread(task)
        except Exception:
            pass
        raise


async def finish_thread(task):
    """Drain owned thread work even when request and shutdown both cancel it."""
    import anyio

    with anyio.CancelScope(shield=True):
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
        return task.result()


@contextmanager
def _connection(runtime_dir: Path) -> Iterator[sqlite3.Connection]:
    connection = sqlite3.connect(_database_path(runtime_dir))
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def _initialize(runtime_dir: Path) -> None:
    runtime_dir.mkdir(parents=True, exist_ok=True)
    # Separate from the index writer lock: indexers can initialize inside it.
    with index_lock(runtime_dir, lock_name="state-migration.lock"):
        _initialize_locked(runtime_dir)


def _initialize_locked(runtime_dir: Path) -> None:
    with _connection(runtime_dir) as connection:
        query_columns = _columns(connection, "queries")
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        legacy_queries = bool(query_columns & {"query", "filters", "client_name"})
        legacy_results = "query_results" in tables
        legacy_errors = "error" in _columns(connection, "index_runs") and connection.execute(
            "SELECT 1 FROM index_runs WHERE error IS NOT NULL LIMIT 1"
        ).fetchone() is not None
        backup_path = runtime_dir / PRIVACY_BACKUP_NAME
        pending_compaction = backup_path.exists() and connection.execute(
            "PRAGMA user_version"
        ).fetchone()[0] < 1
        scrub = legacy_queries or legacy_results or legacy_errors or pending_compaction
        if scrub:
            if not backup_path.exists():
                temporary_backup = backup_path.with_suffix(".sqlite3.tmp")
                with closing(sqlite3.connect(temporary_backup)) as backup:
                    connection.backup(backup)
                temporary_backup.replace(backup_path)
            warnings.warn(
                f"Privacy migration backup retained at {backup_path}; it may contain sensitive data. "
                "Remove it manually after verifying the migrated state.", RuntimeWarning, stacklevel=3,
            )
        connection.execute("BEGIN IMMEDIATE")
        if legacy_queries:
            connection.execute("ALTER TABLE queries RENAME TO legacy_queries")
        # executescript commits an existing transaction. Execute each DDL
        # statement separately so failed migration leaves the old schema usable.
        schema = """
            CREATE TABLE IF NOT EXISTS files (
                path TEXT PRIMARY KEY,
                content_hash TEXT NOT NULL,
                generation TEXT NOT NULL,
                point_count INTEGER NOT NULL,
                point_ids TEXT NOT NULL,
                completed_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS collection_files (
                collection_name TEXT NOT NULL,
                path TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                generation TEXT NOT NULL,
                point_count INTEGER NOT NULL,
                point_ids TEXT NOT NULL,
                completed_at TEXT NOT NULL,
                PRIMARY KEY (collection_name, path)
            );
            CREATE TABLE IF NOT EXISTS index_runs (
                id INTEGER PRIMARY KEY,
                started_at TEXT NOT NULL,
                collection_name TEXT,
                status TEXT NOT NULL,
                generation TEXT,
                added INTEGER NOT NULL,
                changed INTEGER NOT NULL,
                deleted INTEGER NOT NULL,
                error_code TEXT
            );
            CREATE TABLE IF NOT EXISTS queries (
                id INTEGER PRIMARY KEY,
                queried_at TEXT NOT NULL,
                elapsed_ms REAL,
                result_count INTEGER NOT NULL,
                rerank_requested INTEGER,
                rerank_applied INTEGER,
                error_code TEXT
            );
            """
        for statement in schema.split(";"):
            if statement.strip():
                connection.execute(statement)
        if legacy_queries:
            elapsed = "q.elapsed_ms" if "elapsed_ms" in query_columns else "NULL"
            count = "(SELECT COUNT(*) FROM query_results r WHERE r.query_id = q.id)" if legacy_results else "0"
            connection.execute(
                f"INSERT INTO queries(id, queried_at, elapsed_ms, result_count) "
                f"SELECT q.id, q.queried_at, {elapsed}, {count} FROM legacy_queries q"
            )
        if legacy_results:
            connection.execute("DROP TABLE query_results")
        if legacy_queries:
            connection.execute("DROP TABLE legacy_queries")
        _add_column_if_missing(connection, "index_runs", "error_code", "TEXT")
        _add_column_if_missing(connection, "index_runs", "collection_name", "TEXT")
        for column, definition in (("mtime_ns", "INTEGER"), ("size", "INTEGER"),
                                   ("metadata_signature", "TEXT"), ("indexing_version", "TEXT")):
            _add_column_if_missing(connection, "collection_files", column, definition)
        if "error" in _columns(connection, "index_runs"):
            connection.execute("UPDATE index_runs SET error = NULL WHERE error IS NOT NULL")
        if scrub:
            connection.commit()
            connection.execute("VACUUM")
        connection.execute("PRAGMA user_version = 1")
        _prune_operations(connection)


def positive_env_int(name: str, default: int, *, maximum: int = 2147483647) -> int:
    """Invalid or disabled limits fall back to the bounded default."""
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError:
        return default
    return value if 0 < value <= maximum else default


def _prune_operations(connection: sqlite3.Connection) -> None:
    cutoff = (datetime.fromisoformat(_now()) - timedelta(
        days=positive_env_int("KNOWLEDGE_QUERY_RETENTION_DAYS", 30, maximum=365000)
    )).isoformat(timespec="seconds")
    deleted = connection.execute(
        "DELETE FROM queries WHERE julianday(queried_at) < julianday(?)", (cutoff,)
    ).rowcount
    deleted += connection.execute(
        "DELETE FROM queries WHERE id IN (SELECT id FROM queries ORDER BY id DESC LIMIT -1 OFFSET ?)",
        (positive_env_int("KNOWLEDGE_QUERY_MAX_ROWS", 10000),),
    ).rowcount
    index_cutoff = (datetime.fromisoformat(_now()) - timedelta(
        days=positive_env_int("KNOWLEDGE_INDEX_RETENTION_DAYS", 30, maximum=365000)
    )).isoformat(timespec="seconds")
    # Latest status is permanent for each collection, including NULL legacy
    # rows. Only historical runs are subject to age and count retention.
    latest = "SELECT MAX(id) FROM index_runs GROUP BY collection_name"
    deleted += connection.execute(
        f"DELETE FROM index_runs WHERE id NOT IN ({latest}) AND julianday(started_at) < julianday(?)",
        (index_cutoff,),
    ).rowcount
    deleted += connection.execute(
        f"DELETE FROM index_runs WHERE id IN (SELECT id FROM index_runs "
        f"WHERE id NOT IN ({latest}) ORDER BY id DESC LIMIT -1 OFFSET ?)",
        (positive_env_int("KNOWLEDGE_INDEX_MAX_ROWS", 1000),),
    ).rowcount
    # Reuse freed pages for ordinary eviction; compact a substantial backlog.
    # Rewriting the full manifest DB for every query after reaching the row cap
    # would make bounded history unnecessarily expensive.
    if deleted and connection.execute("PRAGMA freelist_count").fetchone()[0] >= 32:
        connection.commit()
        connection.execute("VACUUM")


def _add_column_if_missing(
    connection: sqlite3.Connection, table: str, column: str, definition: str
) -> None:
    if column not in _columns(connection, table):
        connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def _columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {row["name"] for row in connection.execute(f"PRAGMA table_info({table})")}


def _database_path(runtime_dir: Path) -> Path:
    return runtime_dir / DATABASE_NAME


def _database_text(runtime_dir: Path) -> str:
    return _database_path(runtime_dir).read_bytes().decode("utf-8", errors="ignore")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
