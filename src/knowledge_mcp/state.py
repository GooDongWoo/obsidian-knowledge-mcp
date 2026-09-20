"""Durable, body-free local state for incremental indexing and operations."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import msvcrt
from pathlib import Path
import re
import sqlite3
import time
from typing import Iterator, Mapping, Sequence


DATABASE_NAME = "state.sqlite3"
DEFAULT_COLLECTION_NAME = "obsidian_knowledge_v1"
ERROR_CODE = re.compile(r"^[a-z][a-z0-9_]*$")


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
    ) -> None:
        with _connection(self.runtime_dir) as connection:
            connection.execute(
                """
                INSERT INTO collection_files(collection_name, path, content_hash, generation, point_count, point_ids, completed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(collection_name, path) DO UPDATE SET
                    content_hash = excluded.content_hash,
                    generation = excluded.generation,
                    point_count = excluded.point_count,
                    point_ids = excluded.point_ids,
                    completed_at = excluded.completed_at
                """,
                (self.collection_name, path, content_hash, generation, point_count, json.dumps(list(point_ids)), _now()),
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
                )
                for row in connection.execute(
                    "SELECT path, content_hash, generation, point_count, point_ids "
                    "FROM collection_files WHERE collection_name = ?",
                    (self.collection_name,),
                )
            }

    def database_text_for_test(self) -> str:
        return _database_text(self.runtime_dir)


class OperationLog:
    """Stores operational metadata, never indexed or returned document bodies."""

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

    def record_query(
        self,
        *,
        query: str,
        filters: Mapping[str, object] | None = None,
        results: Sequence[Mapping[str, object]],
        client_name: str | None = None,
        elapsed_ms: float | None = None,
    ) -> None:
        with _connection(self.runtime_dir) as connection:
            cursor = connection.execute(
                "INSERT INTO queries(queried_at, query, filters, client_name, elapsed_ms) VALUES (?, ?, ?, ?, ?)",
                (_now(), query, json.dumps(filters or {}, sort_keys=True), client_name, elapsed_ms),
            )
            query_id = cursor.lastrowid
            connection.executemany(
                """
                INSERT INTO query_results(query_id, rank, point_id, source_path, score, security_level)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        query_id,
                        rank,
                        result.get("point_id"),
                        result.get("source_path"),
                        result.get("score"),
                        result.get("security_level"),
                    )
                    for rank, result in enumerate(results, start=1)
                ],
            )

    def index_status(self, collection_name: str | None = None) -> dict[str, object]:
        with _connection(self.runtime_dir) as connection:
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
def index_lock(runtime_dir: Path) -> Iterator[None]:
    """Serialize indexers with a Windows advisory lock, releasing it always."""
    directory = Path(runtime_dir)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "index.lock").open("a+b") as lock_file:
        lock_file.seek(0)
        if not lock_file.read(1):
            lock_file.seek(0)
            lock_file.write(b"0")
            lock_file.flush()
        lock_file.seek(0)
        while True:
            try:
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                break
            except OSError:
                time.sleep(0.1)
        try:
            yield
        finally:
            lock_file.seek(0)
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)


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
    scrubbed_legacy_errors = False
    with _connection(runtime_dir) as connection:
        connection.executescript(
            """
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
                query TEXT NOT NULL,
                filters TEXT NOT NULL DEFAULT '{}',
                client_name TEXT,
                elapsed_ms REAL
            );
            CREATE TABLE IF NOT EXISTS query_results (
                id INTEGER PRIMARY KEY,
                query_id INTEGER NOT NULL REFERENCES queries(id),
                rank INTEGER NOT NULL,
                point_id TEXT,
                source_path TEXT,
                score REAL,
                security_level TEXT
            );
            """
        )
        _add_column_if_missing(connection, "queries", "filters", "TEXT NOT NULL DEFAULT '{}'")
        _add_column_if_missing(connection, "queries", "client_name", "TEXT")
        _add_column_if_missing(connection, "queries", "elapsed_ms", "REAL")
        _add_column_if_missing(connection, "index_runs", "error_code", "TEXT")
        _add_column_if_missing(connection, "index_runs", "collection_name", "TEXT")
        if "error" in _columns(connection, "index_runs"):
            scrubbed_legacy_errors = connection.execute(
                "UPDATE index_runs SET error = NULL WHERE error IS NOT NULL"
            ).rowcount > 0
    if scrubbed_legacy_errors:
        with _connection(runtime_dir) as connection:
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
