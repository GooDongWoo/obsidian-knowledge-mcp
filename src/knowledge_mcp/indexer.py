"""Incremental, per-source indexing with recoverable generation commits."""

import asyncio
from contextlib import nullcontext
from dataclasses import dataclass, field
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Callable

from .config import Settings
from .documents import SourceDocument, chunk_document, discover_sources, parse_source
from .qdrant_store import BM25_OPTIONS, KnowledgeStore
from .state import Manifest, OperationLog, async_index_lock, run_blocking


# Bump the implementation version whenever parsing/chunking semantics change.
PARSER_CHUNKER_SCHEMA = "documents-v1:400:60"


@dataclass(slots=True)
class IndexRunSummary:
    added: int = 0
    changed: int = 0
    deleted: int = 0
    unchanged: int = 0
    skipped: int = 0
    failed: int = 0
    error_codes: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        return "partial" if self.failed else "completed"


class KnowledgeIndexer:
    def __init__(
        self, settings: Settings, store: KnowledgeStore, tokenizer: Any, *,
        manifest: Manifest | None = None, operation_log: OperationLog | None = None,
        parser: Callable[[Path, Settings], SourceDocument] = parse_source,
    ) -> None:
        self.settings = settings
        self.store = store
        self.tokenizer = tokenizer
        self.manifest = manifest if manifest is not None else Manifest(settings.runtime_dir, settings.collection_name)
        self.operation_log = operation_log if operation_log is not None else OperationLog(settings.runtime_dir)
        self.parser = parser
        self.progress = {"stage": "idle", "completed": 0, "total": 0}

    async def sync(self, *, assume_locked: bool = False, force_full_hash: bool = False) -> IndexRunSummary:
        """Serialize writers unless caller holds the index lock; store no document bodies."""
        summary = IndexRunSummary()
        self.progress = {"stage": "waiting", "completed": 0, "total": 0}
        try:
            return await self._sync(summary, assume_locked=assume_locked, force_full_hash=force_full_hash)
        except asyncio.CancelledError:
            # Cancellation can arrive between generation swap and manifest
            # commit. Preserve recovery inputs and don't leave an old success
            # as the reported outcome of this interrupted run.
            self._failure(summary, "sync_cancelled")
            await run_blocking(self._record, summary)
            raise
        finally:
            self.progress["stage"] = "idle"

    async def _sync(self, summary: IndexRunSummary, *, assume_locked: bool, force_full_hash: bool) -> IndexRunSummary:
        async with nullcontext() if assume_locked else async_index_lock(self.settings.runtime_dir):
            stage = "schema_check_failed"
            try:
                await self.store.ensure_schema()
                stage = "discovery_failed"
                sources = await run_blocking(self._sources)
                self.progress = {"stage": "indexing", "completed": 0, "total": len(sources)}
                stage = "manifest_read_failed"
                completed = await run_blocking(self.manifest.completed_files)
                stage = "generation_check_failed"
                inventory = await self.store.collection_inventory()
            except Exception:
                self._failure(summary, stage)
                await run_blocking(self._record, summary)
                return summary

            cleanup_allowed = True
            inventory_dirty = False
            for source_path, path in sources.items():
                stage = "source_read_failed"
                try:
                    previous = completed.get(source_path)
                    source_state = await run_blocking(self._source_state, path)
                    matches = previous and inventory.matches(
                        source_path, previous.generation, previous.point_count, previous.point_ids,
                    )
                    if not force_full_hash and matches and all(
                        getattr(previous, key) == value for key, value in source_state.items()
                    ):
                        summary.unchanged += 1
                        continue
                    file_hash, generation = await run_blocking(self._fingerprint, path)
                    stage = "source_changed_during_parse"
                    if await run_blocking(self._source_state, path) != source_state:
                        self._failure(summary, stage)
                        continue
                    if matches and previous.generation == generation:
                        stage = "manifest_commit_failed"
                        await run_blocking(self.manifest.mark_complete, source_path, file_hash, generation,
                                          previous.point_count, previous.point_ids, **source_state)
                        summary.unchanged += 1
                        continue
                    stage = "parse_failed"
                    document = await run_blocking(self.parser, path, self.settings)
                    if document.status not in {"ready", "skipped_no_text", "skipped_large_file"}:
                        self._failure(summary, "parse_failed")
                        continue
                    stage = "chunk_failed"
                    chunks = await run_blocking(chunk_document, document, self.tokenizer, max_tokens=400, overlap_tokens=60)
                    stage = "source_changed_during_parse"
                    if (await run_blocking(self._fingerprint, path) != (file_hash, generation)
                            or await run_blocking(self._source_state, path) != source_state):
                        self._failure(summary, stage)
                        continue
                    stage = "qdrant_replace_failed"
                    point_ids = await self.store.replace_generation(
                        source_path, generation, chunks, file_hash=file_hash,
                    )
                    inventory.replace(source_path, generation, point_ids)
                    stage = "manifest_commit_failed"
                    await run_blocking(self.manifest.mark_complete, source_path, file_hash, generation,
                                      len(point_ids), point_ids, **source_state)
                    if document.status in {"skipped_no_text", "skipped_large_file"}:
                        summary.skipped += 1
                    elif previous:
                        summary.changed += 1
                    else:
                        summary.added += 1
                except Exception:
                    self._failure(summary, stage)
                    if stage == "manifest_commit_failed":
                        cleanup_allowed = False
                    elif stage == "qdrant_replace_failed":
                        # A failed upsert/delete may have written some points.
                        # Refresh before cleanup instead of trusting the snapshot.
                        inventory_dirty = True
                finally:
                    self.progress["completed"] += 1

            self.progress["stage"] = "cleanup"
            for source_path in sorted(set(completed) - set(sources)):
                stage = "qdrant_delete_failed"
                try:
                    await self.store.delete_source(source_path)
                    inventory.remove(source_path)
                    stage = "manifest_remove_failed"
                    await run_blocking(self.manifest.remove, source_path)
                    summary.deleted += 1
                except Exception:
                    self._failure(summary, stage)
                    if stage == "qdrant_delete_failed":
                        inventory_dirty = True

            # A swap may delete the old generation and crash before committing
            # SQLite. Reindex it first; on failed recovery keep surviving points.
            if cleanup_allowed:
                stage = "orphan_cleanup_failed"
                try:
                    if inventory_dirty:
                        inventory = await self.store.collection_inventory()
                    committed = await run_blocking(self.manifest.completed_files)
                    for source_path, entry in committed.items():
                        if not inventory.matches(
                            source_path, entry.generation, entry.point_count, entry.point_ids,
                        ):
                            cleanup_allowed = False
                            break
                    if cleanup_allowed:
                        await self.store.cleanup_orphans(
                            {path: entry.generation for path, entry in committed.items()}, inventory=inventory,
                        )
                except Exception:
                    self._failure(summary, stage)
            await run_blocking(self._record, summary)
        return summary

    def _sources(self) -> dict[str, Path]:
        root = self.settings.vault_root.resolve()
        return {path.relative_to(root).as_posix(): path for path in discover_sources(self.settings)}

    def _fingerprint(self, path: Path) -> tuple[str, str]:
        file_hash = sha256(path.read_bytes()).hexdigest()
        inputs = {"file_hash": file_hash, **self._metadata_inputs(path), **self._indexing_inputs()}
        generation = self._signature(inputs)
        return file_hash, generation

    def _source_state(self, path: Path) -> dict[str, Any]:
        stat = path.stat()
        return {"mtime_ns": stat.st_mtime_ns, "size": stat.st_size,
                "metadata_signature": self._signature(self._metadata_inputs(path)),
                "indexing_version": self._signature(self._indexing_inputs())}

    def _metadata_inputs(self, path: Path) -> dict[str, Any]:
        sidecar = path.with_name(path.name + ".meta.yaml")
        types = self.settings.project_root / ".knowledge-types.yaml"
        return {
            "sidecar_hash": sha256(sidecar.read_bytes()).hexdigest() if sidecar.is_file() else None,
            "classification_hash": sha256(types.read_bytes()).hexdigest() if types.is_file() else None,
        }

    def _indexing_inputs(self) -> dict[str, Any]:
        return {
            "parser_chunker_schema": PARSER_CHUNKER_SCHEMA,
            "dense_model": self.settings.dense_model,
            "bm25_model": "qdrant/bm25",
            "bm25_options": BM25_OPTIONS,
        }

    @staticmethod
    def _signature(inputs: dict[str, Any]) -> str:
        return sha256(json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()

    @staticmethod
    def _failure(summary: IndexRunSummary, code: str) -> None:
        summary.failed += 1
        if code not in summary.error_codes:
            summary.error_codes.append(code)

    def _record(self, summary: IndexRunSummary) -> None:
        self.operation_log.record_index(
            collection_name=self.settings.collection_name,
            generation=None, added=summary.added, changed=summary.changed, deleted=summary.deleted,
            status=summary.status, error_code=summary.error_codes[0] if summary.error_codes else None,
        )
