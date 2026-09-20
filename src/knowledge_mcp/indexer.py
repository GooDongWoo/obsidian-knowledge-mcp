"""Incremental, per-source indexing with recoverable generation commits."""

from contextlib import nullcontext
from dataclasses import dataclass, field
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Callable

from .config import Settings
from .documents import SourceDocument, chunk_document, discover_sources, parse_source
from .qdrant_store import BM25_OPTIONS, KnowledgeStore
from .state import Manifest, OperationLog, index_lock


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

    async def sync(self, *, assume_locked: bool = False) -> IndexRunSummary:
        """Serialize writers unless caller holds the index lock; store no document bodies."""
        summary = IndexRunSummary()
        with nullcontext() if assume_locked else index_lock(self.settings.runtime_dir):
            stage = "schema_check_failed"
            try:
                await self.store.ensure_schema()
                stage = "discovery_failed"
                sources = {
                    path.relative_to(self.settings.vault_root.resolve()).as_posix(): path
                    for path in discover_sources(self.settings)
                }
                stage = "manifest_read_failed"
                completed = self.manifest.completed_files()
            except Exception:
                self._failure(summary, stage)
                self._record(summary)
                return summary

            cleanup_allowed = True
            for source_path, path in sources.items():
                stage = "source_read_failed"
                try:
                    file_hash, generation = self._fingerprint(path)
                    previous = completed.get(source_path)
                    stage = "generation_check_failed"
                    if previous and previous.generation == generation and await self.store.generation_matches(
                        source_path, previous.generation, previous.point_count, previous.point_ids,
                    ):
                        summary.unchanged += 1
                        continue
                    stage = "parse_failed"
                    document = self.parser(path, self.settings)
                    if document.status not in {"ready", "skipped_no_text", "skipped_large_file"}:
                        self._failure(summary, "parse_failed")
                        continue
                    stage = "chunk_failed"
                    chunks = chunk_document(document, self.tokenizer, max_tokens=400, overlap_tokens=60)
                    stage = "source_changed_during_parse"
                    if self._fingerprint(path) != (file_hash, generation):
                        self._failure(summary, stage)
                        continue
                    stage = "qdrant_replace_failed"
                    point_ids = await self.store.replace_generation(
                        source_path, generation, chunks, file_hash=file_hash,
                    )
                    stage = "manifest_commit_failed"
                    self.manifest.mark_complete(source_path, file_hash, generation, len(point_ids), point_ids)
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

            for source_path in sorted(set(completed) - set(sources)):
                stage = "qdrant_delete_failed"
                try:
                    await self.store.delete_source(source_path)
                    stage = "manifest_remove_failed"
                    self.manifest.remove(source_path)
                    summary.deleted += 1
                except Exception:
                    self._failure(summary, stage)

            # A swap may delete the old generation and crash before committing
            # SQLite. Reindex it first; on failed recovery keep surviving points.
            if cleanup_allowed:
                stage = "orphan_cleanup_failed"
                try:
                    committed = self.manifest.completed_files()
                    for source_path, entry in committed.items():
                        if not await self.store.generation_matches(
                            source_path, entry.generation, entry.point_count, entry.point_ids,
                        ):
                            cleanup_allowed = False
                            break
                    if cleanup_allowed:
                        await self.store.cleanup_orphans({path: entry.generation for path, entry in committed.items()})
                except Exception:
                    self._failure(summary, stage)
            self._record(summary)
        return summary

    def _fingerprint(self, path: Path) -> tuple[str, str]:
        file_hash = sha256(path.read_bytes()).hexdigest()
        sidecar = path.with_name(path.name + ".meta.yaml")
        types = self.settings.project_root / ".knowledge-types.yaml"
        inputs = {
            "file_hash": file_hash,
            "sidecar_hash": sha256(sidecar.read_bytes()).hexdigest() if sidecar.is_file() else None,
            "classification_hash": sha256(types.read_bytes()).hexdigest() if types.is_file() else None,
            "parser_chunker_schema": PARSER_CHUNKER_SCHEMA,
            "dense_model": self.settings.dense_model,
            "bm25_model": "qdrant/bm25",
            "bm25_options": BM25_OPTIONS,
        }
        generation = sha256(json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        return file_hash, generation

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
