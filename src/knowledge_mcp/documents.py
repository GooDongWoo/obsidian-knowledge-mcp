"""Discovery, parsing, and chunking for local Vault sources."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import pathspec
from pypdf import PdfReader
import yaml

from .config import Settings, _load_env_file


SEOUL = timezone(timedelta(hours=9))
SOURCE_SUFFIXES = {".md", ".txt", ".pdf"}
HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
MAX_FILE_SIZE_BYTES = 30 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class Section:
    text: str
    start_line: int | None = None
    end_line: int | None = None
    heading_path: tuple[str, ...] = ()
    page: int | None = None
    paragraph_index: int | None = None


@dataclass(frozen=True, slots=True)
class SourceDocument:
    path: Path
    source_path: str
    title: str
    document_type: str
    document_type_source: str
    file_type: str
    security_level: str
    created_at: str
    modified_at: str
    sections: tuple[Section, ...]
    status: str = "ready"
    error: str | None = None


@dataclass(frozen=True, slots=True)
class Chunk:
    source_path: str
    document: str
    embedding_text: str
    chunk_index: int
    document_type: str
    document_type_source: str
    file_type: str
    security_level: str
    created_at: str
    modified_at: str
    heading_path: tuple[str, ...] = ()
    start_line: int | None = None
    end_line: int | None = None
    page: int | None = None
    paragraph_index: int | None = None


def discover_sources(settings: Settings) -> list[Path]:
    """Return readable, supported files inside the Vault and outside ignores."""
    root = settings.vault_root.resolve()
    spec = _ignore_spec(settings.project_root)
    sources: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in SOURCE_SUFFIXES:
            continue
        try:
            resolved = path.resolve(strict=True)
            if resolved.stat().st_size > MAX_FILE_SIZE_BYTES:
                continue
            relative = resolved.relative_to(root)
            with resolved.open("rb"):
                pass
        except (OSError, ValueError):
            continue
        if not spec.match_file(relative.as_posix()):
            sources.append(resolved)
    return sorted(sources, key=lambda item: item.as_posix())


def parse_source(path: Path, settings: Settings) -> SourceDocument:
    """Parse one supported source without allowing it to escape the Vault."""
    root = settings.vault_root.resolve()
    resolved = path.resolve(strict=True)
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("source path must be inside the Vault") from exc
    if resolved.suffix.lower() not in SOURCE_SUFFIXES:
        raise ValueError(f"unsupported source type: {resolved.suffix}")

    file_type = resolved.suffix.lower().lstrip(".")
    if resolved.stat().st_size > MAX_FILE_SIZE_BYTES:
        document_type, document_type_source = _document_type(
            {}, "fallback", relative.as_posix(), settings.project_root
        )
        created_at, modified_at = _dates(resolved, {}, root, relative)
        return SourceDocument(
            path=resolved,
            source_path=relative.as_posix(),
            title=resolved.stem,
            document_type=document_type,
            document_type_source=document_type_source,
            file_type=file_type,
            security_level="public",
            created_at=created_at,
            modified_at=modified_at,
            sections=(),
            status="skipped_large_file",
            error="file size exceeds 30MB limit",
        )

    raw_text = ""
    metadata: dict[str, Any] = {}
    sections: tuple[Section, ...]
    status = "ready"
    error = None
    if file_type == "md":
        raw_text = resolved.read_text(encoding="utf-8")
        metadata, body_start = _frontmatter(raw_text)
        sections = _markdown_sections(raw_text.splitlines(), body_start)
        metadata_source = "frontmatter"

    elif file_type == "txt":
        raw_text = resolved.read_text(encoding="utf-8")
        metadata = _sidecar_metadata(resolved)
        sections = _text_sections(raw_text.splitlines())
        metadata_source = "sidecar"
    else:
        metadata = _sidecar_metadata(resolved)
        metadata_source = "sidecar"
        try:
            sections = _pdf_sections(resolved)
        except Exception as exc:  # pypdf exposes several file-format exceptions.
            sections = ()
            status = "skipped_error"
            error = str(exc)
        if status == "ready" and not sections:
            status = "skipped_no_text"

    document_type, document_type_source = _document_type(
        metadata, metadata_source, relative.as_posix(), settings.project_root
    )
    security_level = _security_level(metadata)
    created_at, modified_at = _dates(resolved, metadata, root, relative)
    title = str(metadata.get("title") or resolved.stem)
    return SourceDocument(
        path=resolved,
        source_path=relative.as_posix(),
        title=title,
        document_type=document_type,
        document_type_source=document_type_source,
        file_type=file_type,
        security_level=security_level,
        created_at=created_at,
        modified_at=modified_at,
        sections=sections,
        status=status,
        error=error,
    )


def chunk_document(
    document: SourceDocument,
    tokenizer: Any,
    max_tokens: int = 400,
    overlap_tokens: int = 60,
) -> list[Chunk]:
    """Split document text at section boundaries before token windows."""
    if max_tokens < 1 or overlap_tokens < 0 or overlap_tokens >= max_tokens:
        raise ValueError("require max_tokens > overlap_tokens >= 0")
    chunks: list[Chunk] = []
    groups = _section_groups(document.sections, tokenizer, max_tokens, document.file_type)
    for group in groups:
        prefix, content_max_tokens = _embedding_prefix(
            tokenizer, document.title, group.heading_path, max_tokens
        )
        effective_overlap = min(overlap_tokens, content_max_tokens - 1)
        tokens = list(tokenizer.encode(group.text))
        for start in range(0, len(tokens), content_max_tokens - effective_overlap):
            end = min(start + content_max_tokens, len(tokens))
            text = group.text if start == 0 and end == len(tokens) else _decode(tokenizer, tokens[start:end])
            chunks.append(
                Chunk(
                    source_path=document.source_path,
                    document=text,
                    embedding_text=_embedding_text(tokenizer, prefix, text, max_tokens),
                    chunk_index=len(chunks),
                    document_type=document.document_type,
                    document_type_source=document.document_type_source,
                    file_type=document.file_type,
                    security_level=document.security_level,
                    created_at=document.created_at,
                    modified_at=document.modified_at,
                    heading_path=group.heading_path,
                    start_line=group.start_line,
                    end_line=group.end_line,
                    page=group.page,
                    paragraph_index=group.paragraph_index,
                )
            )
            if end == len(tokens):
                break
    return chunks


def _ignore_spec(root: Path) -> pathspec.PathSpec:
    _load_env_file(root)
    lines: list[str] = []
    # 1. Base project .knowledgeignore
    ignore_file = root / ".knowledgeignore"
    if ignore_file.is_file():
        lines.extend(ignore_file.read_text(encoding="utf-8").splitlines())

    # 2. Local ignore file if present
    local_ignore = root / ".knowledgeignore.local"
    if local_ignore.is_file():
        lines.extend(local_ignore.read_text(encoding="utf-8").splitlines())

    # 3. Environment / .env variable KNOWLEDGE_EXTRA_IGNORES
    extra_ignores = os.environ.get("KNOWLEDGE_EXTRA_IGNORES", "")
    if extra_ignores:
        delimiter = ";" if ";" in extra_ignores else ("\n" if "\n" in extra_ignores else ",")
        for pattern in extra_ignores.split(delimiter):
            pattern = pattern.strip()
            if pattern:
                lines.append(pattern)

    return pathspec.PathSpec.from_lines("gitwildmatch", lines)


def _frontmatter(text: str) -> tuple[dict[str, Any], int]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, 0
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            loaded = yaml.safe_load("\n".join(lines[1:index])) or {}
            if not isinstance(loaded, dict):
                raise ValueError("frontmatter must be a mapping")
            return loaded, index + 1
    return {}, 0


def _sidecar_metadata(path: Path) -> dict[str, Any]:
    sidecar = path.with_name(path.name + ".meta.yaml")
    if not sidecar.is_file():
        return {}
    loaded = yaml.safe_load(sidecar.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError("sidecar metadata must be a mapping")
    return loaded


def _markdown_sections(lines: list[str], body_start: int) -> tuple[Section, ...]:
    headings: list[str] = []
    sections: list[Section] = []
    paragraph: list[str] = []
    start_line: int | None = None

    def flush(end_line: int) -> None:
        nonlocal paragraph, start_line
        text = "\n".join(paragraph).strip()
        if text and start_line is not None:
            sections.append(Section(text, start_line, end_line, tuple(headings)))
        paragraph, start_line = [], None

    for line_number, line in enumerate(lines, start=1):
        if line_number <= body_start:
            continue
        match = HEADING.match(line)
        if match:
            flush(line_number - 1)
            level, heading = len(match.group(1)), match.group(2).strip()
            headings[level - 1 :] = [heading]
        elif line.strip():
            if start_line is None:
                start_line = line_number
            paragraph.append(line)
        else:
            flush(line_number - 1)
    flush(len(lines))
    return tuple(sections)


def _text_sections(lines: list[str]) -> tuple[Section, ...]:
    sections: list[Section] = []
    paragraph: list[str] = []
    start_line: int | None = None

    def flush(end_line: int) -> None:
        nonlocal paragraph, start_line
        text = "\n".join(paragraph).strip()
        if text and start_line is not None:
            sections.append(Section(text, start_line, end_line))
        paragraph, start_line = [], None

    for line_number, line in enumerate(lines, start=1):
        if line.strip():
            if start_line is None:
                start_line = line_number
            paragraph.append(line)
        else:
            flush(line_number - 1)
    flush(len(lines))
    return tuple(sections)


def _pdf_sections(path: Path) -> tuple[Section, ...]:
    sections: list[Section] = []
    for page_number, page in enumerate(PdfReader(path).pages, start=1):
        text = page.extract_text() or ""
        for paragraph_index, paragraph in enumerate(re.split(r"\n\s*\n", text), start=1):
            stripped = paragraph.strip()
            if stripped:
                sections.append(Section(stripped, page=page_number, paragraph_index=paragraph_index))
    return tuple(sections)


def _document_type(
    metadata: dict[str, Any], metadata_source: str, relative_path: str, root: Path
) -> tuple[str, str]:
    explicit = metadata.get("document_type")
    if explicit:
        return str(explicit), metadata_source
    types_file = root / ".knowledge-types.yaml"
    if types_file.is_file():
        config = yaml.safe_load(types_file.read_text(encoding="utf-8")) or {}
        for name, details in (config.get("types") or {}).items():
            for pattern in (details or {}).get("paths", []):
                if pathspec.PathSpec.from_lines("gitwildmatch", [pattern]).match_file(relative_path):
                    return str(name), "path_rule"
    lower_path = relative_path.casefold()
    for keyword, name in (("브레인스토밍", "brainstorming"), ("아이디어", "brainstorming"), ("수상", "award"), ("상장", "award"), ("경험", "experience")):
        if keyword in lower_path:
            return name, "inference"
    return "other", "fallback"


def _security_level(metadata: dict[str, Any]) -> str:
    value = metadata.get("security", "public")
    if not isinstance(value, str) or value not in {"public", "private"}:
        raise ValueError(f"unknown security level: {value!r}")
    return value


def _dates(path: Path, metadata: dict[str, Any], root: Path, relative: Path) -> tuple[str, str]:
    git_dates = _git_dates(root, relative)
    stat = path.stat()
    created = _normalise_date(metadata.get("created_at"))
    modified = _normalise_date(metadata.get("modified_at"))
    return (
        created or git_dates[0] or _normalise_date(datetime.fromtimestamp(stat.st_ctime, SEOUL)),
        modified or git_dates[1] or _normalise_date(datetime.fromtimestamp(stat.st_mtime, SEOUL)),
    )


def _git_dates(root: Path, relative: Path) -> tuple[str | None, str | None]:
    creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "log", "--follow", "--format=%aI", "--", relative.as_posix()],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
            creationflags=creationflags,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None, None
    dates = [_normalise_date(item) for item in completed.stdout.splitlines() if item]
    return (dates[-1], dates[0]) if dates else (None, None)


def _normalise_date(value: Any) -> str | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date):
        parsed = datetime.combine(value, datetime.min.time())
    else:
        raw = str(value).replace("Z", "+00:00")
        parsed = datetime.fromisoformat(raw)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=SEOUL)
    return parsed.astimezone(SEOUL).isoformat(timespec="seconds")


@dataclass(frozen=True, slots=True)
class _SectionGroup:
    text: str
    start_line: int | None
    end_line: int | None
    heading_path: tuple[str, ...]
    page: int | None
    paragraph_index: int | None


def _section_groups(
    sections: tuple[Section, ...], tokenizer: Any, max_tokens: int, file_type: str
) -> list[_SectionGroup]:
    groups: list[_SectionGroup] = []
    for section in sections:
        if (
            groups
            and file_type != "pdf"
            and groups[-1].heading_path == section.heading_path
            and len(tokenizer.encode(groups[-1].text)) + len(tokenizer.encode(section.text)) <= max_tokens
        ):
            previous = groups.pop()
            groups.append(
                _SectionGroup(
                    previous.text + "\n\n" + section.text,
                    previous.start_line,
                    section.end_line,
                    previous.heading_path,
                    None,
                    None,
                )
            )
        else:
            groups.append(
                _SectionGroup(
                    section.text,
                    section.start_line,
                    section.end_line,
                    section.heading_path,
                    section.page,
                    section.paragraph_index,
                )
            )
    return groups


def _decode(tokenizer: Any, tokens: list[Any]) -> str:
    return tokenizer.decode(tokens).strip()


def _embedding_prefix(
    tokenizer: Any, title: str, heading_path: tuple[str, ...], max_tokens: int
) -> tuple[str, int]:
    prefix = "\n".join(part for part in (title, " > ".join(heading_path)) if part)
    prefix_tokens = list(tokenizer.encode(prefix))
    reserved = min(len(prefix_tokens), max_tokens - 1)
    if reserved < len(prefix_tokens):
        prefix = _decode(tokenizer, prefix_tokens[:reserved])
    return prefix, max_tokens - reserved


def _embedding_text(tokenizer: Any, prefix: str, text: str, max_tokens: int) -> str:
    embedding = "\n".join(part for part in (prefix, text) if part)
    tokens = list(tokenizer.encode(embedding))
    return embedding if len(tokens) <= max_tokens else _decode(tokenizer, tokens[:max_tokens])
