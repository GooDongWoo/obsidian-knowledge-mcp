from pathlib import Path
import shutil
import subprocess

import pytest

from knowledge_mcp.config import Settings
from knowledge_mcp.documents import chunk_document, discover_sources, parse_source


FIXTURES = Path(__file__).parent / "fixtures"


class WhitespaceTokenizer:
    def encode(self, text):
        return text.split()

    def decode(self, tokens):
        return " ".join(tokens)


@pytest.fixture
def settings(tmp_path):
    vault = tmp_path / "vault"
    project = tmp_path / "project"
    fixtures = vault / "fixtures"
    fixtures.mkdir(parents=True)
    project.mkdir(parents=True)
    for source in FIXTURES.iterdir():
        shutil.copy(source, fixtures / source.name)
    return Settings(
        vault_root=vault,
        runtime_dir=vault / ".knowledge",
        qdrant_url="http://127.0.0.1:6333",
        collection_name="test",
        dense_model="dragonkue/BGE-m3-ko",
        client_name="pytest",
        project_root=project,
    )


def test_markdown_metadata_and_lines(settings):
    doc = parse_source(settings.vault_root / "fixtures/sample.md", settings)

    assert doc.document_type == "brainstorming"
    assert doc.document_type_source == "frontmatter"
    assert doc.security_level == "private"
    assert doc.created_at == "2025-03-01T00:00:00+09:00"
    assert doc.sections[0].start_line == 11
    assert doc.sections[0].heading_path == ("Product thinking",)


def test_unmarked_text_is_public(settings):
    doc = parse_source(settings.vault_root / "fixtures/sample.txt", settings)

    assert doc.security_level == "public"
    assert doc.document_type == "other"
    assert doc.document_type_source == "fallback"
    assert doc.sections[0].start_line == 1


def test_git_timestamp_failure_falls_back_to_filesystem_dates(settings, monkeypatch):
    source = settings.vault_root / "note.txt"
    source.write_text("Text", encoding="utf-8")

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("git", 5)

    monkeypatch.setattr("knowledge_mcp.documents.subprocess.run", timeout)

    doc = parse_source(source, settings)

    assert doc.created_at.endswith("+09:00")
    assert doc.modified_at.endswith("+09:00")


def test_sidecar_metadata_overrides_path_rule_and_unknown_security_rejects(settings):
    source = settings.vault_root / "10_Daily" / "entry.txt"
    source.parent.mkdir(parents=True)
    source.write_text("A public-looking entry.", encoding="utf-8")
    source.with_name("entry.txt.meta.yaml").write_text(
        "document_type: experience\nsecurity: private\n", encoding="utf-8"
    )

    doc = parse_source(source, settings)

    assert doc.document_type == "experience"
    assert doc.document_type_source == "sidecar"
    assert doc.security_level == "private"

    source.with_name("entry.txt.meta.yaml").write_text("security: internal\n", encoding="utf-8")
    with pytest.raises(ValueError, match="security"):
        parse_source(source, settings)


def test_non_string_security_metadata_is_rejected(settings):
    source = settings.vault_root / "note.txt"
    source.write_text("Text", encoding="utf-8")
    source.with_name("note.txt.meta.yaml").write_text("security:\n  - private\n", encoding="utf-8")

    with pytest.raises(ValueError, match="security"):
        parse_source(source, settings)


def test_discovery_applies_ignore_rules_and_does_not_escape_vault(settings):
    included = settings.vault_root / "notes.md"
    ignored = settings.vault_root / "private.txt"
    ignored_dir = settings.vault_root / ".obsidian" / "config.md"
    included.write_text("included", encoding="utf-8")
    ignored.write_text("ignored", encoding="utf-8")
    ignored_dir.parent.mkdir()
    ignored_dir.write_text("ignored", encoding="utf-8")
    (settings.project_root / ".knowledgeignore").write_text("private.txt\n.obsidian/\n", encoding="utf-8")

    sources = discover_sources(settings)
    assert included in sources
    assert ignored not in sources
    assert ignored_dir not in sources


def test_configured_path_rule_precedes_limited_name_inference(settings):
    source = settings.vault_root / "10_Daily" / "수상-notes.txt"
    source.parent.mkdir(parents=True)
    source.write_text("Daily note", encoding="utf-8")
    (settings.project_root / ".knowledge-types.yaml").write_text(
        "types:\n  diary:\n    paths:\n      - '10_Daily/**'\n", encoding="utf-8"
    )

    doc = parse_source(source, settings)

    assert doc.document_type == "diary"
    assert doc.document_type_source == "path_rule"


def test_limited_path_name_inference_has_its_own_provenance(settings):
    source = settings.vault_root / "수상-notes.txt"
    source.write_text("Award note", encoding="utf-8")

    doc = parse_source(source, settings)

    assert doc.document_type == "award"
    assert doc.document_type_source == "inference"


def test_chunks_keep_source_lines_and_prefix_only_embedding_text(settings):
    doc = parse_source(settings.vault_root / "fixtures/sample.md", settings)

    chunks = chunk_document(doc, WhitespaceTokenizer(), max_tokens=12, overlap_tokens=2)

    assert chunks[0].start_line == 11
    assert chunks[0].document.startswith("First paragraph")
    assert chunks[0].embedding_text.startswith("Source fixture\nProduct thinking\n")
    assert "Source fixture" not in chunks[0].document
    assert all(len(WhitespaceTokenizer().encode(chunk.document)) <= 12 for chunk in chunks)
    assert chunks[0].document.split()[-2:] == chunks[1].document.split()[:2]


def test_default_model_window_is_400_with_60_token_body_overlap(settings):
    tokenizer = WhitespaceTokenizer()
    source = settings.vault_root / "long.txt"
    source.write_text(" ".join(f"word{i}" for i in range(401)), encoding="utf-8")

    chunks = chunk_document(parse_source(source, settings), tokenizer)

    assert [len(tokenizer.encode(chunk.document)) for chunk in chunks] == [399, 62]
    assert chunks[0].document.split()[-60:] == chunks[1].document.split()[:60]
    assert all(len(tokenizer.encode(chunk.embedding_text)) <= 400 for chunk in chunks)


def test_embedding_text_including_long_title_and_heading_stays_within_limit(settings):
    tokenizer = WhitespaceTokenizer()
    source = settings.vault_root / "long-context.md"
    title = " ".join(f"title{i}" for i in range(180))
    heading = " ".join(f"heading{i}" for i in range(20))
    body = " ".join(f"body{i}" for i in range(250))
    source.write_text(f"---\ntitle: '{title}'\n---\n\n# {heading}\n\n{body}\n", encoding="utf-8")

    chunks = chunk_document(parse_source(source, settings), tokenizer)

    assert all(len(tokenizer.encode(chunk.embedding_text)) <= 400 for chunk in chunks)
    assert all(chunk.embedding_text.startswith(title) for chunk in chunks)


def test_empty_pdf_returns_a_skipped_no_text_document(settings):
    from pypdf import PdfWriter

    source = settings.vault_root / "empty.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    with source.open("wb") as output:
        writer.write(output)

    document = parse_source(source, settings)

    assert document.status == "skipped_no_text"
    assert document.sections == ()
