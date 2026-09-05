from __future__ import annotations

from types import SimpleNamespace

from services.worker.connectors import pdf
from services.worker.jobs.ingest_pdf import ingest_pdf
from services.worker.dedup.hashing import content_hash
from services.worker.parsing import parse_document


def _reader(monkeypatch):
    pages = [
        SimpleNamespace(extract_text=lambda: "  Agent\x00 tools\n中文\tMCP  "),
        SimpleNamespace(extract_text=lambda: "\x00 \n\x00"),
        SimpleNamespace(extract_text=lambda: "Second page"),
    ]
    monkeypatch.setattr("PyPDF2.PdfReader", lambda *_args: SimpleNamespace(pages=pages))


def test_pdf_parser_cleans_nul_and_preserves_page_citations(monkeypatch):
    _reader(monkeypatch)
    document, _ = parse_document(b"pdf", None, name="guide.pdf", media_type="application/pdf")
    assert [(b.page, b.text) for b in document.blocks] == [
        (1, "Agent tools\n中文\tMCP"), (3, "Second page"),
    ]
    assert [b.block_index for b in document.blocks] == [0, 1]


def test_pdf_chunks_hash_normalized_text_before_storage(monkeypatch):
    _reader(monkeypatch)
    monkeypatch.setattr("services.worker.jobs.ingest_pdf.fetch_pdf_bytes", lambda _path: b"pdf")
    chunks = list(ingest_pdf("guide.pdf", "doc-1", "engineering"))
    assert chunks
    assert {c.page for c in chunks} == {1, 3}
    for chunk in chunks:
        assert "\x00" not in chunk.text
        assert chunk.checksum == content_hash(chunk.text)


def test_legacy_pdf_page_helper_uses_the_same_normalization(monkeypatch):
    _reader(monkeypatch)
    monkeypatch.setattr(pdf, "fetch_pdf_bytes", lambda _path: b"pdf")
    assert pdf.fetch_pdf_pages("guide.pdf") == [
        (1, "Agent tools\n中文\tMCP"), (3, "Second page"),
    ]
