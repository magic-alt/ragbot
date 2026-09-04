from __future__ import annotations

import io
from typing import Any

from .models import DocumentBlock, NormalizedDocument
from .protocol import ParserSpec


class RagbotParser:
    """Compatibility parser for plain text, HTML and legacy PyPDF2 PDFs."""

    def __init__(self, spec: ParserSpec) -> None:
        self._spec = spec

    @property
    def spec(self) -> ParserSpec:
        return self._spec

    def parse(
        self,
        data: bytes,
        *,
        name: str,
        media_type: str = "application/octet-stream",
        uri: str | None = None,
    ) -> NormalizedDocument:
        strategy = self.spec.strategy
        if strategy == "pypdf2":
            return self._parse_pdf(data, name=name, media_type=media_type, uri=uri)
        if strategy == "html":
            return self._parse_html(data, name=name, media_type=media_type, uri=uri)
        return self._parse_text(data, name=name, media_type=media_type, uri=uri)

    def _parse_pdf(self, data: bytes, *, name: str, media_type: str, uri: str | None) -> NormalizedDocument:
        try:
            from PyPDF2 import PdfReader
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError("ragbot/pypdf2 parser requires PyPDF2; install ragbot[worker]") from exc
        blocks: list[DocumentBlock] = []
        reader = PdfReader(io.BytesIO(data))
        for page_number, page in enumerate(reader.pages, 1):
            text = (page.extract_text() or "").strip()
            if text:
                blocks.append(
                    DocumentBlock(
                        block_index=len(blocks),
                        text=text,
                        kind="page",
                        page=page_number,
                    )
                )
        return NormalizedDocument(name=name, media_type=media_type, uri=uri, blocks=blocks)

    def _parse_text(self, data: bytes, *, name: str, media_type: str, uri: str | None) -> NormalizedDocument:
        encoding = str(self.spec.options.get("encoding") or "utf-8")
        text = data.decode(encoding, errors="replace").strip()
        blocks = [DocumentBlock(block_index=0, text=text, kind="text")] if text else []
        return NormalizedDocument(name=name, media_type=media_type, uri=uri, blocks=blocks)

    def _parse_html(self, data: bytes, *, name: str, media_type: str, uri: str | None) -> NormalizedDocument:
        try:
            from bs4 import BeautifulSoup
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError("ragbot/html parser requires beautifulsoup4; install ragbot[worker]") from exc
        encoding = str(self.spec.options.get("encoding") or "utf-8")
        html = data.decode(encoding, errors="replace")
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        blocks: list[DocumentBlock] = []
        current_section: str | None = None
        selectors = ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre", "table"]
        for node in soup.find_all(selectors):
            text = node.get_text(" ", strip=True)
            if not text:
                continue
            tag_name = str(getattr(node, "name", "") or "").lower()
            if tag_name.startswith("h") and len(tag_name) == 2 and tag_name[1].isdigit():
                current_section = text
                kind = "heading"
            elif tag_name == "table":
                kind = "table"
            elif tag_name == "pre":
                kind = "code"
            elif tag_name == "li":
                kind = "list_item"
            else:
                kind = "paragraph"
            blocks.append(
                DocumentBlock(
                    block_index=len(blocks),
                    text=text,
                    kind=kind,
                    section=current_section,
                )
            )
        if not blocks:
            text = soup.get_text("\n", strip=True)
            if text:
                blocks.append(DocumentBlock(block_index=0, text=text, kind="text"))
        return NormalizedDocument(name=name, media_type=media_type, uri=uri, blocks=blocks)


class PyMuPDFParser:
    def __init__(self, spec: ParserSpec) -> None:
        self._spec = spec

    @property
    def spec(self) -> ParserSpec:
        return self._spec

    def parse(
        self,
        data: bytes,
        *,
        name: str,
        media_type: str = "application/pdf",
        uri: str | None = None,
    ) -> NormalizedDocument:
        try:
            import pymupdf
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("pymupdf/blocks parser requires PyMuPDF; install ragbot[parser-pymupdf]") from exc
        blocks: list[DocumentBlock] = []
        sort = bool(self.spec.options.get("sort", True))
        document = pymupdf.open(stream=data, filetype="pdf")
        try:
            for page_number, page in enumerate(document, 1):
                page_blocks = page.get_text("blocks", sort=sort)
                added = 0
                for raw in page_blocks:
                    if len(raw) < 5:
                        continue
                    block_type = int(raw[6]) if len(raw) > 6 and raw[6] is not None else 0
                    if block_type != 0:
                        continue
                    text = str(raw[4] or "").strip()
                    if not text:
                        continue
                    bbox = tuple(float(value) for value in raw[:4])
                    blocks.append(
                        DocumentBlock(
                            block_index=len(blocks),
                            text=text,
                            kind="text_block",
                            page=page_number,
                            bbox=(bbox[0], bbox[1], bbox[2], bbox[3]),
                        )
                    )
                    added += 1
                if added == 0:
                    text = str(page.get_text("text", sort=sort) or "").strip()
                    if text:
                        blocks.append(
                            DocumentBlock(
                                block_index=len(blocks),
                                text=text,
                                kind="page",
                                page=page_number,
                            )
                        )
        finally:
            document.close()
        return NormalizedDocument(name=name, media_type=media_type, uri=uri, blocks=blocks)


class DoclingParser:
    def __init__(self, spec: ParserSpec) -> None:
        self._spec = spec

    @property
    def spec(self) -> ParserSpec:
        return self._spec

    def parse(
        self,
        data: bytes,
        *,
        name: str,
        media_type: str = "application/octet-stream",
        uri: str | None = None,
    ) -> NormalizedDocument:
        try:
            from docling.datamodel.base_models import DocumentStream
            from docling.document_converter import DocumentConverter
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("docling/document parser requires Docling; install ragbot[parser-docling]") from exc
        converter = DocumentConverter()
        result = converter.convert(DocumentStream(name=name, stream=io.BytesIO(data)))
        document = result.document
        blocks: list[DocumentBlock] = []
        current_section: str | None = None
        for item, level in document.iterate_items():
            label_obj = getattr(item, "label", "text")
            label = str(getattr(label_obj, "value", label_obj)).lower()
            text = ""
            if label == "table" and hasattr(item, "export_to_markdown"):
                text = str(item.export_to_markdown(doc=document) or "").strip()
            else:
                text = str(getattr(item, "text", "") or getattr(item, "orig", "") or "").strip()
            if not text:
                continue
            if label in {"section_header", "title"}:
                current_section = text
            provenance = list(getattr(item, "prov", None) or [])
            page = None
            bbox = None
            if provenance:
                prov = provenance[0]
                raw_page = getattr(prov, "page_no", None)
                if raw_page is not None:
                    page = int(raw_page)
                    if page < 1:
                        page += 1
                raw_bbox = getattr(prov, "bbox", None)
                if raw_bbox is not None and all(hasattr(raw_bbox, key) for key in ("l", "t", "r", "b")):
                    bbox = (
                        float(raw_bbox.l),
                        float(raw_bbox.t),
                        float(raw_bbox.r),
                        float(raw_bbox.b),
                    )
            blocks.append(
                DocumentBlock(
                    block_index=len(blocks),
                    text=text,
                    kind=label or "text",
                    page=page,
                    section=current_section,
                    bbox=bbox,
                    metadata={"docling_level": int(level)},
                )
            )
        if not blocks:
            text = str(document.export_to_markdown() or "").strip()
            if text:
                blocks.append(DocumentBlock(block_index=0, text=text, kind="document"))
        return NormalizedDocument(name=name, media_type=media_type, uri=uri, blocks=blocks)


class UnstructuredParser:
    def __init__(self, spec: ParserSpec) -> None:
        self._spec = spec

    @property
    def spec(self) -> ParserSpec:
        return self._spec

    def parse(
        self,
        data: bytes,
        *,
        name: str,
        media_type: str = "application/octet-stream",
        uri: str | None = None,
    ) -> NormalizedDocument:
        try:
            from unstructured.partition.auto import partition
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "unstructured/elements parser requires Unstructured; install ragbot[parser-unstructured]"
            ) from exc
        kwargs: dict[str, Any] = {
            "file": io.BytesIO(data),
            "metadata_filename": name,
        }
        if media_type and media_type != "application/octet-stream":
            kwargs["content_type"] = media_type
        elements = partition(**kwargs)
        blocks: list[DocumentBlock] = []
        current_section: str | None = None
        for element in elements:
            text = str(element).strip()
            if not text:
                continue
            category = str(getattr(element, "category", element.__class__.__name__) or "text").lower()
            if category == "title":
                current_section = text
            metadata = getattr(element, "metadata", None)
            page = getattr(metadata, "page_number", None) if metadata is not None else None
            if page is not None:
                page = int(page)
                if page < 1:
                    page += 1
            blocks.append(
                DocumentBlock(
                    block_index=len(blocks),
                    text=text,
                    kind=category,
                    page=page,
                    section=current_section,
                )
            )
        return NormalizedDocument(name=name, media_type=media_type, uri=uri, blocks=blocks)
