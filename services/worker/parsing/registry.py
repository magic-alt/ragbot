from __future__ import annotations

import json
from functools import lru_cache
from pathlib import PurePosixPath
from typing import Any, Mapping

from .adapters import DoclingParser, PyMuPDFParser, RagbotParser, UnstructuredParser
from .models import NormalizedDocument
from .protocol import DocumentParser, ParserSpec

_VALID_STRATEGIES = {
    "ragbot": {"text", "html", "pypdf2"},
    "pymupdf": {"blocks"},
    "docling": {"document"},
    "unstructured": {"elements"},
}


def resolve_parser_spec(
    config: Mapping[str, Any] | None,
    *,
    name: str,
    media_type: str = "application/octet-stream",
) -> ParserSpec:
    raw = dict(config or {})
    default_provider, default_strategy = _default_parser(name=name, media_type=media_type)
    provider = str(raw.get("provider") or default_provider).strip().lower()
    if provider not in _VALID_STRATEGIES:
        raise ValueError(
            f"Unsupported parser provider {provider!r}; expected one of {sorted(_VALID_STRATEGIES)}"
        )
    strategy = str(raw.get("strategy") or (default_strategy if provider == default_provider else _default_strategy(provider))).strip().lower()
    if strategy not in _VALID_STRATEGIES[provider]:
        raise ValueError(
            f"Unsupported parser strategy {provider}/{strategy}; expected one of {sorted(_VALID_STRATEGIES[provider])}"
        )
    options = raw.get("options") or {}
    if not isinstance(options, Mapping):
        raise ValueError("parser options must be an object")
    options_json = json.dumps(dict(options), sort_keys=True, separators=(",", ":"), default=str)
    return ParserSpec(
        provider=provider,
        strategy=strategy,
        version=int(raw.get("version", 1)),
        options_json=options_json,
    )


def parser_metadata(
    config: Mapping[str, Any] | None,
    *,
    name: str,
    media_type: str = "application/octet-stream",
) -> dict[str, object]:
    return resolve_parser_spec(config, name=name, media_type=media_type).metadata()


def parse_document(
    data: bytes,
    config: Mapping[str, Any] | None,
    *,
    name: str,
    media_type: str = "application/octet-stream",
    uri: str | None = None,
) -> tuple[NormalizedDocument, dict[str, object]]:
    spec = resolve_parser_spec(config, name=name, media_type=media_type)
    parser = _build_parser(spec)
    document = parser.parse(data, name=name, media_type=media_type, uri=uri)
    return document, spec.metadata()


@lru_cache(maxsize=64)
def _build_parser(spec: ParserSpec) -> DocumentParser:
    if spec.provider == "ragbot":
        return RagbotParser(spec)
    if spec.provider == "pymupdf":
        return PyMuPDFParser(spec)
    if spec.provider == "docling":
        return DoclingParser(spec)
    if spec.provider == "unstructured":
        return UnstructuredParser(spec)
    raise ValueError(f"Unsupported parser provider: {spec.provider}")


def _default_parser(*, name: str, media_type: str) -> tuple[str, str]:
    suffix = PurePosixPath(name).suffix.lower()
    normalized_media = (media_type or "").split(";", 1)[0].strip().lower()
    if suffix == ".pdf" or normalized_media == "application/pdf":
        return "ragbot", "pypdf2"
    if suffix in {".html", ".htm"} or normalized_media in {"text/html", "application/xhtml+xml"}:
        return "ragbot", "html"
    return "ragbot", "text"


def _default_strategy(provider: str) -> str:
    return {
        "ragbot": "text",
        "pymupdf": "blocks",
        "docling": "document",
        "unstructured": "elements",
    }[provider]
