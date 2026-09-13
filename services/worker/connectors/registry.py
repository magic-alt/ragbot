from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable, Iterable, Mapping, Optional
from urllib.parse import urlsplit, urlunsplit

from services.api.app.storage.models import Chunk, Source
from services.api.app.storage.protocol import Repo
from services.platform import TypedRegistry
from services.worker.chunking import resolve_chunking_spec
from services.worker.parsing import resolve_parser_spec

from .credentials import validate_secret_ref


ConfigBuilder = Callable[[str, Mapping[str, Any]], dict[str, Any]]
Canonicalizer = Callable[[str], str]
ConfigValidator = Callable[[Mapping[str, Any]], None]
LocationMatcher = Callable[[str], int]
LocationExtractor = Callable[[Mapping[str, Any]], Optional[str]]
ConnectorRunner = Callable[[Source, Repo, Iterable[Chunk]], Iterable[Chunk]]

_NOTION_ID = re.compile(r"([0-9a-fA-F]{32})(?:[/?#]|$)")
_INLINE_SECRET_MARKERS = (
    "access_token",
    "refresh_token",
    "api_key",
    "apikey",
    "password",
    "private_key",
    "client_secret",
    "secret_access_key",
)


@dataclass(frozen=True)
class ConnectorCapabilities:
    parsing: bool = False
    incremental: bool = False
    remote: bool = False
    credentials: bool = False
    multi_document: bool = False

    def as_dict(self) -> dict[str, bool]:
        return {
            "parsing": self.parsing,
            "incremental": self.incremental,
            "remote": self.remote,
            "credentials": self.credentials,
            "multi_document": self.multi_document,
        }


@dataclass(frozen=True)
class ConnectorSpec:
    source_type: str
    runner: ConnectorRunner
    build_config: ConfigBuilder
    canonicalize: Canonicalizer
    match_location: LocationMatcher
    source_location: LocationExtractor
    validate_config: ConfigValidator = lambda _config: None
    capabilities: ConnectorCapabilities = field(default_factory=ConnectorCapabilities)
    default_chunk_size: int = 800
    default_chunk_overlap: int = 100
    default_chunk_strategy: Optional[str] = None
    parser_validation_name: Optional[str] = None
    parser_media_type: str = "application/octet-stream"
    version: str = "1"
    optional_dependency: Optional[str] = None

    @property
    def component_id(self) -> str:
        return f"connector:{self.source_type}"

    def public_metadata(self) -> dict[str, object]:
        return {
            "id": self.component_id,
            "source_type": self.source_type,
            "version": self.version,
            "capabilities": self.capabilities.as_dict(),
            "optional_dependency": self.optional_dependency,
        }


class ConnectorRegistry:
    """Registry of normalized Source connectors used by API, CLI and worker."""

    ENTRYPOINT_GROUP = "ragbot.connectors"

    def __init__(self) -> None:
        self._registry: TypedRegistry[ConnectorSpec] = TypedRegistry(
            kind="connector",
            entrypoint_group=self.ENTRYPOINT_GROUP,
        )

    def register(self, spec: ConnectorSpec, *, replace: bool = False) -> ConnectorSpec:
        self._registry.register(spec, replace=replace)
        return spec

    def get(self, source_type: str) -> ConnectorSpec:
        return self._registry.get(f"connector:{str(source_type).strip().lower()}")

    def source_types(self) -> tuple[str, ...]:
        return tuple(spec.source_type for spec in self._registry.values())

    def specs(self) -> tuple[ConnectorSpec, ...]:
        return self._registry.values()

    def infer_source_type(self, location: str) -> str:
        ranked = sorted(
            ((int(spec.match_location(location)), spec.source_type) for spec in self.specs()),
            key=lambda item: (-item[0], item[1]),
        )
        if not ranked or ranked[0][0] <= 0:
            raise ValueError(f"No connector can infer source type for location: {location!r}")
        return ranked[0][1]

    def canonical_location(self, location: str, source_type: Optional[str] = None) -> str:
        resolved = source_type or self.infer_source_type(location)
        return self.get(resolved).canonicalize(location)

    def build_source_config(
        self,
        source_type: str,
        location: str,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        spec = self.get(source_type)
        config = spec.build_config(location, dict(extra or {}))
        self.validate_source_config(source_type, config)
        return config

    def validate_source_config(self, source_type: str, config: Mapping[str, Any]) -> None:
        spec = self.get(source_type)
        raw_chunking = config.get("chunking")
        if raw_chunking is not None and not isinstance(raw_chunking, Mapping):
            raise ValueError("config.chunking must be an object")
        resolve_chunking_spec(
            raw_chunking,
            chunk_size=int(config.get("chunk_size", spec.default_chunk_size)),
            chunk_overlap=int(config.get("chunk_overlap", spec.default_chunk_overlap)),
            default_strategy=spec.default_chunk_strategy,
        )

        raw_parsing = config.get("parsing")
        if raw_parsing is not None:
            if not spec.capabilities.parsing:
                raise ValueError(f"source_type={source_type} does not accept config.parsing")
            if not isinstance(raw_parsing, Mapping):
                raise ValueError("config.parsing must be an object")
            resolve_parser_spec(
                raw_parsing,
                name=spec.parser_validation_name or "document.txt",
                media_type=spec.parser_media_type,
            )
        spec.validate_config(config)

    def source_location(self, source: Source) -> Optional[str]:
        return self.get(source.source_type).source_location(source.config or {})

    def ingest(
        self,
        source: Source,
        repo: Repo,
        previous_chunks: Iterable[Chunk] = (),
    ) -> Iterable[Chunk]:
        spec = self.get(source.source_type)
        self.validate_source_config(source.source_type, source.config or {})
        return spec.runner(source, repo, previous_chunks)

    def public_metadata(self) -> list[dict[str, object]]:
        return [spec.public_metadata() for spec in self.specs()]


@lru_cache(maxsize=1)
def connector_registry() -> ConnectorRegistry:
    registry = ConnectorRegistry()
    for spec in _builtin_specs():
        registry.register(spec)
    registry.specs()
    return registry


def _require_string(config: Mapping[str, Any], key: str, source_type: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"source_type={source_type} requires non-empty config.{key}")
    return value.strip()


def _reject_inline_secrets(config: Mapping[str, Any]) -> None:
    offending = []
    for key, value in config.items():
        normalized = str(key).strip().lower()
        if normalized == "credential_ref":
            continue
        if any(marker in normalized for marker in _INLINE_SECRET_MARKERS) and value not in (None, ""):
            offending.append(str(key))
    if offending:
        raise ValueError(
            "Cloud connector credentials must not be stored in Source.config; "
            f"use credential_ref=env:VARIABLE instead (inline fields: {', '.join(sorted(offending))})"
        )


def _validate_path(config: Mapping[str, Any], source_type: str) -> None:
    _require_string(config, "path", source_type)


def _validate_web(config: Mapping[str, Any]) -> None:
    _require_string(config, "url", "web")


def _validate_s3(config: Mapping[str, Any]) -> None:
    _require_string(config, "bucket", "s3")


def _validate_gdrive(config: Mapping[str, Any]) -> None:
    _reject_inline_secrets(config)
    _require_string(config, "folder_id", "gdrive")
    validate_secret_ref(_require_string(config, "credential_ref", "gdrive"))
    credential_type = str(config.get("credential_type", "access_token")).strip().lower()
    if credential_type not in {"access_token", "google_json"}:
        raise ValueError("gdrive credential_type must be access_token or google_json")


def _validate_notion(config: Mapping[str, Any]) -> None:
    _reject_inline_secrets(config)
    _require_string(config, "page_id", "notion")
    validate_secret_ref(_require_string(config, "credential_ref", "notion"))


def _validate_confluence(config: Mapping[str, Any]) -> None:
    _reject_inline_secrets(config)
    _require_string(config, "base_url", "confluence")
    _require_string(config, "space_key", "confluence")
    validate_secret_ref(_require_string(config, "credential_ref", "confluence"))
    auth_type = str(config.get("auth_type", "basic")).strip().lower()
    if auth_type not in {"basic", "bearer"}:
        raise ValueError("confluence auth_type must be basic or bearer")
    if auth_type == "basic":
        _require_string(config, "email", "confluence")


def _build_path(location: str, extra: Mapping[str, Any]) -> dict[str, Any]:
    config = dict(extra)
    config["path"] = location.strip()
    return config


def _build_web(location: str, extra: Mapping[str, Any]) -> dict[str, Any]:
    config = dict(extra)
    config["url"] = location.strip()
    return config


def _build_s3(location: str, extra: Mapping[str, Any]) -> dict[str, Any]:
    config = dict(extra)
    parsed = urlsplit(location.strip())
    if parsed.scheme.lower() != "s3" or not parsed.netloc:
        raise ValueError("source_type=s3 requires location like s3://bucket/prefix")
    config["bucket"] = parsed.netloc
    config["prefix"] = parsed.path.lstrip("/")
    return config


def _build_gdrive(location: str, extra: Mapping[str, Any]) -> dict[str, Any]:
    config = dict(extra)
    value = location.strip()
    parsed = urlsplit(value)
    if parsed.scheme.lower() in {"gdrive", "googledrive"}:
        folder_id = (parsed.netloc or parsed.path).strip("/")
    elif (parsed.hostname or "").lower() in {"drive.google.com", "www.drive.google.com"}:
        folder_id = _drive_folder_id(parsed.path)
    else:
        folder_id = value
    if not folder_id:
        raise ValueError("gdrive location requires a Drive folder ID")
    config["folder_id"] = folder_id
    return config


def _build_notion(location: str, extra: Mapping[str, Any]) -> dict[str, Any]:
    config = dict(extra)
    value = location.strip()
    parsed = urlsplit(value)
    page_id = _extract_notion_page_id(value)
    if parsed.scheme.lower() == "notion" and not page_id:
        page_id = (parsed.netloc + parsed.path).strip("/")
    if not page_id:
        page_id = value if "/" not in value else ""
    if not page_id:
        raise ValueError("notion location requires a page ID or Notion page URL")
    config["page_id"] = page_id
    return config


def _build_confluence(location: str, extra: Mapping[str, Any]) -> dict[str, Any]:
    config = dict(extra)
    value = location.strip()
    parsed = urlsplit(value)
    if parsed.scheme.lower() == "confluence":
        host = parsed.hostname or parsed.netloc
        space = parsed.path.strip("/")
        if not host or not space:
            raise ValueError("confluence location requires confluence://host/SPACE")
        config.setdefault("base_url", f"https://{host}/wiki")
        config["space_key"] = space
    elif parsed.scheme.lower() in {"http", "https"} and parsed.hostname:
        space = _confluence_space_key(parsed.path)
        if not space:
            raise ValueError("Confluence URL must contain /spaces/SPACE")
        wiki_prefix = "/wiki" if parsed.path.startswith("/wiki/") else ""
        config.setdefault("base_url", f"{parsed.scheme.lower()}://{parsed.netloc}{wiki_prefix}")
        config["space_key"] = space
    else:
        config["space_key"] = value
    return config


def _canonical_local(location: str) -> str:
    value = location.strip()
    normalized = value.rstrip("/\\")
    return normalized or value


def _canonical_http(location: str) -> str:
    value = location.strip()
    parsed = urlsplit(value)
    if parsed.scheme.lower() not in {"http", "https"}:
        return _canonical_local(value)
    netloc = parsed.netloc.lower()
    path = parsed.path.rstrip("/") or "/"
    return urlunsplit((parsed.scheme.lower(), netloc, path, parsed.query, ""))


def _canonical_s3(location: str) -> str:
    parsed = urlsplit(location.strip())
    bucket = (parsed.netloc or "").lower()
    prefix = parsed.path.strip("/")
    return f"s3://{bucket}/{prefix}" if prefix else f"s3://{bucket}"


def _canonical_gdrive(location: str) -> str:
    config = _build_gdrive(location, {})
    return f"gdrive://{config['folder_id']}"


def _canonical_notion(location: str) -> str:
    config = _build_notion(location, {})
    return f"notion://{str(config['page_id']).replace('-', '').lower()}"


def _canonical_confluence(location: str) -> str:
    config = _build_confluence(location, {})
    base_url = str(config.get("base_url") or "")
    host = (urlsplit(base_url).hostname or urlsplit(location).hostname or "").lower()
    space = str(config.get("space_key") or "").upper()
    return f"confluence://{host}/{space}" if host else f"confluence:///{space}"


def _source_path(config: Mapping[str, Any]) -> Optional[str]:
    value = config.get("path")
    return value if isinstance(value, str) and value.strip() else None


def _source_web(config: Mapping[str, Any]) -> Optional[str]:
    value = config.get("url")
    return value if isinstance(value, str) and value.strip() else None


def _source_s3(config: Mapping[str, Any]) -> Optional[str]:
    bucket = config.get("bucket")
    if not isinstance(bucket, str) or not bucket.strip():
        return None
    prefix = str(config.get("prefix") or "").strip("/")
    return f"s3://{bucket}/{prefix}" if prefix else f"s3://{bucket}"


def _source_gdrive(config: Mapping[str, Any]) -> Optional[str]:
    folder_id = config.get("folder_id")
    return f"gdrive://{folder_id}" if folder_id else None


def _source_notion(config: Mapping[str, Any]) -> Optional[str]:
    page_id = config.get("page_id")
    return f"notion://{str(page_id).replace('-', '').lower()}" if page_id else None


def _source_confluence(config: Mapping[str, Any]) -> Optional[str]:
    base_url = str(config.get("base_url") or "")
    host = (urlsplit(base_url).hostname or "").lower()
    space = str(config.get("space_key") or "").upper()
    return f"confluence://{host}/{space}" if host and space else None


def _match_s3(location: str) -> int:
    return 100 if urlsplit(location.strip()).scheme.lower() == "s3" else 0


def _match_gdrive(location: str) -> int:
    parsed = urlsplit(location.strip())
    host = (parsed.hostname or "").lower()
    if parsed.scheme.lower() in {"gdrive", "googledrive"}:
        return 110
    return 105 if host in {"drive.google.com", "www.drive.google.com"} and "/folders/" in parsed.path.lower() else 0


def _match_notion(location: str) -> int:
    parsed = urlsplit(location.strip())
    host = (parsed.hostname or "").lower()
    if parsed.scheme.lower() == "notion":
        return 110
    return 105 if host in {"notion.so", "www.notion.so"} or host.endswith(".notion.site") else 0


def _match_confluence(location: str) -> int:
    parsed = urlsplit(location.strip())
    host = (parsed.hostname or "").lower()
    if parsed.scheme.lower() == "confluence":
        return 110
    return 105 if host.endswith(".atlassian.net") and "/spaces/" in parsed.path.lower() else 0


def _match_pdf(location: str) -> int:
    parsed = urlsplit(location.strip())
    target = parsed.path if parsed.scheme.lower() in {"http", "https"} else location
    return 90 if target.lower().rstrip("/\\").endswith(".pdf") else 0


def _match_repo(location: str) -> int:
    parsed = urlsplit(location.strip())
    host = (parsed.hostname or "").lower()
    target = parsed.path if parsed.scheme.lower() in {"http", "https"} else location
    if target.lower().rstrip("/\\").endswith(".git"):
        return 80
    return 75 if parsed.scheme.lower() in {"http", "https"} and host in {"github.com", "gitlab.com", "bitbucket.org"} else 0


def _match_web(location: str) -> int:
    return 50 if urlsplit(location.strip()).scheme.lower() in {"http", "https"} else 0


def _match_local(_location: str) -> int:
    return 1


def _common(source: Source, repo: Repo) -> dict[str, Any]:
    config = source.config or {}
    return {
        "doc_id": config.get("doc_id") or f"doc-{source.source_id}",
        "tenant_id": source.tenant_id,
        "version": config.get("version", "1.0"),
        "tags": source.tags,
        "acl_hash": _resolve_acl_hash(source, repo),
        "chunking": config.get("chunking"),
    }


def _resolve_acl_hash(source: Source, repo: Repo) -> str:
    if source.acl_policy_id:
        policy_hash = repo.get_policy_hash(source.acl_policy_id)
        if policy_hash:
            return policy_hash
    return "public"


def _run_pdf(source: Source, repo: Repo, _previous: Iterable[Chunk]) -> Iterable[Chunk]:
    from services.worker.jobs.ingest_pdf import ingest_pdf
    config = source.config or {}
    return ingest_pdf(path=config["path"], chunk_size=int(config.get("chunk_size", 800)), chunk_overlap=int(config.get("chunk_overlap", 100)), parsing=config.get("parsing"), **_common(source, repo))


def _run_web(source: Source, repo: Repo, _previous: Iterable[Chunk]) -> Iterable[Chunk]:
    from services.worker.jobs.ingest_web import ingest_web
    config = source.config or {}
    return ingest_web(url=config["url"], chunk_size=int(config.get("chunk_size", 800)), chunk_overlap=int(config.get("chunk_overlap", 100)), parsing=config.get("parsing"), **_common(source, repo))


def _run_repo(source: Source, repo: Repo, _previous: Iterable[Chunk]) -> Iterable[Chunk]:
    from services.worker.jobs.ingest_repo import ingest_repo
    config = source.config or {}
    return ingest_repo(url_or_path=config["path"], ref=config.get("ref"), chunk_size=int(config.get("chunk_size", 600)), chunk_overlap=int(config.get("chunk_overlap", 100)), **_common(source, repo))


def _run_local(source: Source, repo: Repo, _previous: Iterable[Chunk]) -> Iterable[Chunk]:
    from services.worker.jobs.ingest_text import ingest_local_fs
    config = source.config or {}
    return ingest_local_fs(directory=config["path"], extensions=config.get("extensions"), chunk_size=int(config.get("chunk_size", 800)), chunk_overlap=int(config.get("chunk_overlap", 100)), parsing=config.get("parsing"), **_common(source, repo))


def _run_s3(source: Source, repo: Repo, _previous: Iterable[Chunk]) -> Iterable[Chunk]:
    from services.worker.jobs.ingest_s3 import ingest_s3
    config = source.config or {}
    return ingest_s3(bucket=config["bucket"], prefix=config.get("prefix", ""), endpoint_url=config.get("endpoint_url"), region_name=config.get("region_name"), credential_env_prefix=config.get("credential_env_prefix"), extensions=config.get("extensions"), max_object_bytes=int(config.get("max_object_bytes", 20 * 1024 * 1024)), chunk_size=int(config.get("chunk_size", 800)), chunk_overlap=int(config.get("chunk_overlap", 100)), parsing=config.get("parsing"), **_common(source, repo))


def _run_gdrive(source: Source, repo: Repo, previous: Iterable[Chunk]) -> Iterable[Chunk]:
    from services.worker.jobs.ingest_google_drive import ingest_google_drive
    config = source.config or {}
    return ingest_google_drive(folder_id=config["folder_id"], credential_ref=config["credential_ref"], credential_type=config.get("credential_type", "access_token"), recursive=bool(config.get("recursive", True)), max_file_bytes=int(config.get("max_file_bytes", 20 * 1024 * 1024)), chunk_size=int(config.get("chunk_size", 800)), chunk_overlap=int(config.get("chunk_overlap", 100)), previous_chunks=previous, parsing=config.get("parsing"), **_common(source, repo))


def _run_notion(source: Source, repo: Repo, previous: Iterable[Chunk]) -> Iterable[Chunk]:
    from services.worker.jobs.ingest_notion import ingest_notion
    config = source.config or {}
    return ingest_notion(page_id=config["page_id"], credential_ref=config["credential_ref"], recursive=bool(config.get("recursive", True)), notion_version=config.get("notion_version", "2022-06-28"), chunk_size=int(config.get("chunk_size", 800)), chunk_overlap=int(config.get("chunk_overlap", 100)), previous_chunks=previous, **_common(source, repo))


def _run_confluence(source: Source, repo: Repo, previous: Iterable[Chunk]) -> Iterable[Chunk]:
    from services.worker.jobs.ingest_confluence import ingest_confluence
    config = source.config or {}
    return ingest_confluence(base_url=config["base_url"], space_key=config["space_key"], credential_ref=config["credential_ref"], auth_type=config.get("auth_type", "basic"), email=config.get("email"), root_page_id=config.get("root_page_id"), chunk_size=int(config.get("chunk_size", 800)), chunk_overlap=int(config.get("chunk_overlap", 100)), previous_chunks=previous, **_common(source, repo))


def _drive_folder_id(path: str) -> str:
    parts = [part for part in path.split("/") if part]
    try:
        index = parts.index("folders")
    except ValueError:
        return ""
    return parts[index + 1] if index + 1 < len(parts) else ""


def _extract_notion_page_id(value: str) -> str:
    compact = str(value or "").replace("-", "")
    matches = list(_NOTION_ID.finditer(compact))
    return matches[-1].group(1) if matches else ""


def _confluence_space_key(path: str) -> str:
    parts = [part for part in path.split("/") if part]
    lowered = [part.lower() for part in parts]
    try:
        index = lowered.index("spaces")
    except ValueError:
        return ""
    return parts[index + 1] if index + 1 < len(parts) else ""


def _builtin_specs() -> tuple[ConnectorSpec, ...]:
    local_caps = ConnectorCapabilities(parsing=True, multi_document=True)
    remote_multi = ConnectorCapabilities(parsing=True, remote=True, multi_document=True)
    remote_incremental = ConnectorCapabilities(parsing=True, incremental=True, remote=True, credentials=True, multi_document=True)
    return (
        ConnectorSpec(source_type="local_fs", runner=_run_local, build_config=_build_path, canonicalize=_canonical_local, match_location=_match_local, source_location=_source_path, validate_config=lambda config: _validate_path(config, "local_fs"), capabilities=local_caps, parser_validation_name="document.txt"),
        ConnectorSpec(source_type="pdf", runner=_run_pdf, build_config=_build_path, canonicalize=_canonical_http, match_location=_match_pdf, source_location=_source_path, validate_config=lambda config: _validate_path(config, "pdf"), capabilities=ConnectorCapabilities(parsing=True), parser_validation_name="document.pdf", parser_media_type="application/pdf"),
        ConnectorSpec(source_type="web", runner=_run_web, build_config=_build_web, canonicalize=_canonical_http, match_location=_match_web, source_location=_source_web, validate_config=_validate_web, capabilities=ConnectorCapabilities(parsing=True, remote=True), parser_validation_name="index.html"),
        ConnectorSpec(source_type="repo", runner=_run_repo, build_config=_build_path, canonicalize=_canonical_http, match_location=_match_repo, source_location=_source_path, validate_config=lambda config: _validate_path(config, "repo"), capabilities=ConnectorCapabilities(remote=True), default_chunk_size=600, default_chunk_strategy="structural"),
        ConnectorSpec(source_type="s3", runner=_run_s3, build_config=_build_s3, canonicalize=_canonical_s3, match_location=_match_s3, source_location=_source_s3, validate_config=_validate_s3, capabilities=remote_multi, parser_validation_name="document.txt", optional_dependency="ragbot[s3]"),
        ConnectorSpec(source_type="gdrive", runner=_run_gdrive, build_config=_build_gdrive, canonicalize=_canonical_gdrive, match_location=_match_gdrive, source_location=_source_gdrive, validate_config=_validate_gdrive, capabilities=remote_incremental, parser_validation_name="document.txt", optional_dependency="ragbot[saas]"),
        ConnectorSpec(source_type="notion", runner=_run_notion, build_config=_build_notion, canonicalize=_canonical_notion, match_location=_match_notion, source_location=_source_notion, validate_config=_validate_notion, capabilities=ConnectorCapabilities(incremental=True, remote=True, credentials=True, multi_document=True), optional_dependency="ragbot[saas]"),
        ConnectorSpec(source_type="confluence", runner=_run_confluence, build_config=_build_confluence, canonicalize=_canonical_confluence, match_location=_match_confluence, source_location=_source_confluence, validate_config=_validate_confluence, capabilities=ConnectorCapabilities(incremental=True, remote=True, credentials=True, multi_document=True), optional_dependency="ragbot[saas]"),
    )
