"""S3/MinIO object-store ingestion.

Credentials are deliberately not stored in Source.config. boto3 uses its normal
credential chain, or an optional environment prefix can map deployment secrets
to AWS-style client arguments. Custom S3-compatible endpoints are constrained by
an explicit production allowlist so object-store support does not reopen SSRF.
"""
from __future__ import annotations

import hashlib
import io
import logging
import os
import uuid
from pathlib import PurePosixPath
from typing import Iterable, Optional

from services.api.app.storage.models import Chunk
from services.worker.connectors.security import csv_values, validate_remote_url
from services.worker.dedup.hashing import content_hash
from services.worker.jobs.ingest_text import _extract_section, _split_text

logger = logging.getLogger(__name__)
DEFAULT_EXTENSIONS = {".txt", ".md", ".markdown", ".rst", ".py", ".js", ".ts", ".java", ".c", ".cc", ".cpp", ".h", ".hpp", ".go", ".rs", ".yaml", ".yml", ".json", ".toml", ".ini", ".pdf"}


def ingest_s3(
    *,
    bucket: str,
    prefix: str,
    doc_id: str,
    tenant_id: str,
    endpoint_url: Optional[str] = None,
    region_name: Optional[str] = None,
    credential_env_prefix: Optional[str] = None,
    extensions: Optional[list[str]] = None,
    max_object_bytes: int = 20 * 1024 * 1024,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    version: str = "1.0",
    tags: Optional[list] = None,
    acl_hash: Optional[str] = None,
) -> Iterable[Chunk]:
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError("S3/MinIO ingestion requires boto3; install ragbot[s3] or boto3") from exc

    if not bucket.strip():
        raise ValueError("S3 bucket must not be empty")
    if max_object_bytes <= 0:
        raise ValueError("max_object_bytes must be > 0")
    ext_set = _normalize_extensions(extensions)
    client_kwargs = {}
    if endpoint_url:
        client_kwargs["endpoint_url"] = _validate_custom_endpoint(endpoint_url)
    if region_name:
        client_kwargs["region_name"] = region_name
    if credential_env_prefix:
        env_prefix = credential_env_prefix.strip().upper()
        access = os.getenv(f"{env_prefix}_ACCESS_KEY_ID")
        secret = os.getenv(f"{env_prefix}_SECRET_ACCESS_KEY")
        token = os.getenv(f"{env_prefix}_SESSION_TOKEN")
        if access:
            client_kwargs["aws_access_key_id"] = access
        if secret:
            client_kwargs["aws_secret_access_key"] = secret
        if token:
            client_kwargs["aws_session_token"] = token
    client = boto3.client("s3", **client_kwargs)

    paginator = client.get_paginator("list_objects_v2")
    total_objects = 0
    total_chunks = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix or ""):
        for item in page.get("Contents", []):
            key = str(item.get("Key") or "")
            if not key or key.endswith("/"):
                continue
            suffix = PurePosixPath(key).suffix.lower()
            if suffix not in ext_set:
                continue
            size = int(item.get("Size") or 0)
            if size > max_object_bytes:
                logger.warning("Skipping oversized S3 object: s3://%s/%s (%d bytes)", bucket, key, size)
                continue
            body = client.get_object(Bucket=bucket, Key=key)["Body"].read(max_object_bytes + 1)
            if len(body) > max_object_bytes:
                logger.warning("Skipping S3 object exceeding hard read limit: s3://%s/%s", bucket, key)
                continue
            text = _extract_object_text(body, suffix)
            if not text.strip():
                continue

            object_doc_id = f"{doc_id}:{hashlib.sha256(key.encode('utf-8')).hexdigest()[:16]}"
            uri = f"s3://{bucket}/{key}"
            sections = _split_text(text, chunk_size, chunk_overlap)
            for index, section in enumerate(sections):
                yield Chunk(
                    chunk_id=uuid.uuid4().hex,
                    doc_id=object_doc_id,
                    tenant_id=tenant_id,
                    chunk_index=index,
                    text=section,
                    path=uri,
                    section=_extract_section(section) if suffix in {".md", ".markdown"} else None,
                    checksum=content_hash(section),
                    metadata={
                        "source_type": "s3",
                        "bucket": bucket,
                        "object_key": key,
                        "filename": PurePosixPath(key).name,
                        "etag": str(item.get("ETag") or "").strip('"'),
                        "version": version,
                        "tags": tags or [],
                        "acl_hash": acl_hash or "public",
                    },
                )
                total_chunks += 1
            total_objects += 1

    logger.info("S3 ingestion complete: bucket=%s prefix=%s objects=%d chunks=%d", bucket, prefix, total_objects, total_chunks)


def _validate_custom_endpoint(endpoint_url: str) -> str:
    allowed_hosts = csv_values("RAGBOT_S3_ALLOWED_HOSTS")
    environment = os.getenv("RAGBOT_ENV", "development").strip().lower()
    if environment in {"production", "prod"} and not allowed_hosts:
        raise ValueError(
            "Production custom S3/MinIO endpoint_url requires RAGBOT_S3_ALLOWED_HOSTS"
        )
    # An explicitly allowlisted S3-compatible endpoint may legitimately be a
    # private MinIO service. Without that explicit allowlist, reuse the default
    # remote-source private-network policy.
    return validate_remote_url(
        endpoint_url,
        allowed_hosts=allowed_hosts or None,
        allow_private=True if allowed_hosts else None,
    )


def _extract_object_text(body: bytes, suffix: str) -> str:
    if suffix == ".pdf":
        try:
            from PyPDF2 import PdfReader
        except ImportError as exc:
            raise RuntimeError("PDF objects require PyPDF2; install ragbot[worker]") from exc
        reader = PdfReader(io.BytesIO(body))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
    return body.decode("utf-8", errors="replace")


def _normalize_extensions(values: Optional[list[str]]) -> set[str]:
    if not values:
        return set(DEFAULT_EXTENSIONS)
    return {value.lower() if value.startswith(".") else f".{value.lower()}" for value in values}
