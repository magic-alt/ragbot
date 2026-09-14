from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from .uri import upload_object_id


@dataclass(frozen=True)
class StoredUpload:
    object_id: str
    sha256: str
    size_bytes: int
    storage_backend: str
    storage_key: str


@runtime_checkable
class UploadStore(Protocol):
    """Ragbot-owned port for server-managed uploaded objects.

    Object storage may be authoritative. `materialize_path()` returns a bounded,
    verified local parser copy when needed; callers must not infer that the
    returned path is the canonical durable location.
    """

    def temporary_path(self, object_id: str) -> Path: ...

    def commit_pdf(
        self,
        temporary: Path,
        *,
        object_id: str,
        sha256: str,
        size_bytes: int,
    ) -> StoredUpload: ...

    def materialize_path(self, uri: str) -> Path: ...

    def local_path(self, uri: str) -> Path: ...

    def delete_object(self, object_id: str, *, sha256: str | None = None) -> bool: ...


class FilesystemUploadStore:
    """Development/single-node adapter with content-addressed blob deduplication."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.blob_root = self.root / "blobs"
        self.object_root = self.root / "objects"
        self.tmp_root = self.root / "tmp"
        for directory in (self.blob_root, self.object_root, self.tmp_root):
            directory.mkdir(parents=True, exist_ok=True)

    @property
    def backend_id(self) -> str:
        return "filesystem"

    def temporary_path(self, object_id: str) -> Path:
        return self.tmp_root / f"{object_id}.part"

    def commit_pdf(
        self,
        temporary: Path,
        *,
        object_id: str,
        sha256: str,
        size_bytes: int,
    ) -> StoredUpload:
        blob = self.blob_root / f"{sha256}.pdf"
        object_path = self.object_root / f"{object_id}.pdf"
        if not blob.exists():
            os.replace(temporary, blob)
        else:
            temporary.unlink(missing_ok=True)
        object_path.unlink(missing_ok=True)
        try:
            os.link(blob, object_path)
        except OSError:
            shutil.copy2(blob, object_path)
        return StoredUpload(
            object_id=object_id,
            sha256=sha256,
            size_bytes=size_bytes,
            storage_backend="filesystem",
            storage_key=f"objects/{object_id}.pdf",
        )

    def materialize_path(self, uri: str) -> Path:
        return self.local_path(uri)

    def local_path(self, uri: str) -> Path:
        object_id = upload_object_id(uri)
        path = (self.object_root / f"{object_id}.pdf").resolve()
        try:
            path.relative_to(self.object_root.resolve())
        except ValueError as exc:
            raise ValueError("Upload object escapes configured upload root") from exc
        if not path.is_file():
            raise ValueError(f"Upload object is not available: {object_id}")
        return path

    def delete_object(self, object_id: str, *, sha256: str | None = None) -> bool:
        object_path = self.object_root / f"{object_id}.pdf"
        existed = object_path.exists()
        object_path.unlink(missing_ok=True)
        if sha256:
            blob = self.blob_root / f"{sha256}.pdf"
            if blob.exists():
                try:
                    if blob.stat().st_nlink <= 1:
                        blob.unlink(missing_ok=True)
                except OSError:
                    pass
        return existed


def build_upload_store_from_env() -> UploadStore:
    backend = os.getenv("RAGBOT_UPLOAD_STORE", "filesystem").strip().lower()
    if backend == "filesystem":
        root = os.getenv("RAGBOT_UPLOAD_DIR", "").strip()
        if not root:
            environment = os.getenv("RAGBOT_ENV", "development").strip().lower()
            if environment in {"production", "prod"}:
                raise ValueError("Production filesystem uploads require RAGBOT_UPLOAD_DIR")
            data_root = Path(os.getenv("RAGBOT_DATA_DIR", "data")).expanduser().resolve()
            root = str(data_root.parent / "tmp" / "ragbot-uploads")
        return FilesystemUploadStore(root)

    if backend in {"s3", "minio"}:
        from .s3_store import S3UploadStore

        bucket = os.getenv("RAGBOT_UPLOAD_S3_BUCKET", "").strip()
        if not bucket:
            raise ValueError("S3/MinIO uploads require RAGBOT_UPLOAD_S3_BUCKET")
        endpoint = os.getenv("RAGBOT_UPLOAD_S3_ENDPOINT_URL", "").strip() or None
        region = os.getenv("RAGBOT_UPLOAD_S3_REGION", "").strip() or None
        return S3UploadStore(
            bucket=bucket,
            prefix=os.getenv("RAGBOT_UPLOAD_S3_PREFIX", "ragbot/uploads"),
            endpoint_url=endpoint,
            region_name=region,
            materialize_dir=os.getenv("RAGBOT_UPLOAD_MATERIALIZE_DIR", "").strip() or None,
            multipart_threshold=_positive_int_env(
                "RAGBOT_UPLOAD_MULTIPART_THRESHOLD_BYTES", 8 * 1024 * 1024
            ),
            multipart_chunksize=_positive_int_env(
                "RAGBOT_UPLOAD_MULTIPART_CHUNK_BYTES", 8 * 1024 * 1024
            ),
            max_concurrency=_positive_int_env("RAGBOT_UPLOAD_MAX_CONCURRENCY", 4),
            max_object_bytes=_positive_int_env(
                "RAGBOT_UPLOAD_MAX_OBJECT_BYTES", 100 * 1024 * 1024
            ),
        )

    raise ValueError(
        f"Unsupported RAGBOT_UPLOAD_STORE={backend!r}; supported: filesystem, s3, minio"
    )


def _positive_int_env(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be > 0")
    return value
