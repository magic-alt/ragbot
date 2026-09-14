from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Optional

from .store import StoredUpload
from .uri import upload_object_id


class S3UploadStore:
    """Authoritative S3/MinIO UploadStore with content-addressed physical blobs.

    Logical objects are small JSON pointers below ``objects/``. The PDF bytes
    live once below ``blobs/<sha256>.pdf``. API and worker replicas therefore do
    not need a shared PVC; workers materialize a verified local cache copy only
    for parsers that require filesystem paths.
    """

    def __init__(
        self,
        *,
        bucket: str,
        prefix: str = "ragbot/uploads",
        endpoint_url: Optional[str] = None,
        region_name: Optional[str] = None,
        materialize_dir: str | Path | None = None,
        multipart_threshold: int = 8 * 1024 * 1024,
        multipart_chunksize: int = 8 * 1024 * 1024,
        max_concurrency: int = 4,
        max_object_bytes: int = 100 * 1024 * 1024,
        client: Any = None,
    ) -> None:
        if not str(bucket).strip():
            raise ValueError("S3 upload store requires a bucket")
        self.bucket = str(bucket).strip()
        self.prefix = str(prefix or "").strip("/")
        self.max_object_bytes = max(1, int(max_object_bytes))
        root = materialize_dir or Path(tempfile.gettempdir()) / "ragbot-upload-cache"
        self.materialize_root = Path(root).expanduser().resolve()
        self.materialize_root.mkdir(parents=True, exist_ok=True)
        self.tmp_root = self.materialize_root / "incoming"
        self.tmp_root.mkdir(parents=True, exist_ok=True)

        if client is None:
            try:
                import boto3
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("ragbot[s3] is required for the S3 UploadStore") from exc
            kwargs: dict[str, Any] = {}
            if endpoint_url:
                kwargs["endpoint_url"] = endpoint_url
            if region_name:
                kwargs["region_name"] = region_name
            client = boto3.client("s3", **kwargs)
        self._client = client

        try:
            from boto3.s3.transfer import TransferConfig
        except ImportError:  # fake client tests may run without boto3
            self._transfer_config = None
        else:
            self._transfer_config = TransferConfig(
                multipart_threshold=max(5 * 1024 * 1024, int(multipart_threshold)),
                multipart_chunksize=max(5 * 1024 * 1024, int(multipart_chunksize)),
                max_concurrency=max(1, int(max_concurrency)),
                use_threads=True,
            )

    @property
    def backend_id(self) -> str:
        return "s3"

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
        size = int(size_bytes)
        if size <= 0 or size > self.max_object_bytes:
            raise ValueError(
                f"Upload object size must be within 1..{self.max_object_bytes} bytes"
            )
        digest = str(sha256).strip().lower()
        actual_digest, actual_size = _hash_file(temporary, self.max_object_bytes)
        if actual_size != size or actual_digest != digest:
            raise ValueError("Upload temporary file checksum/size does not match commit metadata")

        blob_key = self._blob_key(digest)
        if not self._exists(blob_key):
            kwargs: dict[str, Any] = {
                "ExtraArgs": {
                    "ContentType": "application/pdf",
                    "Metadata": {"sha256": digest},
                }
            }
            if self._transfer_config is not None:
                kwargs["Config"] = self._transfer_config
            self._client.upload_file(str(temporary), self.bucket, blob_key, **kwargs)

        pointer_key = self._pointer_key(object_id)
        pointer = json.dumps(
            {
                "version": 1,
                "object_id": object_id,
                "sha256": digest,
                "size_bytes": size,
                "blob_key": blob_key,
                "media_type": "application/pdf",
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        self._client.put_object(
            Bucket=self.bucket,
            Key=pointer_key,
            Body=pointer,
            ContentType="application/json",
        )
        temporary.unlink(missing_ok=True)
        return StoredUpload(
            object_id=object_id,
            sha256=digest,
            size_bytes=size,
            storage_backend="s3",
            storage_key=pointer_key,
        )

    def materialize_path(self, uri: str) -> Path:
        object_id = upload_object_id(uri)
        pointer = self._read_pointer(object_id)
        expected_size = int(pointer["size_bytes"])
        if expected_size <= 0 or expected_size > self.max_object_bytes:
            raise ValueError("Upload object pointer exceeds configured materialization limit")
        expected_sha = str(pointer["sha256"])
        blob_key = str(pointer["blob_key"])
        target = self.materialize_root / f"{object_id}.pdf"

        if target.is_file():
            digest, size = _hash_file(target, self.max_object_bytes)
            if digest == expected_sha and size == expected_size:
                return target
            target.unlink(missing_ok=True)

        part = self.materialize_root / f"{object_id}.download.part"
        part.unlink(missing_ok=True)
        response = self._client.get_object(Bucket=self.bucket, Key=blob_key)
        body = response["Body"]
        hasher = hashlib.sha256()
        written = 0
        try:
            with part.open("wb") as handle:
                while True:
                    data = body.read(1024 * 1024)
                    if not data:
                        break
                    written += len(data)
                    if written > self.max_object_bytes or written > expected_size:
                        raise ValueError("Downloaded upload object exceeds expected size")
                    hasher.update(data)
                    handle.write(data)
        except Exception:
            part.unlink(missing_ok=True)
            raise
        finally:
            close = getattr(body, "close", None)
            if callable(close):
                close()

        if written != expected_size or hasher.hexdigest() != expected_sha:
            part.unlink(missing_ok=True)
            raise ValueError("Downloaded upload object checksum/size mismatch")
        os.replace(part, target)
        return target

    def local_path(self, uri: str) -> Path:
        """Compatibility facade; object storage remains authoritative."""
        return self.materialize_path(uri)

    def delete_object(self, object_id: str, *, sha256: str | None = None) -> bool:
        pointer_key = self._pointer_key(object_id)
        existed = self._exists(pointer_key)
        if existed:
            self._client.delete_object(Bucket=self.bucket, Key=pointer_key)
        (self.materialize_root / f"{object_id}.pdf").unlink(missing_ok=True)
        # Intentionally retain the content-addressed blob. Another logical
        # UploadedObject may reference the same SHA. Blob GC must be driven by
        # PostgreSQL metadata across all logical objects, never by this adapter.
        return existed

    def _read_pointer(self, object_id: str) -> dict[str, Any]:
        response = self._client.get_object(
            Bucket=self.bucket,
            Key=self._pointer_key(object_id),
        )
        body = response["Body"]
        try:
            raw = body.read(64 * 1024 + 1)
        finally:
            close = getattr(body, "close", None)
            if callable(close):
                close()
        if len(raw) > 64 * 1024:
            raise ValueError("Upload object pointer is unexpectedly large")
        data = json.loads(raw.decode("utf-8"))
        if str(data.get("object_id") or "") != object_id:
            raise ValueError("Upload pointer object identity mismatch")
        for key in ("sha256", "size_bytes", "blob_key"):
            if key not in data:
                raise ValueError(f"Upload pointer missing {key}")
        return data

    def _exists(self, key: str) -> bool:
        try:
            self._client.head_object(Bucket=self.bucket, Key=key)
            return True
        except Exception as exc:
            response = getattr(exc, "response", {}) or {}
            code = str((response.get("Error") or {}).get("Code") or "")
            status = (response.get("ResponseMetadata") or {}).get("HTTPStatusCode")
            if code in {"404", "NoSuchKey", "NotFound"} or status == 404:
                return False
            raise

    def _key(self, value: str) -> str:
        return f"{self.prefix}/{value}" if self.prefix else value

    def _blob_key(self, sha256: str) -> str:
        return self._key(f"blobs/{sha256}.pdf")

    def _pointer_key(self, object_id: str) -> str:
        return self._key(f"objects/{object_id}.json")


def _hash_file(path: Path, limit: int) -> tuple[str, int]:
    hasher = hashlib.sha256()
    size = 0
    with Path(path).open("rb") as handle:
        while True:
            data = handle.read(1024 * 1024)
            if not data:
                break
            size += len(data)
            if size > limit:
                raise ValueError(f"Upload object exceeds configured limit: {limit}")
            hasher.update(data)
    return hasher.hexdigest(), size
