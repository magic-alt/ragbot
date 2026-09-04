from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
from urllib.parse import quote, unquote, urlsplit

MANAGED_DATA_SCHEME = "ragbot-data"
MANAGED_DATA_PREFIX = f"{MANAGED_DATA_SCHEME}:///"
LEGACY_DOCKER_DATA_ROOT = PurePosixPath("/data")


def is_managed_data_uri(value: str) -> bool:
    return urlsplit(str(value).strip()).scheme.lower() == MANAGED_DATA_SCHEME


def managed_data_uri(relative: str | PurePosixPath) -> str:
    """Return a canonical executor-independent URI below Ragbot's managed data root."""
    raw = str(relative).replace("\\", "/").lstrip("/")
    path = _validate_relative(raw)
    encoded = "" if path == PurePosixPath(".") else quote(path.as_posix(), safe="/-._~")
    return f"{MANAGED_DATA_PREFIX}{encoded}"


def canonical_managed_data_uri(value: str) -> str:
    parsed = urlsplit(str(value).strip())
    if parsed.scheme.lower() != MANAGED_DATA_SCHEME:
        raise ValueError(f"Unsupported managed data URI scheme: {parsed.scheme or '<missing>'}")
    if parsed.netloc:
        raise ValueError("ragbot-data URI must not include an authority")
    if parsed.query or parsed.fragment:
        raise ValueError("ragbot-data URI must not include query or fragment components")
    return managed_data_uri(unquote(parsed.path).lstrip("/"))


def managed_data_root() -> Path:
    """Resolve the executor-local physical root for ragbot-data URIs."""
    configured = os.getenv("RAGBOT_DATA_DIR", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()

    roots = [value.strip() for value in os.getenv("RAGBOT_ALLOWED_LOCAL_SOURCE_ROOTS", "").split(os.pathsep) if value.strip()]
    if len(roots) == 1:
        return Path(roots[0]).expanduser().resolve()

    raise ValueError(
        "ragbot-data URI requires RAGBOT_DATA_DIR (or exactly one RAGBOT_ALLOWED_LOCAL_SOURCE_ROOTS entry)"
    )


def resolve_managed_data_uri(value: str) -> str:
    canonical = canonical_managed_data_uri(value)
    relative = _managed_relative(canonical)
    root = managed_data_root()
    candidate = root if relative == PurePosixPath(".") else (root / Path(*relative.parts)).resolve()
    if not _is_within(candidate, root):
        raise ValueError("ragbot-data URI escapes RAGBOT_DATA_DIR")
    return str(candidate)


def resolve_local_source_reference(value: str) -> str:
    """Resolve portable/legacy Ragbot local references into this executor's filesystem."""
    raw = str(value).strip()
    if is_managed_data_uri(raw):
        return resolve_managed_data_uri(raw)

    legacy_relative = _legacy_data_relative(raw)
    if legacy_relative is not None:
        try:
            root = managed_data_root()
        except ValueError:
            root = None
        if root is not None:
            candidate = root if legacy_relative == PurePosixPath(".") else (root / Path(*legacy_relative.parts)).resolve()
            if not _is_within(candidate, root):
                raise ValueError("Legacy /data source escapes RAGBOT_DATA_DIR")
            return str(candidate)

    return str(Path(raw).expanduser().resolve())


def resolve_allowed_local_root(value: str) -> str:
    """Normalize allowlist roots using the same managed-root alias rules as sources."""
    raw = str(value).strip()
    if is_managed_data_uri(raw):
        return resolve_managed_data_uri(raw)
    if raw == "/data" or raw.startswith("/data/"):
        try:
            return resolve_local_source_reference(raw)
        except ValueError:
            pass
    return str(Path(raw).expanduser().resolve())


def _managed_relative(value: str) -> PurePosixPath:
    parsed = urlsplit(value)
    return _validate_relative(unquote(parsed.path).lstrip("/"))


def _legacy_data_relative(value: str) -> PurePosixPath | None:
    normalized = value.replace("\\", "/")
    if normalized == "/data":
        return PurePosixPath(".")
    if not normalized.startswith("/data/"):
        return None
    return _validate_relative(normalized[len("/data/") :])


def _validate_relative(value: str) -> PurePosixPath:
    normalized = value.replace("\\", "/").strip("/")
    if not normalized:
        return PurePosixPath(".")
    path = PurePosixPath(normalized)
    if path.is_absolute() or any(part == ".." for part in path.parts):
        raise ValueError("Managed data path must remain below the Ragbot data root")
    return path


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
