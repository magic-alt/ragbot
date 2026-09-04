from __future__ import annotations

from urllib.parse import unquote, urlsplit

UPLOAD_SCHEME = "ragbot-upload"
UPLOAD_PREFIX = f"{UPLOAD_SCHEME}:///"


def upload_uri(object_id: str) -> str:
    value = str(object_id).strip()
    if not value or any(ch not in "0123456789abcdef" for ch in value.lower()) or len(value) != 32:
        raise ValueError("Upload object_id must be a 32-character hexadecimal identifier")
    return f"{UPLOAD_PREFIX}{value.lower()}"


def is_upload_uri(value: str) -> bool:
    return urlsplit(str(value).strip()).scheme.lower() == UPLOAD_SCHEME


def upload_object_id(value: str) -> str:
    parsed = urlsplit(str(value).strip())
    if parsed.scheme.lower() != UPLOAD_SCHEME:
        raise ValueError(f"Unsupported upload URI scheme: {parsed.scheme or '<missing>'}")
    if parsed.netloc:
        raise ValueError("ragbot-upload URI must not include an authority")
    if parsed.query or parsed.fragment:
        raise ValueError("ragbot-upload URI must not include query or fragment components")
    object_id = unquote(parsed.path).strip("/")
    canonical = upload_uri(object_id)
    return canonical.removeprefix(UPLOAD_PREFIX)
