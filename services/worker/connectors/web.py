from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urljoin

import requests

from .security import csv_values, validate_remote_url

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 30
_MAX_CONTENT_LENGTH = 5 * 1024 * 1024  # 5 MB
_REDIRECT_STATUSES = {301, 302, 303, 307, 308}
_ALLOWED_CONTENT_TYPES = ("text/", "application/json", "application/xml", "application/xhtml+xml")


@dataclass(frozen=True)
class WebResource:
    body: bytes
    content_type: str
    encoding: str
    url: str


def fetch_web_resource(
    url: str,
    timeout: int = _DEFAULT_TIMEOUT,
    max_length: int = _MAX_CONTENT_LENGTH,
) -> WebResource:
    """Fetch bounded bytes while revalidating every redirect destination."""
    allowed_hosts = csv_values("RAGBOT_WEB_ALLOWED_HOSTS")
    max_redirects = int(os.getenv("RAGBOT_WEB_MAX_REDIRECTS", "5"))
    current_url = url

    for redirect_count in range(max_redirects + 1):
        validate_remote_url(current_url, allowed_hosts=allowed_hosts)
        response = requests.get(
            current_url,
            timeout=timeout,
            headers={"User-Agent": "ragbot-crawler/1.0"},
            allow_redirects=False,
            stream=True,
        )
        try:
            if response.status_code in _REDIRECT_STATUSES:
                location = response.headers.get("Location")
                if not location:
                    raise ValueError("Web source redirect did not include Location")
                if redirect_count >= max_redirects:
                    raise ValueError("Web source exceeded redirect limit")
                current_url = urljoin(current_url, location)
                continue

            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
            if content_type and not any(content_type.startswith(prefix) for prefix in _ALLOWED_CONTENT_TYPES):
                raise ValueError(f"Unsupported web source content type: {content_type}")

            content_length = response.headers.get("Content-Length")
            if content_length:
                try:
                    if int(content_length) > max_length:
                        raise ValueError(f"Web source exceeds {max_length} byte limit")
                except ValueError as exc:
                    if "exceeds" in str(exc):
                        raise
                    logger.debug("Ignoring invalid Content-Length: %s", content_length)

            raw_bytes = bytearray()
            for chunk in response.iter_content(chunk_size=65536):
                if not chunk:
                    continue
                if len(raw_bytes) + len(chunk) > max_length:
                    raise ValueError(f"Web source exceeds {max_length} byte limit")
                raw_bytes.extend(chunk)

            return WebResource(
                body=bytes(raw_bytes),
                content_type=content_type or "text/plain",
                encoding=response.encoding or "utf-8",
                url=current_url,
            )
        finally:
            close = getattr(response, "close", None)
            if callable(close):
                close()

    raise ValueError("Web source exceeded redirect limit")


def fetch_web(url: str, timeout: int = _DEFAULT_TIMEOUT, max_length: int = _MAX_CONTENT_LENGTH) -> str:
    """Backward-compatible text helper; production ingestion uses raw resources."""
    resource = fetch_web_resource(url, timeout=timeout, max_length=max_length)
    raw = resource.body.decode(resource.encoding, errors="replace")
    if "html" in resource.content_type:
        return _extract_html_text(raw) or raw
    return raw


def _extract_html_text(html: str) -> Optional[str]:
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        logger.warning("beautifulsoup4 not installed; returning raw HTML text")
        return None
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)
