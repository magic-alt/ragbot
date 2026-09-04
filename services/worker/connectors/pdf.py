from __future__ import annotations

import logging
import os
import tempfile
from typing import List
from urllib.parse import urljoin

import requests

from .security import csv_values, validate_local_source_path, validate_remote_url

logger = logging.getLogger(__name__)

_REDIRECT_STATUSES = {301, 302, 303, 307, 308}
_DEFAULT_MAX_PDF_BYTES = 25 * 1024 * 1024


def fetch_pdf_pages(path: str) -> List[tuple[int, str]]:
    """Read an allowed local/remote PDF while preserving 1-based page identity."""
    try:
        from PyPDF2 import PdfReader
    except ImportError as exc:
        raise RuntimeError("PyPDF2 is required for PDF ingestion") from exc

    temporary = False
    if path.startswith(("http://", "https://")):
        path = _download_to_temp(path, suffix=".pdf")
        temporary = True
    else:
        path = validate_local_source_path(path)

    if not os.path.isfile(path):
        raise FileNotFoundError(f"PDF not found: {path}")

    try:
        reader = PdfReader(path)
        pages: List[tuple[int, str]] = []
        for page_number, page in enumerate(reader.pages, 1):
            text = page.extract_text()
            if text and text.strip():
                pages.append((page_number, text.strip()))
        return pages
    finally:
        if temporary:
            try:
                os.unlink(path)
            except OSError:
                logger.warning("Unable to remove temporary PDF: %s", path)


def fetch_pdf(path: str) -> str:
    """Backward-compatible flattened PDF text helper."""
    return "\n\n".join(text for _page, text in fetch_pdf_pages(path))


def _download_to_temp(url: str, suffix: str = "") -> str:
    max_bytes = int(os.getenv("RAGBOT_PDF_MAX_BYTES", str(_DEFAULT_MAX_PDF_BYTES)))
    max_redirects = int(os.getenv("RAGBOT_PDF_MAX_REDIRECTS", "5"))
    allowed_hosts = csv_values("RAGBOT_PDF_ALLOWED_HOSTS")
    current_url = url

    for redirect_count in range(max_redirects + 1):
        validate_remote_url(current_url, allowed_hosts=allowed_hosts)
        response = requests.get(current_url, timeout=60, allow_redirects=False, stream=True)
        try:
            if response.status_code in _REDIRECT_STATUSES:
                location = response.headers.get("Location")
                if not location:
                    raise ValueError("PDF redirect did not include Location")
                if redirect_count >= max_redirects:
                    raise ValueError("PDF source exceeded redirect limit")
                current_url = urljoin(current_url, location)
                continue

            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
            if content_type and content_type not in {"application/pdf", "application/octet-stream"}:
                raise ValueError(f"Unsupported PDF content type: {content_type}")
            content_length = response.headers.get("Content-Length")
            if content_length and content_length.isdigit() and int(content_length) > max_bytes:
                raise ValueError(f"PDF source exceeds {max_bytes} byte limit")

            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            written = 0
            try:
                for chunk in response.iter_content(chunk_size=65536):
                    if not chunk:
                        continue
                    written += len(chunk)
                    if written > max_bytes:
                        raise ValueError(f"PDF source exceeds {max_bytes} byte limit")
                    tmp.write(chunk)
                tmp.close()
                return tmp.name
            except Exception:
                tmp.close()
                try:
                    os.unlink(tmp.name)
                except OSError:
                    pass
                raise
        finally:
            close = getattr(response, "close", None)
            if callable(close):
                close()

    raise ValueError("PDF source exceeded redirect limit")
