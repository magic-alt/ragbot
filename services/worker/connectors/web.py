from __future__ import annotations

import logging
from typing import Optional

import requests

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 30
_MAX_CONTENT_LENGTH = 5 * 1024 * 1024  # 5 MB


def fetch_web(url: str, timeout: int = _DEFAULT_TIMEOUT, max_length: int = _MAX_CONTENT_LENGTH) -> str:
    """Fetch a web page and return its text content.

    Uses BeautifulSoup for HTML parsing when available, otherwise falls
    back to raw text extraction.
    """
    response = requests.get(url, timeout=timeout, headers={"User-Agent": "ragbot-crawler/0.1"})
    response.raise_for_status()
    content_type = response.headers.get("Content-Type", "")
    raw = response.text[:max_length]

    if "html" in content_type:
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
    text = soup.get_text(separator="\n", strip=True)
    return text
