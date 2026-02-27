from __future__ import annotations

import logging
import os
import tempfile
from typing import List

logger = logging.getLogger(__name__)


def fetch_pdf(path: str) -> str:
    """Download or read a PDF and return the extracted text."""
    if path.startswith(("http://", "https://")):
        path = _download_to_temp(path, suffix=".pdf")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"PDF not found: {path}")
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        logger.warning("PyPDF2 not installed; returning path as placeholder")
        return path
    reader = PdfReader(path)
    pages: List[str] = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text.strip())
    return "\n\n".join(pages)


def _download_to_temp(url: str, suffix: str = "") -> str:
    import requests

    response = requests.get(url, timeout=60)
    response.raise_for_status()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(response.content)
    tmp.close()
    return tmp.name
