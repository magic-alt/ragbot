from __future__ import annotations

from typing import Iterable


def ingest_pdf(path: str) -> Iterable[str]:
    yield f"[PDF] {path}"

