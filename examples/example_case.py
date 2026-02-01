from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Tuple

from PyPDF2 import PdfReader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from services.api.app.agent.graph import build_default_services
from services.api.app.auth.acl import build_policy
from services.api.app.main import chat
from services.api.app.storage.models import Chunk, Document, TableData
from services.worker.jobs.embed_and_upsert import embed_and_upsert

DEFAULT_PDF = r"E:\work\Project\ragbot\examples\OpenVLA_AnOpen-Source Vision-Language-Action_Model.pdf"


def main() -> None:
    services = build_default_services()
    repo = services.repo
    qdrant = services.qdrant

    policy = build_policy("p1", "tenant-a", {"allow_users": ["u1"]})
    repo.add_policy(policy)

    pdf_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(DEFAULT_PDF)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc_updated_at = datetime.fromtimestamp(pdf_path.stat().st_mtime).date().isoformat()
    ingested_at = datetime.now().date().isoformat()
    doc = Document(
        doc_id="pdf-1",
        tenant_id="tenant-a",
        source_type="pdf",
        title=pdf_path.stem,
        uri=f"file://{pdf_path}",
        version="v1",
        doc_updated_at=doc_updated_at,
        ingested_at=ingested_at,
        tags=["pdf"],
        acl_policy_id=policy.acl_policy_id,
    )
    repo.add_document(doc)

    chunks = _pdf_to_chunks(pdf_path, doc, policy.policy_hash)
    embed_and_upsert(repo, qdrant, chunks)

    table = TableData(
        name="sales",
        columns=[{"name": "region", "type": "text"}, {"name": "amount", "type": "int"}],
        rows=[
            {"region": "cn", "amount": 10},
            {"region": "us", "amount": 20},
        ],
    )
    repo.register_table(table)

    print("Doc RAG:")
    print(chat("这篇论文提出了什么模型？", "tenant-a", "u1", services))

    print("\nSQL:")
    print(chat("select region from sales where region = 'cn'", "tenant-a", "u1", services))

    print("\nCode:")
    print(chat("class SqlEngine", "tenant-a", "u1", services))


def _pdf_to_chunks(pdf_path: Path, doc: Document, acl_hash: str) -> List[Chunk]:
    pages = _load_pdf_text(pdf_path)
    chunks: List[Chunk] = []
    chunk_index = 0
    for page_num, text in pages:
        for part in _split_text(text):
            chunk = Chunk(
                chunk_id=f"{doc.doc_id}-p{page_num}-c{chunk_index}",
                doc_id=doc.doc_id,
                tenant_id=doc.tenant_id,
                chunk_index=chunk_index,
                text=part,
                page=page_num,
                metadata={
                    "source_type": doc.source_type,
                    "ingested_at": doc.ingested_at,
                    "doc_updated_at": doc.doc_updated_at,
                    "version": doc.version,
                    "acl_hash": acl_hash,
                    "tags": doc.tags,
                },
            )
            chunks.append(chunk)
            chunk_index += 1
    return chunks


def _load_pdf_text(pdf_path: Path) -> List[Tuple[int, str]]:
    reader = PdfReader(str(pdf_path))
    pages: List[Tuple[int, str]] = []
    for idx, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        cleaned = " ".join(text.split())
        if cleaned:
            pages.append((idx, cleaned))
    return pages


def _split_text(text: str, chunk_size: int = 500, overlap: int = 50) -> Iterable[str]:
    if len(text) <= chunk_size:
        yield text
        return
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        yield text[start:end]
        if end == len(text):
            break
        start = max(0, end - overlap)


if __name__ == "__main__":
    main()
