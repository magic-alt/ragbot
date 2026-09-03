from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from services.api.app.retrieval.cross_encoder import NoOpReranker
from services.api.app.retrieval.embedder import HashEmbedder
from services.api.app.retrieval.pg_fts import fts_search
from services.api.app.retrieval.qdrant import InMemoryQdrant
from services.api.app.retrieval.service import Retriever
from services.api.app.routes.admin_ui import create_admin_ui_router
from services.api.app.storage.models import Chunk
from services.api.app.storage.repo import InMemoryRepo
from services.worker.jobs.embed_and_upsert import embed_and_upsert


def _chunk(chunk_id: str, text: str, *, page: int = 1) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc-1",
        tenant_id="engineering",
        chunk_index=page - 1,
        text=text,
        page=page,
        metadata={"source_type": "pdf", "acl_hash": "public"},
    )


def test_local_lexical_search_ignores_question_stopwords() -> None:
    repo = InMemoryRepo()
    repo.add_chunk(_chunk("generic", "The model is useful and the system is designed for deployment."))
    repo.add_chunk(
        _chunk(
            "lora",
            "QLoRA combines 4-bit quantization with LoRA adapters for parameter-efficient fine-tuning.",
        )
    )

    hits = fts_search(
        repo,
        "What is the difference between LoRA and QLoRA?",
        {"tenant_id": "engineering"},
        5,
    )

    assert hits
    assert hits[0][0].chunk_id == "lora"


def test_local_lexical_search_uses_cjk_bigrams() -> None:
    repo = InMemoryRepo()
    repo.add_chunk(_chunk("gpu", "大模型部署可以通过量化和混合精度减少显存占用。"))
    repo.add_chunk(_chunk("other", "这是关于数据库索引和事务处理的章节。"))

    hits = fts_search(repo, "如何减少显存占用", {"tenant_id": "engineering"}, 5)

    assert hits
    assert hits[0][0].chunk_id == "gpu"


def test_retriever_attaches_rank_trace_and_hash_warning() -> None:
    repo = InMemoryRepo()
    qdrant = InMemoryQdrant(dim=64)
    embedder = HashEmbedder(dim=64)
    chunks = [
        _chunk("lora", "LoRA and QLoRA are parameter-efficient fine-tuning methods.", page=10),
        _chunk("attention", "Self-attention computes relationships between tokens.", page=2),
    ]
    embed_and_upsert(repo, qdrant, chunks, embedder=embedder)
    retriever = Retriever(repo, qdrant, embedder=embedder, reranker=NoOpReranker())

    results = retriever.retrieve(
        "LoRA QLoRA fine tuning",
        {"tenant_id": "engineering"},
        top_k=2,
    )

    assert results
    trace = results[0].metadata["_retrieval"]
    assert trace["final_rank"] == 1
    assert trace["embedding_model"] == "hash-64"
    assert trace["vector"] is not None or trace["lexical"] is not None

    diagnostics = retriever.diagnostics("大模型部署时如何减少显存占用？")
    assert diagnostics["semantic_embedding"] is False
    assert diagnostics["embedding_model"] == "hash-64"
    assert any("CJK" in warning for warning in diagnostics["warnings"])


def test_admin_ui_exposes_pdf_and_retrieval_workbench() -> None:
    app = FastAPI()
    app.include_router(create_admin_ui_router())

    response = TestClient(app).get("/admin/ui")

    assert response.status_code == 200
    assert "Ragbot Control Plane" in response.text
    assert "Single PDF ingestion" in response.text
    assert "Retrieval Playground" in response.text
    assert "vector, lexical and RRF" in response.text
    assert "sessionStorage" in response.text
