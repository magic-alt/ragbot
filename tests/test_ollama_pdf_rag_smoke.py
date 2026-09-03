from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "ollama_pdf_rag_test.py"

_spec = importlib.util.spec_from_file_location("ollama_pdf_rag_test", SCRIPT)
assert _spec and _spec.loader
mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = mod
_spec.loader.exec_module(mod)


def _args(tmp_path: Path) -> SimpleNamespace:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return SimpleNamespace(
        model="qwen3.8:27b-mlx",
        ollama_timeout=300.0,
        reasoning_effort="none",
        docker_ollama_url="http://host.docker.internal:11434",
        embedding_model="qwen3-embedding:8b",
        embedding_dim=None,
        collection=None,
        data_dir=data_dir,
        port=8000,
        ollama_url="http://127.0.0.1:11434",
        server="http://127.0.0.1:8000",
        api_key=None,
        tenant="ollama-pdf-smoke",
        user="ollama-pdf-smoke",
        top_k=5,
        query="关键技术指标是什么？",
    )


def test_container_location_maps_host_data_to_data_mount(tmp_path: Path) -> None:
    args = _args(tmp_path)
    nested = args.data_dir / "manuals"
    nested.mkdir()
    pdf = nested / "spec.pdf"
    pdf.write_bytes(b"%PDF-test")

    assert mod._container_location(pdf, args.data_dir) == "/data/manuals/spec.pdf"


def test_manifest_forces_fresh_vectorization_and_container_paths(tmp_path: Path) -> None:
    args = _args(tmp_path)
    pdf = args.data_dir / "sample.pdf"
    pdf.write_bytes(b"%PDF-test")

    sources = mod._manifest_sources([pdf], data_dir=args.data_dir, tag="smoke")

    assert sources == [
        {
            "location": "/data/sample.pdf",
            "source_type": "pdf",
            "name": "sample.pdf",
            "tags": ["smoke"],
            "reuse_source": False,
            "dedupe_active_job": False,
        }
    ]


def test_embedding_probe_auto_detects_8b_dimension_and_collection(tmp_path: Path) -> None:
    args = _args(tmp_path)
    fake = {"data": [{"index": 0, "embedding": [0.0] * 4096}]}

    with patch.object(mod._impl, "_request_json", return_value=fake):
        actual = mod._probe_embedding(args)

    assert actual == 4096
    assert args.embedding_dim == 4096
    assert args.collection == "rag_chunks_smoke_qwen3_embedding_8b_4096"


def test_embedding_probe_rejects_explicit_dimension_mismatch(tmp_path: Path) -> None:
    args = _args(tmp_path)
    args.embedding_dim = 1024
    fake = {"data": [{"index": 0, "embedding": [0.0] * 4096}]}

    with patch.object(mod._impl, "_request_json", return_value=fake):
        with pytest.raises(mod.UserError, match="dimension mismatch"):
            mod._probe_embedding(args)


def test_explicit_collection_is_preserved_after_auto_dimension(tmp_path: Path) -> None:
    args = _args(tmp_path)
    args.collection = "my_qdrant_collection"
    fake = {"data": [{"index": 0, "embedding": [0.0] * 4096}]}

    with patch.object(mod._impl, "_request_json", return_value=fake):
        mod._probe_embedding(args)

    assert args.embedding_dim == 4096
    assert args.collection == "my_qdrant_collection"


def test_compose_env_uses_resolved_ollama_embedding_contract(tmp_path: Path) -> None:
    args = _args(tmp_path)
    args.embedding_dim = 4096
    args.collection = "rag_chunks_smoke_qwen3_embedding_8b_4096"

    env = mod._compose_env(args)

    assert env["RAGBOT_LLM_PROVIDER"] == "ollama"
    assert env["OLLAMA_MODEL"] == "qwen3.8:27b-mlx"
    assert env["EMBEDDING_MODEL"] == "qwen3-embedding:8b"
    assert env["EMBEDDING_API_KEY"] == "ollama"
    assert env["EMBEDDING_BASE_URL"] == "http://host.docker.internal:11434"
    assert env["QDRANT_DIM"] == "4096"
    assert env["QDRANT_COLLECTION"] == "rag_chunks_smoke_qwen3_embedding_8b_4096"
    assert env["RAGBOT_DATA_DIR"] == str(args.data_dir.resolve())


def test_compose_env_rejects_unresolved_dimension(tmp_path: Path) -> None:
    args = _args(tmp_path)
    with pytest.raises(ValueError, match="must be resolved"):
        mod._compose_env(args)


def test_search_requires_real_semantic_results(tmp_path: Path) -> None:
    args = _args(tmp_path)
    response = {
        "chunks": [{"chunk_id": "c1", "score": 0.8, "text": "evidence"}],
        "diagnostics": {
            "semantic_embedding": True,
            "embedding_model": "qwen3-embedding:8b",
            "vector_store": "QdrantClientAdapter",
        },
    }

    with patch.object(mod._impl, "_request_json", return_value=response):
        result = mod._search(args)

    assert result["chunks"][0]["chunk_id"] == "c1"


def test_search_rejects_hash_fallback(tmp_path: Path) -> None:
    args = _args(tmp_path)
    response = {
        "chunks": [{"chunk_id": "c1", "score": 0.1, "text": "lexical"}],
        "diagnostics": {
            "semantic_embedding": False,
            "embedding_model": "hash-4096",
        },
    }

    with patch.object(mod._impl, "_request_json", return_value=response):
        with pytest.raises(mod.UserError, match="not using semantic embeddings"):
            mod._search(args)


def test_manifest_written_with_selected_tenant(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mod, "TMP_DIR", tmp_path)
    monkeypatch.setattr(mod._impl, "TMP_DIR", tmp_path)
    path = mod._write_manifest(
        [
            {
                "location": "/data/a.pdf",
                "source_type": "pdf",
                "reuse_source": False,
                "dedupe_active_job": False,
            }
        ],
        tenant="tenant-smoke",
        batch_index=1,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["tenant_id"] == "tenant-smoke"
    assert payload["sources"][0]["location"] == "/data/a.pdf"
