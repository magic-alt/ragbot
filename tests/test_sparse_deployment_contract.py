from __future__ import annotations

from pathlib import Path


def test_sparse_compose_overlays_keep_api_worker_contract_in_sync() -> None:
    root = Path("docker-compose.sparse.yml").read_text(encoding="utf-8")
    infra = Path("infra/docker/docker-compose.sparse.yml").read_text(encoding="utf-8")
    keys = (
        "RAGBOT_SPARSE_ENABLED",
        "RAGBOT_SPARSE_PROVIDER",
        "RAGBOT_SPARSE_MODEL",
        "RAGBOT_SPARSE_REVISION",
        "RAGBOT_SPARSE_VECTOR_NAME",
        "RAGBOT_SPARSE_MODIFIER",
        "RAGBOT_SPARSE_TOKENIZER",
        "RAGBOT_SPARSE_LANGUAGE",
        "RAGBOT_SPARSE_BATCH_SIZE",
    )
    for content in (root, infra):
        assert content.count("RAGBOT_INSTALL_SPARSE") == 3
        for key in keys:
            assert content.count(key) == 2, (key, content.count(key))


def test_sparse_build_arg_is_opt_in_in_both_dockerfiles() -> None:
    for filename in ("Dockerfile", "infra/docker/Dockerfile"):
        content = Path(filename).read_text(encoding="utf-8")
        assert "ARG RAGBOT_INSTALL_SPARSE=false" in content
        assert 'if [ "$RAGBOT_INSTALL_SPARSE" = "true" ]' in content
        assert 'fastembed>=0.7' in content
