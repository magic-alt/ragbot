from __future__ import annotations

from services.api.app.routes import quick_import, sources
from services.api.app.storage.models import Chunk, Source
from services.platform import RuntimeProfile, TypedRegistry
from services.worker import pipeline
from services.worker.connectors.registry import (
    ConnectorCapabilities,
    ConnectorRegistry,
    ConnectorSpec,
    connector_registry,
)


def _toy_spec() -> ConnectorSpec:
    def build(location, extra):
        return {**dict(extra), "path": location.strip()}

    def validate(config):
        if not str(config.get("path") or "").strip():
            raise ValueError("toy path required")

    def run(source, _repo, _previous):
        return [
            Chunk(
                chunk_id="toy-chunk",
                doc_id=f"doc-{source.source_id}",
                tenant_id=source.tenant_id,
                chunk_index=0,
                text="toy evidence",
                path=source.config["path"],
                checksum="toy",
            )
        ]

    return ConnectorSpec(
        source_type="toy",
        runner=run,
        build_config=build,
        canonicalize=lambda value: value.strip().lower(),
        match_location=lambda value: 200 if value.lower().startswith("toy://") else 0,
        source_location=lambda config: str(config.get("path") or "") or None,
        validate_config=validate,
        capabilities=ConnectorCapabilities(multi_document=False),
    )


def test_typed_registry_rejects_duplicates_and_reports_available_ids():
    registry = TypedRegistry(kind="test")
    spec = _toy_spec()
    registry.register(spec)
    assert registry.get("connector:toy") is spec
    try:
        registry.register(spec)
    except ValueError as exc:
        assert "Duplicate test component registration" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("duplicate registration unexpectedly accepted")


def test_connector_registry_owns_inference_config_validation_and_execution(monkeypatch):
    registry = ConnectorRegistry()
    registry.register(_toy_spec())

    assert registry.infer_source_type("toy://Example") == "toy"
    assert registry.canonical_location("  TOY://Example  ", "toy") == "toy://example"
    assert registry.build_source_config("toy", "toy://example", {"tag": "x"}) == {
        "tag": "x",
        "path": "toy://example",
    }

    source = Source(
        source_id="toy-source",
        tenant_id="tenant-a",
        source_type="toy",
        name="Toy",
        config={"path": "toy://example"},
    )
    monkeypatch.setattr(pipeline, "connector_registry", lambda: registry)
    chunks = list(pipeline._run_connector(source, object()))
    assert [chunk.chunk_id for chunk in chunks] == ["toy-chunk"]


def test_api_and_quick_import_use_connector_registry_without_source_switches(monkeypatch):
    registry = ConnectorRegistry()
    registry.register(_toy_spec())
    monkeypatch.setattr(sources, "connector_registry", lambda: registry)
    monkeypatch.setattr(quick_import, "connector_registry", lambda: registry)

    sources._validate_source_type("toy")
    sources._validate_source_config("toy", {"path": "toy://example"})
    assert quick_import.infer_source_type("toy://example") == "toy"
    assert quick_import.build_source_config("toy", "toy://example") == {"path": "toy://example"}
    assert quick_import.canonical_location("TOY://Example") == "toy://example"


def test_builtin_connector_metadata_is_capability_driven():
    registry = connector_registry()
    source_types = set(registry.source_types())
    assert {"local_fs", "pdf", "web", "repo", "s3", "gdrive", "notion", "confluence"} <= source_types
    assert registry.get("gdrive").capabilities.incremental is True
    assert registry.get("gdrive").capabilities.credentials is True
    assert registry.get("local_fs").capabilities.multi_document is True
    assert registry.get("pdf").capabilities.multi_document is False
    assert registry.get("repo").default_chunk_strategy == "structural"


def test_runtime_profile_is_typed_and_non_secret():
    env = {
        "RAGBOT_ENV": "production",
        "POSTGRES_DSN": "postgresql://secret",
        "QDRANT_URL": "https://qdrant.internal",
        "QDRANT_API_KEY": "secret",
        "EMBEDDING_MODEL": "text-embedding-3-small",
        "EMBEDDING_API_KEY": "secret",
        "RAGBOT_LLM_PROVIDER": "ollama",
        "RAGBOT_RERANK_ENABLED": "true",
        "RAGBOT_RERANK_PROVIDER": "local",
        "RAGBOT_UPLOAD_STORE": "filesystem",
    }
    profile = RuntimeProfile.from_environment(env, connector_ids=["connector:pdf"])
    public = profile.as_public_dict()
    assert public == {
        "environment": "production",
        "repository_provider": "postgres",
        "vector_provider": "qdrant",
        "embedding_provider": "openai-compatible",
        "llm_provider": "ollama",
        "reranker_provider": "local",
        "upload_provider": "filesystem",
        "connector_ids": ["connector:pdf"],
    }
    serialized = repr(public)
    assert "postgresql://secret" not in serialized
    assert "QDRANT_API_KEY" not in serialized
