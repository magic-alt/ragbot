# Ragbot Platform Extension Boundary

Ragbot's platform SPI separates stable product semantics from vendor/framework adapters. The public extension boundary is registry-based; request handlers and ingestion orchestration must not grow provider/source-specific switch statements.

## Component IDs

Runtime components use stable `<kind>:<provider>` IDs, for example:

- `repository:postgres`
- `vector:qdrant`
- `embedding:openai-compatible`
- `llm:openai`
- `llm:ollama`
- `reranker:cohere`
- `connector:gdrive`

`RuntimeProfile` is a non-secret snapshot of the selected providers. `/admin/runtime` exposes the profile and registered component capabilities for diagnostics without returning DSNs, API keys or connector credentials.

## Connector SPI

`services.worker.connectors.registry.ConnectorSpec` owns the connector-specific behavior that was previously spread across API validation, Quick Import and the worker pipeline:

1. source-type inference from locations;
2. canonical location identity;
3. `location -> Source.config` construction;
4. configuration validation, including parser/chunker eligibility;
5. capability metadata such as incremental, credentials, remote and multi-document behavior;
6. ingestion execution.

Built-in connectors are registered once. API, CLI/Quick Import and worker execution resolve the same specification.

Third-party packages may expose one immutable `ConnectorSpec` through the Python entry-point group `ragbot.connectors`. Entry points are discovered deterministically during process bootstrap/registry discovery. User request data is never interpreted as an import path.

A connector adapter may import its heavy/vendor dependency lazily inside its runner so installing Ragbot core does not pull every integration dependency.

## Runtime component SPI

`services.api.app.runtime_registry.RuntimeFactorySpec` is the first common selection boundary for repository, vector, embedding, LLM and reranker providers. Optional packages can register a factory through `ragbot.runtime_components`.

The factory accepts a typed/normalized config mapping from the bootstrap layer. This change intentionally keeps several existing provider builders internally environment-backed for compatibility. Follow-up issues own the deeper constructor injection and transport work:

- #51 LLM provider/router plane;
- #52 embedding plane;
- #55 storage capability split.

Those changes should extend this registry rather than add new branches to `factory.py`.

## Compatibility rules

- Parser/Chunker ports remain authoritative for document transformation.
- PostgreSQL remains the reference durable control-plane store; Qdrant remains a derived vector index.
- Connector registration must not weaken tenant/ACL filtering or source-generation fencing.
- Vendor IDs/config fields must not leak into Agent nodes or generic ingestion orchestration.
- Secrets are references or deployment configuration; public capability metadata must be non-secret.
- Unknown configured provider IDs fail fast and list the available registrations.

## Conformance expectation

A new connector should be testable by constructing/registering a `ConnectorSpec` and then using the existing source validation, Quick Import and pipeline paths without editing those modules. A new runtime provider should register one `RuntimeFactorySpec`, select its provider ID through bootstrap configuration, and require no new provider branch in `factory.py`.
