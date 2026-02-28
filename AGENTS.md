# Repository Guidelines

## Project Structure & Module Organization
`ragbot` is organized by runtime surface:
- `services/api/app/`: FastAPI gateway, agent graph/nodes, auth, retrieval, storage, observability.
- `services/worker/`: ingestion pipeline (`connectors/`, `jobs/`, dedup logic).
- `cli/`: `rag` command-line entrypoint.
- `contracts/`: shared API/tooling contracts (`types.py`, `types.ts`, `openapi.yaml`, `tools.schema.json`).
- `eval/`: evaluation runners and datasets.
- `infra/`: Docker, Helm chart, SQL migrations, and Qdrant init scripts.
- `tests/`: regression/integration tests (currently centered in `tests/test_agent.py`).

## Build, Test, and Development Commands
- `pip install -r requirements.txt`: install baseline runtime dependencies.
- `pip install -e ".[dev]"`: install project + dev extras (`pytest`, `pytest-asyncio`) for local development.
- `uvicorn services.api.app.api:app --reload --host 0.0.0.0 --port 8000`: run API locally with hot reload.
- `python -m pytest tests/test_agent.py -v`: run the main test suite.
- `docker-compose -f infra/docker/docker-compose.yml up -d`: start local infra stack (Postgres, Qdrant, Ollama, Jaeger).
- `rag ask "Postgres 在系统中的作用"`: quick CLI smoke test after install.

## Coding Style & Naming Conventions
Use Python 3.10+ with 4-space indentation, explicit type hints, and small focused modules. Follow existing naming:
- files/functions/variables: `snake_case`
- classes: `PascalCase`
- constants/env-backed settings: `UPPER_SNAKE_CASE`

When API or tool payloads change, keep contracts synchronized across `contracts/types.py`, `contracts/types.ts`, and `contracts/openapi.yaml` in the same PR.

## Testing Guidelines
Tests run via `pytest` and primarily use `unittest.TestCase` patterns. Name tests `test_*` and group behavior by suite class (for example `RouteTests`, `RetrievalAclTests`). For agent or retrieval changes, add/extend coverage in `tests/test_agent.py` and run the full file before opening a PR.

## Commit & Pull Request Guidelines
History mostly follows Conventional Commits (`feat:`, `fix:`, `docs:`). Keep commits scoped and imperative, for example:
- `feat: add repo filter to /search endpoint`
- `fix: handle missing API key in metrics route`

PRs should include:
- concise change summary and rationale
- linked issue/task (if available)
- test evidence (command + result)
- contract/docs updates when behavior or endpoints change

## Security & Configuration Tips
Do not commit secrets. Configure runtime via environment variables (`OPENAI_API_KEY`, `POSTGRES_DSN`, `QDRANT_URL`, `RAGBOT_API_KEYS`). Use least-privilege API keys and validate auth-sensitive changes with dedicated tests.
