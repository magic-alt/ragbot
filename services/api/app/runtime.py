from __future__ import annotations

import os

from .auth.principal import validate_principal_coverage


def environment_name() -> str:
    return os.getenv("RAGBOT_ENV", "development").strip().lower() or "development"


def is_production() -> bool:
    return environment_name() in {"production", "prod"}


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def validate_production_environment() -> None:
    """Fail fast when a production deployment would silently degrade.

    Development keeps the convenient in-memory/hash fallbacks. Production must
    explicitly configure durable state, semantic embeddings, scoped caller
    identities and the durable ingestion worker path before the API is allowed
    to serve requests. The Agent SQL tool is fail-closed and, when explicitly
    enabled, must use a database identity separate from Ragbot's control-plane
    database plus an explicit schema allowlist.
    """
    if not is_production():
        return

    required = {
        "POSTGRES_DSN": "durable metadata, jobs and lexical retrieval",
        "QDRANT_URL": "durable vector retrieval",
        "EMBEDDING_MODEL": "semantic embeddings",
        "RAGBOT_API_KEYS": "authenticated API access",
        "RAGBOT_API_KEY_PRINCIPALS": "tenant/user-bound caller identity",
    }
    missing = [
        f"{name} ({reason})"
        for name, reason in required.items()
        if not os.getenv(name, "").strip()
    ]

    if not (os.getenv("EMBEDDING_API_KEY", "").strip() or os.getenv("OPENAI_API_KEY", "").strip()):
        missing.append("EMBEDDING_API_KEY or OPENAI_API_KEY (embedding authentication)")

    if missing:
        raise RuntimeError(
            "Production configuration is incomplete; refusing unsafe fallback: "
            + "; ".join(missing)
        )

    ingestion_mode = os.getenv("RAGBOT_INGESTION_MODE", "auto").strip().lower()
    if ingestion_mode not in {"auto", "worker"}:
        raise RuntimeError(
            "Production ingestion must use the durable worker path; "
            "set RAGBOT_INGESTION_MODE=worker or leave it as auto"
        )

    if _env_flag("RAGBOT_SQL_TOOL_ENABLED", False):
        sql_dsn = os.getenv("RAGBOT_SQL_DSN", "").strip()
        control_dsn = os.getenv("POSTGRES_DSN", "").strip()
        allowed_schemas = os.getenv("RAGBOT_SQL_ALLOWED_SCHEMAS", "").strip()
        if not sql_dsn:
            raise RuntimeError(
                "Production Agent SQL requires an isolated RAGBOT_SQL_DSN; "
                "POSTGRES_DSN is reserved for Ragbot control-plane state"
            )
        if sql_dsn == control_dsn:
            raise RuntimeError(
                "RAGBOT_SQL_DSN must not equal POSTGRES_DSN in production; "
                "use a dedicated read-only database role/query surface"
            )
        if not allowed_schemas:
            raise RuntimeError(
                "Production Agent SQL requires RAGBOT_SQL_ALLOWED_SCHEMAS"
            )

    api_keys = {key.strip() for key in os.environ["RAGBOT_API_KEYS"].split(",") if key.strip()}
    if not api_keys:
        raise RuntimeError("Production RAGBOT_API_KEYS must contain at least one key")
    try:
        validate_principal_coverage(api_keys)
    except ValueError as exc:
        raise RuntimeError(f"Invalid production API principal configuration: {exc}") from exc
