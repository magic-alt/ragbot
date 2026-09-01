from __future__ import annotations

import os

from .auth.principal import validate_principal_coverage


def environment_name() -> str:
    return os.getenv("RAGBOT_ENV", "development").strip().lower() or "development"


def is_production() -> bool:
    return environment_name() in {"production", "prod"}


def validate_production_environment() -> None:
    """Fail fast when a production deployment would silently degrade.

    Development keeps the convenient in-memory/hash fallbacks. Production must
    explicitly configure durable state, semantic embeddings and scoped caller
    identities before the API is allowed to serve requests.
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
    missing = [f"{name} ({reason})" for name, reason in required.items() if not os.getenv(name, "").strip()]

    if not (os.getenv("EMBEDDING_API_KEY", "").strip() or os.getenv("OPENAI_API_KEY", "").strip()):
        missing.append("EMBEDDING_API_KEY or OPENAI_API_KEY (embedding authentication)")

    if missing:
        raise RuntimeError(
            "Production configuration is incomplete; refusing unsafe fallback: "
            + "; ".join(missing)
        )

    api_keys = {key.strip() for key in os.environ["RAGBOT_API_KEYS"].split(",") if key.strip()}
    if not api_keys:
        raise RuntimeError("Production RAGBOT_API_KEYS must contain at least one key")
    try:
        validate_principal_coverage(api_keys)
    except ValueError as exc:
        raise RuntimeError(f"Invalid production API principal configuration: {exc}") from exc
