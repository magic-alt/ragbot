"""Database migration runner used by CI, Docker Compose, and Kubernetes.

PostgreSQL's ``/docker-entrypoint-initdb.d`` only runs on a brand-new data
volume. Ragbot therefore owns an explicit migration lifecycle so upgrades of
long-lived deployments apply new SQL files before the API becomes ready.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

_MIGRATION_LOCK_ID = 724_268_071
_DEFAULT_MIGRATIONS_DIR = Path(__file__).resolve().parents[4] / "infra" / "migrations"


def migration_files(directory: Optional[Path] = None) -> list[Path]:
    root = directory or Path(os.getenv("RAGBOT_MIGRATIONS_DIR", _DEFAULT_MIGRATIONS_DIR))
    if not root.exists() or not root.is_dir():
        raise RuntimeError(f"Migration directory does not exist: {root}")
    files = sorted(path for path in root.glob("*.sql") if path.is_file())
    if not files:
        raise RuntimeError(f"No SQL migrations found in: {root}")
    return files


def apply_migrations(
    dsn: str,
    directory: Optional[Path] = None,
) -> list[str]:
    """Apply every unapplied migration exactly once and return applied names."""
    if not dsn.strip():
        raise RuntimeError("POSTGRES_DSN is required to apply migrations")

    try:
        import psycopg
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "psycopg is required for database migrations; install the postgres extra"
        ) from exc

    files = migration_files(directory)
    applied_now: list[str] = []

    with psycopg.connect(dsn) as conn:
        conn.execute("SELECT pg_advisory_lock(%s)", (_MIGRATION_LOCK_ID,))
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    migration_name TEXT PRIMARY KEY,
                    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            conn.commit()

            rows = conn.execute("SELECT migration_name FROM schema_migrations").fetchall()
            applied = {row[0] for row in rows}

            for path in files:
                if path.name in applied:
                    continue
                logger.info("Applying database migration %s", path.name)
                sql = path.read_text(encoding="utf-8")
                try:
                    with conn.transaction():
                        conn.execute(sql)
                        conn.execute(
                            "INSERT INTO schema_migrations (migration_name) VALUES (%s)",
                            (path.name,),
                        )
                except Exception:
                    logger.exception("Database migration failed: %s", path.name)
                    raise
                applied_now.append(path.name)
        finally:
            conn.execute("SELECT pg_advisory_unlock(%s)", (_MIGRATION_LOCK_ID,))

    return applied_now


def main() -> int:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    dsn = os.getenv("POSTGRES_DSN", "")
    applied = apply_migrations(dsn)
    if applied:
        logger.info("Applied %d migration(s): %s", len(applied), ", ".join(applied))
    else:
        logger.info("Database schema is already up to date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
