from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _postgres_block(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    start = text.index("\n  postgres:\n")
    end = text.index("\n  qdrant:\n", start)
    return text[start:end]


def test_controller_compose_does_not_publish_postgres_to_host() -> None:
    block = _postgres_block(ROOT / "docker-compose.yml")
    assert "\n    ports:\n" not in block
    assert "5432:5432" not in block
    assert "RAGBOT_POSTGRES_PORT" not in block


def test_infra_compose_does_not_publish_postgres_to_host() -> None:
    block = _postgres_block(ROOT / "infra" / "docker" / "docker-compose.yml")
    assert "\n    ports:\n" not in block
    assert "5432:5432" not in block
    assert "RAGBOT_POSTGRES_PORT" not in block
