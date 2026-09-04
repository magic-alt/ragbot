from __future__ import annotations

from typing import Any, Dict, List, Optional

from contracts.types import SqlResult


class DisabledSqlEngine:
    """Fail-closed SQL tool used unless an isolated query database is enabled."""

    enabled = False

    def query(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> SqlResult:
        raise PermissionError(
            "Agent SQL tool is disabled. Configure RAGBOT_SQL_TOOL_ENABLED=true "
            "with an isolated RAGBOT_SQL_DSN to enable it."
        )

    def introspect_schema(self) -> List[Dict[str, Any]]:
        return []

    def close(self) -> None:
        return None
