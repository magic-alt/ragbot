from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .models import NormalizedDocument


@dataclass(frozen=True)
class ParserSpec:
    """Stable index-contract identity for one parser implementation/configuration."""

    provider: str
    strategy: str
    version: int = 1
    options_json: str = "{}"

    def __post_init__(self) -> None:
        if self.version < 1:
            raise ValueError("parser version must be >= 1")
        try:
            decoded = json.loads(self.options_json)
        except json.JSONDecodeError as exc:
            raise ValueError("parser options_json must contain valid JSON") from exc
        if not isinstance(decoded, dict):
            raise ValueError("parser options must be a JSON object")

    @property
    def options(self) -> dict[str, object]:
        return dict(json.loads(self.options_json))

    @property
    def config_hash(self) -> str:
        payload = {
            "provider": self.provider,
            "strategy": self.strategy,
            "version": self.version,
            "options": self.options,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:20]

    def metadata(self) -> dict[str, object]:
        return {
            "parser_provider": self.provider,
            "parser_strategy": self.strategy,
            "parser_version": self.version,
            "parser_config_hash": self.config_hash,
        }


@runtime_checkable
class DocumentParser(Protocol):
    """Parser port owned by Ragbot; external libraries stay behind adapters."""

    @property
    def spec(self) -> ParserSpec: ...

    def parse(
        self,
        data: bytes,
        *,
        name: str,
        media_type: str = "application/octet-stream",
        uri: str | None = None,
    ) -> NormalizedDocument: ...
