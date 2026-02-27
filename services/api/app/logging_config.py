"""Structured logging configuration for ragbot.

Call ``setup_logging()`` once at application startup to configure
JSON-formatted structured logging.  Falls back to standard library
``logging`` format when the optional ``python-json-logger`` package
is not installed.

Usage::

    from services.api.app.logging_config import setup_logging
    setup_logging(level="INFO", json_format=True)
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Optional


def setup_logging(
    level: Optional[str] = None,
    json_format: Optional[bool] = None,
) -> None:
    """Configure root logger with optional JSON formatting.

    Args:
        level: Log level name (default: ``LOG_LEVEL`` env var or ``"INFO"``).
        json_format: If True, emit JSON lines.  Default is True when
            ``LOG_FORMAT=json`` env var is set, otherwise plain text.
    """
    level = level or os.getenv("LOG_LEVEL", "INFO")
    if json_format is None:
        json_format = os.getenv("LOG_FORMAT", "").lower() == "json"

    root = logging.getLogger()
    root.setLevel(level.upper())

    # Remove any existing handlers to avoid duplicates on re-initialization
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level.upper())

    if json_format:
        formatter = _json_formatter()
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    handler.setFormatter(formatter)
    root.addHandler(handler)

    # Suppress overly noisy third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def _json_formatter() -> logging.Formatter:
    try:
        from pythonjsonlogger import jsonlogger

        return jsonlogger.JsonFormatter(
            fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
            rename_fields={"asctime": "timestamp", "levelname": "level"},
        )
    except ImportError:
        logging.getLogger(__name__).warning(
            "python-json-logger not installed; using plain text format"
        )
        return logging.Formatter(
            fmt="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
