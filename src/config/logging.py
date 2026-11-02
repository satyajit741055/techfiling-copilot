"""
Logging Configuration for TechFiling Copilot.

Provides a centralized logging setup that can be used across all modules.

Features:
- Structured logging (text or JSON format)
- File rotation for production
- Environment-specific formatting
- Third-party library noise reduction
- Correlation ids (request, thread, turn) on every record, from ``log_context``

Usage:
    from config.logging import get_logger

    logger = get_logger(__name__)
    logger.info("Processing started")
    logger.error("Something went wrong", exc_info=True)
"""

import contextlib
import json
import logging
import sys
from collections.abc import Iterator
from contextvars import ContextVar, Token
from datetime import UTC, datetime
from functools import lru_cache
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Literal

from .settings import ROOT, settings

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

# Correlation ids for the current request / chat turn. Context variables follow asyncio
# tasks and ``asyncio.to_thread``, so retrieval threads and graph nodes log the same ids.
LOG_CONTEXT_FIELDS = ("request_id", "thread_id", "turn_id")
_LOG_CONTEXT: dict[str, ContextVar[str | None]] = {
    name: ContextVar(f"log_{name}", default=None) for name in LOG_CONTEXT_FIELDS
}
_STANDARD_ATTRS = set(vars(logging.LogRecord("", 0, "", 0, "", (), None))) | {
    "message",
    "asctime",
    "log_ctx",
    *LOG_CONTEXT_FIELDS,
}


def bind_log_context(**ids: str | None) -> list[Token[str | None]]:
    """Set correlation ids for the current context; undo with ``reset_log_context``."""
    return [_LOG_CONTEXT[name].set(value) for name, value in ids.items() if value is not None]


def reset_log_context(tokens: list[Token[str | None]]) -> None:
    for token in reversed(tokens):
        # A generator closed from another context cannot reset; its context ends anyway.
        with contextlib.suppress(ValueError):
            token.var.reset(token)


@contextlib.contextmanager
def log_context(**ids: str | None) -> Iterator[None]:
    tokens = bind_log_context(**ids)
    try:
        yield
    finally:
        reset_log_context(tokens)


def current_log_context() -> dict[str, str]:
    return {name: value for name, var in _LOG_CONTEXT.items() if (value := var.get()) is not None}


class LogContextFilter(logging.Filter):
    """Adds the correlation ids to each record (``log_ctx`` is the short text form)."""

    def filter(self, record: logging.LogRecord) -> bool:
        ids = current_log_context()
        for name in LOG_CONTEXT_FIELDS:
            setattr(record, name, ids.get(name))
        short = [
            f"{label}={ids[name][:8]}"
            for name, label in (("request_id", "req"), ("turn_id", "turn"))
            if name in ids
        ]
        record.log_ctx = f"[{' '.join(short)}] " if short else ""
        return True


class JSONFormatter(logging.Formatter):
    """
    JSON formatter for structured logging in production.

    Outputs logs as JSON objects with timestamp, level, logger name, message, the
    correlation ids and any ``extra=`` fields passed to the log call.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        for name in LOG_CONTEXT_FIELDS:
            if value := getattr(record, name, None):
                log_data[name] = value

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields if present
        log_data.update(
            {key: value for key, value in vars(record).items() if key not in _STANDARD_ATTRS}
        )
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)

        return json.dumps(log_data, default=str)


def setup_logging(
    level: LogLevel | None = None,
    format_string: str | None = None,
) -> None:
    """
    Configure the root logger for the application.

    Args:
        level: Log level (defaults to settings.LOG_LEVEL)
        format_string: Custom format string (optional, ignored for JSON format)
    """
    level = level or settings.LOG_LEVEL

    # Determine formatter based on settings
    if settings.LOG_FORMAT == "json":
        formatter: logging.Formatter = JSONFormatter()
    else:
        # Text format
        if format_string is None:
            if settings.is_dev:
                # More verbose format for development
                format_string = (
                    "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(log_ctx)s%(message)s"
                )
            else:
                # Cleaner format for production
                format_string = "%(asctime)s | %(levelname)-8s | %(log_ctx)s%(message)s"
        formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")

    # Configure handlers
    handlers: list[logging.Handler] = []

    # Filing text contains characters (e.g. checkboxes) the Windows console codec cannot encode
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    # Console handler (always)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    handlers.append(console_handler)

    # File handler (if configured)
    if settings.LOG_FILE:
        log_file_path = Path(settings.LOG_FILE)
        if not log_file_path.is_absolute():
            log_file_path = ROOT / log_file_path
        log_file_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            log_file_path,
            maxBytes=settings.LOG_FILE_MAX_BYTES,
            backupCount=settings.LOG_FILE_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    for handler in handlers:
        handler.addFilter(LogContextFilter())

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level),
        handlers=handlers,
        force=True,  # Override any existing configuration
    )

    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


@lru_cache(maxsize=None)
def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for the given module name.

    Uses lru_cache to ensure we return the same logger instance
    for the same name, avoiding duplicate handlers.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance

    Example:
        logger = get_logger(__name__)
        logger.info("Starting process")
    """
    return logging.getLogger(name)


# Initialize logging on module import
setup_logging()
