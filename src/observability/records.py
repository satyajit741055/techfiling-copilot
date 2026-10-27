"""Per-query record schema, JSONL sink and request-scoped access to the latest record."""

from __future__ import annotations

import json
import logging
import threading
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
TIMEOUT_MARKERS = ("timeout", "timedout", "deadlineexceeded")


@dataclass
class QueryRecord:
    """One /questions request. Flat and JSON-serialisable so it loads straight into pandas/SQL."""

    timestamp: str
    request_id: str | None
    doc_id: str | None
    status: str  # success | error | timeout
    error_type: str | None = None
    error_stage: str | None = None  # retrieval | generation
    error_message: str | None = None
    process_seq: int = 0
    is_cold: bool = False
    retriever_init_seconds: float = 0.0
    retrieval_seconds: float = 0.0
    generation_seconds: float = 0.0
    llm_seconds: float = 0.0
    total_seconds: float = 0.0
    llm_attempts: int = 0
    retries: int = 0
    provider: str | None = None
    model: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    input_cost_usd: float = 0.0
    output_cost_usd: float = 0.0
    infra_cost_usd: float = 0.0
    total_cost_usd: float = 0.0
    validation_passed: bool | None = None
    insufficient_context: bool | None = None
    context_k: int | None = None
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def is_timeout_error(error: BaseException) -> bool:
    """True for timeouts of any provider SDK (httpx, openai, requests, ollama, builtins)."""
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        names = [cls.__name__.lower() for cls in type(current).__mro__]
        if isinstance(current, TimeoutError) or any(
            marker in name for name in names for marker in TIMEOUT_MARKERS
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class JsonlMetricsWriter:
    """Thread-safe, append-only daily JSONL writer. Never raises into the request path."""

    def __init__(self, directory: Path | str):
        self.directory = Path(directory)
        self._lock = threading.Lock()

    def write(self, record: QueryRecord) -> None:
        try:
            day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            line = json.dumps(record.to_dict(), ensure_ascii=False)
            with self._lock:
                self.directory.mkdir(parents=True, exist_ok=True)
                with (self.directory / f"queries-{day}.jsonl").open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
        except Exception:
            logger.exception("Failed to write query metrics record")


_last_record: ContextVar[QueryRecord | None] = ContextVar("last_query_record", default=None)


_request_id: ContextVar[str | None] = ContextVar("query_request_id", default=None)


def set_request_id(request_id: str | None) -> None:
    _request_id.set(request_id)


def get_request_id() -> str | None:
    return _request_id.get()


def set_last_record(record: QueryRecord | None) -> None:
    _last_record.set(record)


def get_last_record() -> QueryRecord | None:
    return _last_record.get()


def load_records(path: Path) -> list[dict[str, Any]]:
    """Load records from one JSONL file or every queries-*.jsonl / cases.jsonl in a directory."""
    if not path.exists():
        raise FileNotFoundError(
            f"No metrics found at {path}. Production logs are written by the API after it serves "
            "/questions requests; for a benchmark run pass --logs <results dir>/queries.jsonl."
        )
    files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    if not files:
        raise FileNotFoundError(f"No .jsonl files in {path}")
    records: list[dict[str, Any]] = []
    for file in files:
        with file.open(encoding="utf-8") as fh:
            records.extend(json.loads(line) for line in fh if line.strip())
    return records
