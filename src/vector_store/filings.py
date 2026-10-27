"""Filing catalog metadata (``<doc_id>.document.json`` written by the parser)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_filing_metadata(
    parsed_dir: Path, doc_ids: list[str] | None = None
) -> dict[str, dict[str, Any]]:
    """Map ``doc_id`` to its parsed document metadata; missing files are skipped."""
    wanted = set(doc_ids) if doc_ids is not None else None
    filings = {}
    for path in sorted(Path(parsed_dir).glob("*.document.json")):
        meta = json.loads(path.read_text(encoding="utf-8"))
        if wanted is None or meta["doc_id"] in wanted:
            filings[meta["doc_id"]] = meta
    return filings
