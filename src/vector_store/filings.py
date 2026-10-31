"""Filing catalog metadata (``<doc_id>.document.json`` written by the parser)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_filing_metadata(
    parsed_dir: Path, doc_ids: list[str] | None = None
) -> dict[str, dict[str, Any]]:
    """Map ``doc_id`` to its parsed document metadata; missing files are skipped.

    With ``doc_ids`` only those documents' files are read, so unrelated or stale
    files in ``parsed_dir`` are never opened.
    """
    parsed_dir = Path(parsed_dir)
    paths = (
        sorted(parsed_dir.glob("*.document.json"))
        if doc_ids is None
        else [parsed_dir / f"{d}.document.json" for d in doc_ids]
    )
    filings = {}
    for path in paths:
        if path.is_file():
            meta = json.loads(path.read_text(encoding="utf-8"))
            filings[meta["doc_id"]] = meta
    return filings
