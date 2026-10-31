"""Export a readable view of an embedded Qdrant ``storage.sqlite`` file.

The database stores pickled Qdrant ``PointStruct`` objects rather than ordinary
human-readable rows. This script opens the database in read-only mode, decodes
the points, and writes:

* ``summary.json`` - collection statistics and payload/vector information
* ``points.jsonl`` - readable payloads plus vector summaries

Only the first 100 matching points are exported by default. Use ``--limit 0``
to export every matching point and ``--include-vectors`` to include complete
dense and sparse vectors.

Example:
    uv run python exploration/explore_qdrant_sqlite.py data/qdrant/collection/techfiling_20261030T163855Z/storage.sqlite

The point column contains Python pickle data. Only open databases you trust,
because unpickling can execute code supplied by the database.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sqlite3
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

DEFAULT_LIMIT = 100
TABLE_NAME = "points"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "database",
        type=Path,
        help="Path to the Qdrant collection's storage.sqlite file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "qdrant_storage_view",
        help="Directory for summary.json and points.jsonl",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"Maximum matching points to export; use 0 for all (default: {DEFAULT_LIMIT})",
    )
    parser.add_argument(
        "--query",
        help="Case-insensitive text to find in each point's payload",
    )
    parser.add_argument(
        "--doc-id",
        help="Export only points belonging to this document ID",
    )
    parser.add_argument(
        "--include-vectors",
        action="store_true",
        help="Write complete dense and sparse vectors instead of vector summaries",
    )
    return parser.parse_args()


def connect_read_only(database: Path) -> sqlite3.Connection:
    """Open SQLite using a URI that prevents this script from writing."""
    try:
        uri = f"{database.resolve().as_uri()}?mode=ro"
        return sqlite3.connect(uri, uri=True)
    except sqlite3.Error as exc:
        raise SystemExit(f"Could not open SQLite database read-only: {database}\n{exc}") from exc


def jsonable(value: Any) -> Any:
    """Convert common Qdrant/Python values into JSON-compatible values."""
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, bytes | bytearray):
        return [jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        return jsonable(value.model_dump())
    return str(value)


def vector_summary(vectors: Mapping[str, Any], include_vectors: bool) -> dict[str, Any]:
    result: dict[str, Any] = {}

    dense = vectors.get("dense")
    if dense is not None:
        dense_values = jsonable(dense)
        if include_vectors:
            result["dense"] = dense_values
        else:
            result["dense"] = {
                "dimensions": len(dense_values),
                "sample": dense_values[:8],
            }

    sparse = vectors.get("bm25")
    if sparse is not None:
        indices = jsonable(getattr(sparse, "indices", []))
        values = jsonable(getattr(sparse, "values", []))
        if include_vectors:
            result["bm25"] = {"indices": indices, "values": values}
        else:
            result["bm25"] = {
                "non_zero": len(indices),
                "sample": [
                    {"index": index, "value": value}
                    for index, value in zip(indices[:8], values[:8])
                ],
            }
    return result


def matches(payload: Mapping[str, Any], query: str | None, doc_id: str | None) -> bool:
    if doc_id and payload.get("doc_id") != doc_id:
        return False
    if query:
        searchable = json.dumps(jsonable(payload), ensure_ascii=False).casefold()
        return query.casefold() in searchable
    return True


def main() -> None:
    args = parse_args()
    database = args.database.expanduser().resolve()
    if not database.is_file():
        raise SystemExit(f"Database file does not exist: {database}")
    if args.limit < 0:
        raise SystemExit("--limit must be 0 or a positive integer")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    points_path = output_dir / "points.jsonl"
    summary_path = output_dir / "summary.json"

    summary: dict[str, Any] = {
        "database": str(database),
        "table": TABLE_NAME,
        "limit": args.limit,
        "query": args.query,
        "doc_id": args.doc_id,
        "include_vectors": args.include_vectors,
        "rows_scanned": 0,
        "matching_points": 0,
        "exported_points": 0,
        "payload_keys": Counter(),
        "documents": Counter(),
        "tickers": Counter(),
        "dense_dimensions": Counter(),
        "sparse_non_zero_counts": Counter(),
    }

    try:
        connection = connect_read_only(database)
        with connection:
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
            if TABLE_NAME not in tables:
                raise SystemExit(f"Expected table {TABLE_NAME!r} was not found in {database}")

            total_rows = connection.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0]
            summary["database_rows"] = total_rows

            with points_path.open("w", encoding="utf-8") as points_file:
                for row_id, blob in connection.execute(
                    f"SELECT id, point FROM {TABLE_NAME} ORDER BY rowid"
                ):
                    summary["rows_scanned"] += 1
                    try:
                        point = pickle.loads(blob)
                    except (pickle.PickleError, EOFError, ImportError, AttributeError) as exc:
                        raise SystemExit(
                            f"Could not decode row {row_id!r}. "
                            "Run this with the project's qdrant-client environment.\n"
                            f"{exc}"
                        ) from exc

                    payload = jsonable(getattr(point, "payload", None) or {})
                    if not isinstance(payload, dict):
                        payload = {"value": payload}
                    for key in payload:
                        summary["payload_keys"][key] += 1
                    if not matches(payload, args.query, args.doc_id):
                        continue

                    summary["matching_points"] += 1
                    vectors = getattr(point, "vector", None) or {}
                    vector_info = vector_summary(vectors, args.include_vectors)
                    dense = vectors.get("dense")
                    sparse = vectors.get("bm25")
                    if dense is not None:
                        summary["dense_dimensions"][len(dense)] += 1
                    if sparse is not None:
                        summary["sparse_non_zero_counts"][len(getattr(sparse, "indices", []))] += 1
                    if payload.get("doc_id"):
                        summary["documents"][str(payload["doc_id"])] += 1
                    if payload.get("ticker"):
                        summary["tickers"][str(payload["ticker"])] += 1

                    if args.limit and summary["exported_points"] >= args.limit:
                        continue
                    record = {
                        "row_id": row_id,
                        "point_id": str(getattr(point, "id", row_id)),
                        "payload": payload,
                        "vectors": vector_info,
                    }
                    points_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    summary["exported_points"] += 1
    finally:
        if "connection" in locals():
            connection.close()

    summary["payload_keys"] = dict(sorted(summary["payload_keys"].items()))
    summary["documents"] = dict(sorted(summary["documents"].items()))
    summary["tickers"] = dict(sorted(summary["tickers"].items()))
    summary["dense_dimensions"] = dict(summary["dense_dimensions"])
    summary["sparse_non_zero_counts"] = dict(summary["sparse_non_zero_counts"])
    with summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, ensure_ascii=False, indent=2)
        summary_file.write("\n")

    print(f"Scanned {summary['rows_scanned']:,} rows; found {summary['matching_points']:,} matches.")
    print(f"Exported {summary['exported_points']:,} points to {points_path}")
    print(f"Wrote statistics to {summary_path}")


if __name__ == "__main__":
    main()
