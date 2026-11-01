"""Qdrant-backed chunk index: dense + sparse (BM25) vectors, payload filters, server-side RRF."""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
from qdrant_client import QdrantClient, models

from config.logging import get_logger
from vector_store.filters import check_group_by, check_sequence, merge_filters, to_qdrant_filter
from vector_store.sparse import AVG_DOC_TOKENS, K1, B, encode_document, encode_query

logger = get_logger(__name__)

DENSE = "dense"
SPARSE = "bm25"
PAYLOAD_INDEXES = {
    "doc_id": models.PayloadSchemaType.KEYWORD,
    "ticker": models.PayloadSchemaType.KEYWORD,
    "form": models.PayloadSchemaType.KEYWORD,
    "section": models.PayloadSchemaType.KEYWORD,
    "type": models.PayloadSchemaType.KEYWORD,
    "table_id": models.PayloadSchemaType.KEYWORD,
    "fiscal_year": models.PayloadSchemaType.INTEGER,
}
FILING_KEY = "filing"
_ID_NAMESPACE = uuid.UUID("5b1f1c1e-8a58-4d6b-9d46-0c7c4f7d2a10")


def point_id(chunk_id: str) -> str:
    """Deterministic point id, so re-ingesting a chunk is an idempotent upsert."""
    return str(uuid.uuid5(_ID_NAMESPACE, chunk_id))


class QdrantChunkStore:
    """Thin wrapper over one Qdrant collection holding filing chunks."""

    def __init__(
        self,
        url: str,
        collection: str,
        *,
        api_key: str | None = None,
        timeout: float = 30.0,
        client: QdrantClient | None = None,
        path: str | Path | None = None,
    ) -> None:
        """``path`` opens an embedded on-disk store (no server) and takes precedence over ``url``."""
        self.collection = collection
        self.local = client is None and (path is not None or url == ":memory:")
        if client is not None:
            self.client = client
        elif path is not None:
            Path(path).mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(path))
        elif url == ":memory:":
            self.client = QdrantClient(location=":memory:")
        else:
            self.client = QdrantClient(url=url, api_key=api_key, timeout=int(timeout))

    @classmethod
    def from_settings(
        cls, settings: Any | None = None, collection: str | None = None
    ) -> QdrantChunkStore:
        if settings is None:
            from config.settings import settings as global_settings

            settings = global_settings
        return cls(
            settings.QDRANT_URL,
            collection or settings.QDRANT_COLLECTION,
            api_key=settings.QDRANT_API_KEY,
            timeout=settings.QDRANT_TIMEOUT,
            path=getattr(settings, "QDRANT_PATH", None),
        )

    def exists(self) -> bool:
        return self.client.collection_exists(self.collection)

    def count(self) -> int:
        return self.client.count(self.collection, exact=True).count if self.exists() else 0

    def index_metadata(self) -> dict[str, Any]:
        """Embedding model and BM25 metadata stored on the collection (manifest-shaped)."""
        metadata = self.client.get_collection(self.collection).config.metadata or {}
        return {
            "embedding_model": metadata.get("embedding_model", {}),
            "bm25": metadata.get("bm25", {}),
        }

    def avg_doc_tokens(self) -> float:
        """BM25 average chunk length stored on the collection (legacy collections: default)."""
        return float(self.index_metadata()["bm25"].get("avg_doc_tokens", AVG_DOC_TOKENS))

    @staticmethod
    def _sparse_text(chunk: Mapping[str, Any]) -> str:
        return chunk.get("embed_text") or chunk.get("text", "")

    def dimension(self) -> int:
        params = self.client.get_collection(self.collection).config.params.vectors
        return int(params[DENSE].size)  # type: ignore[index]

    def ensure_collection(
        self,
        dimension: int,
        embedding_model: Mapping[str, Any] | None = None,
        avg_doc_tokens: float = AVG_DOC_TOKENS,
    ) -> None:
        """Create the collection and payload indexes if missing; verify the dimension otherwise.

        ``avg_doc_tokens`` is recorded only on creation; an existing collection keeps its
        value so later upserts stay consistent with the points already indexed.
        """
        if self.exists():
            if self.dimension() != dimension:
                raise ValueError(
                    f"Collection {self.collection!r} has dimension {self.dimension()}, "
                    f"embeddings have {dimension}; use a new collection per embedding model"
                )
            stored = self.index_metadata()["embedding_model"].get("name")
            incoming = (embedding_model or {}).get("name")
            if stored and incoming and stored != incoming:
                raise ValueError(
                    f"Collection {self.collection!r} holds {stored!r} embeddings, not {incoming!r}; "
                    "use a new collection per embedding model"
                )
            self.ensure_payload_indexes()
            return
        self.client.create_collection(
            self.collection,
            vectors_config={
                DENSE: models.VectorParams(size=dimension, distance=models.Distance.COSINE)
            },
            sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
            metadata={
                "embedding_model": dict(embedding_model or {}),
                "bm25": {"avg_doc_tokens": round(avg_doc_tokens, 2), "k1": K1, "b": B},
            },
        )
        self.ensure_payload_indexes()
        logger.info(f"Created Qdrant collection {self.collection!r} (dim={dimension})")

    def ensure_payload_indexes(self) -> list[str]:
        """Create any missing filter/group-by payload indexes; return the fields created.

        Embedded Qdrant ignores payload indexes (filters still work by scanning), so this
        is a no-op there. On a server, older collections gain new indexes without a rebuild.
        """
        if self.local:
            return []
        existing = self.client.get_collection(self.collection).payload_schema or {}
        created = [field for field in PAYLOAD_INDEXES if field not in existing]
        for field in created:
            self.client.create_payload_index(self.collection, field, PAYLOAD_INDEXES[field])
        return created

    def upsert_chunks(
        self,
        chunks: Sequence[Mapping[str, Any]],
        vectors: np.ndarray,
        batch_size: int = 256,
        filings: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> int:
        """Upsert chunks with their dense vectors; sparse vectors are computed here.

        ``filings`` maps ``doc_id`` to filing metadata (company, ticker, fiscal year, ...),
        stored on each of that document's chunks under ``filing`` so the catalog can be
        served from the collection itself.
        """
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks versus {len(vectors)} vectors")
        avg_doc_tokens = self.avg_doc_tokens()
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            points = []
            for chunk, vector in zip(batch, vectors[start : start + batch_size]):
                indices, values = encode_document(self._sparse_text(chunk), avg_doc_tokens)
                points.append(
                    models.PointStruct(
                        id=point_id(chunk["chunk_id"]),
                        vector={
                            DENSE: np.asarray(vector, dtype=np.float32).tolist(),
                            SPARSE: models.SparseVector(indices=indices, values=values),
                        },
                        payload={**chunk, **self._filing_payload(chunk, filings)},
                    )
                )
            self.client.upsert(self.collection, points=points, wait=True)
        return len(chunks)

    @staticmethod
    def _filing_payload(
        chunk: Mapping[str, Any], filings: Mapping[str, Mapping[str, Any]] | None
    ) -> dict[str, Any]:
        filing = (filings or {}).get(chunk["doc_id"])
        return {FILING_KEY: dict(filing)} if filing else {}

    def index_chunks(
        self,
        chunks: Sequence[Mapping[str, Any]],
        vectors: np.ndarray,
        embedding_model: Mapping[str, Any] | None = None,
        *,
        replace_documents: bool = True,
        filings: Mapping[str, Mapping[str, Any]] | None = None,
        avg_doc_tokens: float = AVG_DOC_TOKENS,
    ) -> int:
        """Ensure the collection, upsert the chunks, then drop each document's stale chunks.

        Upserting first means a failed run never leaves a document without chunks.
        ``avg_doc_tokens`` (BM25 ``avgdl``) applies only when the collection is created.
        The fixed default beat the measured corpus average (~129) on the retrieval goldens.
        """
        self.ensure_collection(int(vectors.shape[1]), embedding_model, avg_doc_tokens)
        count = self.upsert_chunks(chunks, vectors, filings=filings)
        if replace_documents:
            new_ids: dict[str, list[str]] = {}
            for chunk in chunks:
                new_ids.setdefault(chunk["doc_id"], []).append(point_id(chunk["chunk_id"]))
            for doc_id, ids in new_ids.items():
                self.client.delete(
                    self.collection,
                    points_selector=models.FilterSelector(
                        filter=models.Filter(
                            must=[
                                models.FieldCondition(
                                    key="doc_id", match=models.MatchValue(value=doc_id)
                                )
                            ],
                            must_not=[models.HasIdCondition(has_id=cast(Any, ids))],
                        )
                    ),
                    wait=True,
                )
        return count

    def delete_document(self, doc_id: str) -> None:
        self.client.delete(
            self.collection,
            points_selector=models.FilterSelector(filter=self._filter({"doc_id": doc_id})),
            wait=True,
        )

    def doc_ids(self) -> list[str]:
        """Distinct ``doc_id`` values currently indexed."""
        found: set[str] = set()
        offset = None
        while True:
            points, offset = self.client.scroll(
                self.collection,
                limit=1000,
                offset=offset,
                with_payload=["doc_id"],
                with_vectors=False,
            )
            found.update(p.payload["doc_id"] for p in points if p.payload and "doc_id" in p.payload)
            if offset is None:
                return sorted(found)

    def list_filings(self) -> list[dict[str, Any]]:
        """One filing-metadata record per indexed document, sorted by ``doc_id``.

        Documents indexed without metadata are returned as ``{"doc_id": ...}`` only.
        """
        records = []
        for doc_id in self.doc_ids():
            points, _ = self.client.scroll(
                self.collection,
                scroll_filter=self._filter({"doc_id": doc_id}),
                limit=1,
                with_payload=[FILING_KEY],
                with_vectors=False,
            )
            filing = (points[0].payload or {}).get(FILING_KEY) if points else None
            records.append({**(filing or {}), "doc_id": doc_id})
        return records

    def scroll_chunks(self, batch_size: int = 1000) -> Iterable[dict[str, Any]]:
        offset = None
        while True:
            points, offset = self.client.scroll(
                self.collection,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            yield from (self._chunk(p.payload) for p in points)
            if offset is None:
                return

    @staticmethod
    def _chunk(payload: Mapping[str, Any]) -> dict[str, Any]:
        """Payload without the filing-catalog metadata, i.e. the original chunk dict."""
        return {k: v for k, v in payload.items() if k != FILING_KEY}

    @staticmethod
    def _filter(filters: Mapping[str, Any] | None) -> models.Filter | None:
        return to_qdrant_filter(merge_filters(filters))

    def get_chunks(self, chunk_ids: Sequence[str]) -> list[dict[str, Any]]:
        """Chunks by ``chunk_id`` in request order; duplicates collapsed, unknown ids skipped."""
        check_sequence("chunk_ids", chunk_ids)
        wanted = list(dict.fromkeys(chunk_ids))
        if not wanted:
            return []
        points = self.client.retrieve(
            self.collection, [point_id(c) for c in wanted], with_payload=True, with_vectors=False
        )
        found = {p.payload["chunk_id"]: self._chunk(p.payload) for p in points if p.payload}
        return [found[c] for c in wanted if c in found]

    @staticmethod
    def _query_args(
        query_text: str,
        query_vector: np.ndarray,
        limit: int,
        query_filter: models.Filter | None,
        *,
        mode: str,
        dense_k: int,
        sparse_k: int,
        rrf_k: int,
        dense_weight: float,
        sparse_weight: float,
    ) -> dict[str, Any]:
        """``query_points``/``query_points_groups`` arguments for one search ``mode``."""
        dense_vec = np.asarray(query_vector, dtype=np.float32).tolist()
        indices, values = encode_query(query_text)
        sparse_vec = models.SparseVector(indices=indices, values=values)
        if mode not in ("dense", "sparse", "hybrid"):
            raise ValueError(f"Unknown search mode: {mode}")
        if mode == "dense" or not indices:
            return {"query": dense_vec, "using": DENSE, "query_filter": query_filter}
        if mode == "sparse":
            return {"query": sparse_vec, "using": SPARSE, "query_filter": query_filter}
        return {
            "prefetch": [
                models.Prefetch(
                    query=dense_vec, using=DENSE, filter=query_filter, limit=max(limit, dense_k)
                ),
                models.Prefetch(
                    query=sparse_vec, using=SPARSE, filter=query_filter, limit=max(limit, sparse_k)
                ),
            ],
            "query": models.RrfQuery(
                rrf=models.Rrf(k=rrf_k, weights=[dense_weight, sparse_weight])
            ),
        }

    def search(
        self,
        query_text: str,
        query_vector: np.ndarray,
        k: int = 5,
        *,
        filters: Mapping[str, Any] | None = None,
        mode: str = "hybrid",
        dense_k: int = 50,
        sparse_k: int = 50,
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
    ) -> list[tuple[float, dict[str, Any]]]:
        """Return ``(score, chunk)`` pairs. ``hybrid`` fuses dense and sparse with weighted RRF.

        ``filters`` accepts scalars, value lists and numeric ranges (see ``vector_store.filters``).
        """
        args = self._query_args(
            query_text,
            query_vector,
            k,
            self._filter(filters),
            mode=mode,
            dense_k=dense_k,
            sparse_k=sparse_k,
            rrf_k=rrf_k,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
        )
        response = self.client.query_points(self.collection, limit=k, with_payload=True, **args)
        return [(float(p.score), self._chunk(p.payload)) for p in response.points]

    def search_grouped(
        self,
        query_text: str,
        query_vector: np.ndarray,
        group_by: str,
        *,
        groups: Sequence[Any] | None = None,
        group_size: int = 5,
        group_limit: int = 10,
        filters: Mapping[str, Any] | None = None,
        mode: str = "hybrid",
        dense_k: int = 50,
        sparse_k: int = 50,
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
    ) -> dict[Any, list[tuple[float, dict[str, Any]]]]:
        """Top ``group_size`` hits per value of ``group_by`` (``doc_id``, ``ticker``, ``fiscal_year``).

        With ``groups`` (known values), runs one filtered search per group so every group
        gets its own full candidate pool; every requested group is returned, possibly empty.
        Without ``groups``, one ``query_points_groups`` call finds the best ``group_limit``
        groups, ranked by their best hit. That is a single round trip, but hybrid fusion
        happens across all groups first, so a weaker group may get fewer hits.
        """
        merged = merge_filters(filters)
        check_group_by(group_by, merged, groups)
        if group_size < 1 or group_limit < 1:
            raise ValueError("group_size and group_limit must be at least 1")
        tuning = {
            "mode": mode,
            "dense_k": dense_k,
            "sparse_k": sparse_k,
            "rrf_k": rrf_k,
            "dense_weight": dense_weight,
            "sparse_weight": sparse_weight,
        }
        if groups is not None:
            return {
                group: self.search(
                    query_text,
                    query_vector,
                    group_size,
                    filters={**merged, group_by: group},
                    **tuning,
                )
                for group in dict.fromkeys(groups)
            }
        limit = group_size * group_limit
        args = self._query_args(query_text, query_vector, limit, to_qdrant_filter(merged), **tuning)
        response = self.client.query_points_groups(
            self.collection,
            group_by=group_by,
            group_size=group_size,
            limit=group_limit,
            with_payload=True,
            **args,
        )
        return {
            group.id: [(float(p.score), self._chunk(p.payload)) for p in group.hits]
            for group in response.groups
        }
