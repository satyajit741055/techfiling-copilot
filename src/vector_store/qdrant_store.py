"""Qdrant-backed chunk index: dense + sparse (BM25) vectors, payload filters, server-side RRF."""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
from qdrant_client import QdrantClient, models

from config.logging import get_logger
from vector_store.sparse import encode_document, encode_query

logger = get_logger(__name__)

DENSE = "dense"
SPARSE = "bm25"
FILTER_FIELDS = ("doc_id", "section", "type")
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
    ) -> None:
        self.collection = collection
        if client is not None:
            self.client = client
        elif url == ":memory:":
            self.client = QdrantClient(location=":memory:")
        else:
            self.client = QdrantClient(url=url, api_key=api_key, timeout=int(timeout))

    @classmethod
    def from_settings(cls, settings: Any | None = None) -> QdrantChunkStore:
        if settings is None:
            from config.settings import settings as global_settings

            settings = global_settings
        return cls(
            settings.QDRANT_URL,
            settings.QDRANT_COLLECTION,
            api_key=settings.QDRANT_API_KEY,
            timeout=settings.QDRANT_TIMEOUT,
        )

    def exists(self) -> bool:
        return self.client.collection_exists(self.collection)

    def count(self) -> int:
        return self.client.count(self.collection, exact=True).count if self.exists() else 0

    def index_metadata(self) -> dict[str, Any]:
        """Embedding model metadata stored on the collection (manifest-shaped)."""
        info = self.client.get_collection(self.collection)
        return {"embedding_model": (info.config.metadata or {}).get("embedding_model", {})}

    def dimension(self) -> int:
        params = self.client.get_collection(self.collection).config.params.vectors
        return int(params[DENSE].size)

    def ensure_collection(
        self, dimension: int, embedding_model: Mapping[str, Any] | None = None
    ) -> None:
        """Create the collection and payload indexes if missing; verify the dimension otherwise."""
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
            return
        self.client.create_collection(
            self.collection,
            vectors_config={
                DENSE: models.VectorParams(size=dimension, distance=models.Distance.COSINE)
            },
            sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
            metadata={"embedding_model": dict(embedding_model or {})},
        )
        for field in FILTER_FIELDS:
            self.client.create_payload_index(
                self.collection, field, models.PayloadSchemaType.KEYWORD
            )
        logger.info(f"Created Qdrant collection {self.collection!r} (dim={dimension})")

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
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            points = []
            for chunk, vector in zip(batch, vectors[start : start + batch_size]):
                indices, values = encode_document(chunk.get("embed_text") or chunk.get("text", ""))
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
    ) -> int:
        """Ensure the collection, upsert the chunks, then drop each document's stale chunks.

        Upserting first means a failed run never leaves a document without chunks.
        """
        self.ensure_collection(int(vectors.shape[1]), embedding_model)
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
                            must_not=[models.HasIdCondition(has_id=ids)],
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
        if not filters:
            return None
        return models.Filter(
            must=[
                models.FieldCondition(key=key, match=models.MatchValue(value=value))
                for key, value in filters.items()
            ]
        )

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
        """Return ``(score, chunk)`` pairs. ``hybrid`` fuses dense and sparse with weighted RRF."""
        query_filter = self._filter(filters)
        dense_vec = np.asarray(query_vector, dtype=np.float32).tolist()
        indices, values = encode_query(query_text)
        sparse_vec = models.SparseVector(indices=indices, values=values)

        if mode == "dense" or not indices:
            response = self.client.query_points(
                self.collection,
                query=dense_vec,
                using=DENSE,
                query_filter=query_filter,
                limit=k,
                with_payload=True,
            )
        elif mode == "sparse":
            response = self.client.query_points(
                self.collection,
                query=sparse_vec,
                using=SPARSE,
                query_filter=query_filter,
                limit=k,
                with_payload=True,
            )
        elif mode == "hybrid":
            response = self.client.query_points(
                self.collection,
                prefetch=[
                    models.Prefetch(
                        query=dense_vec, using=DENSE, filter=query_filter, limit=max(k, dense_k)
                    ),
                    models.Prefetch(
                        query=sparse_vec, using=SPARSE, filter=query_filter, limit=max(k, sparse_k)
                    ),
                ],
                query=models.RrfQuery(
                    rrf=models.Rrf(k=rrf_k, weights=[dense_weight, sparse_weight])
                ),
                limit=k,
                with_payload=True,
            )
        else:
            raise ValueError(f"Unknown search mode: {mode}")
        return [(float(p.score), self._chunk(p.payload)) for p in response.points]
