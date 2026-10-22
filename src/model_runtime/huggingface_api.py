"""Hugging Face hosted-inference embedding and reranking backends."""

from __future__ import annotations

from collections.abc import Sequence
from functools import cached_property
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from .types import BGE_QUERY_PREFIX, l2_normalize

API_BATCH_SIZE = 32


class HuggingFaceApiEmbedder:
    """Bi-encoder embeddings from the Hugging Face feature-extraction API."""

    backend = "huggingface_api"

    def __init__(
        self,
        model_name: str,
        *,
        api_key: str | None,
        provider: str = "hf-inference",
        timeout: float = 120.0,
        cache_dir: str | Path | None = None,
        query_prefix: str = BGE_QUERY_PREFIX,
        client: Any | None = None,
    ) -> None:
        if client is None:
            from huggingface_hub import InferenceClient

            client = InferenceClient(provider=provider, api_key=api_key, timeout=timeout)
        self.client = client
        self.model_name = model_name
        self.query_prefix = query_prefix
        self.cache_dir = cache_dir

    def _embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.asarray(
            self.client.feature_extraction(texts, model=self.model_name, normalize=True),
            dtype=np.float32,
        )
        if vectors.ndim == 1:
            vectors = vectors[None, :]
        if vectors.ndim != 2 or len(vectors) != len(texts):
            raise ValueError(
                f"{self.model_name} returned embeddings of shape {vectors.shape} for "
                f"{len(texts)} texts; expected one pooled vector per text"
            )
        return l2_normalize(vectors)

    @cached_property
    def dimension(self) -> int:
        return int(self._embed(["dimension probe"]).shape[1])

    @cached_property
    def tokenizer(self) -> Any:
        from transformers import AutoTokenizer

        cache_dir = str(self.cache_dir) if self.cache_dir else None
        return AutoTokenizer.from_pretrained(self.model_name, cache_dir=cache_dir)

    @property
    def max_seq_length(self) -> int:
        return int(self.tokenizer.model_max_length)

    def encode_query(self, query: str) -> np.ndarray:
        return self._embed([self.query_prefix + query])[0]

    def encode_documents(self, texts: Sequence[str], batch_size: int = API_BATCH_SIZE) -> np.ndarray:
        texts = list(texts)
        step = max(1, min(batch_size, API_BATCH_SIZE))
        batches = [self._embed(texts[i:i + step]) for i in range(0, len(texts), step)]
        return np.vstack(batches) if batches else np.empty((0, 0), dtype=np.float32)


class HuggingFaceApiReranker:
    """Cross-encoder scores from a Hugging Face endpoint.

    ``url`` may contain ``{model}``. URLs ending in ``/rerank`` are called with the
    Text Embeddings Inference format (``{"query", "texts"}``); all other URLs use the
    pair-classification format (``{"inputs": [{"text", "text_pair"}, ...]}``).
    """

    backend = "huggingface_api"

    def __init__(
        self,
        model_name: str,
        *,
        api_key: str | None,
        url: str,
        timeout: float = 120.0,
        client: httpx.Client | None = None,
    ) -> None:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.client = client or httpx.Client(headers=headers, timeout=timeout)
        self.model_name = model_name
        self.url = url.format(model=model_name)
        self.tei_format = self.url.rstrip("/").endswith("/rerank")

    def _post(self, payload: dict[str, Any]) -> Any:
        response = self.client.post(self.url, json=payload)
        if response.is_error:
            raise RuntimeError(
                f"Reranker request to {self.url} failed ({response.status_code}): "
                f"{response.text[:500]}"
            )
        return response.json()

    def _score_tei(self, query: str, documents: list[str]) -> np.ndarray:
        ranked = self._post({"query": query, "texts": documents, "truncate": True})
        scores = np.empty(len(documents), dtype=np.float32)
        for item in ranked:
            scores[item["index"]] = item["score"]
        return scores

    def _score_pairs(self, query: str, documents: list[str]) -> np.ndarray:
        result = self._post({"inputs": [{"text": query, "text_pair": text} for text in documents]})
        if isinstance(result, dict):
            result = [result]
        scores = [item[0]["score"] if isinstance(item, list) else item["score"] for item in result]
        if len(scores) != len(documents):
            raise ValueError(f"Reranker returned {len(scores)} scores for {len(documents)} documents")
        return np.asarray(scores, dtype=np.float32)

    def score(self, query: str, documents: Sequence[str]) -> np.ndarray:
        documents = list(documents)
        if not documents:
            return np.empty(0, dtype=np.float32)
        score_batch = self._score_tei if self.tei_format else self._score_pairs
        return np.concatenate([
            score_batch(query, documents[i:i + API_BATCH_SIZE])
            for i in range(0, len(documents), API_BATCH_SIZE)
        ])
