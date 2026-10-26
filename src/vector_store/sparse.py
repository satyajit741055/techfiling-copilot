"""BM25-style sparse vectors for Qdrant's server-side lexical search.

Tokens are hashed to stable 32-bit ids, so no corpus-wide vocabulary or fit step is
needed and new documents can be indexed incrementally. Documents carry the BM25 term
frequency component; the inverse-document-frequency part is applied by Qdrant
(``Modifier.IDF``) from live collection statistics, so it stays correct as the corpus grows.
"""

from __future__ import annotations

import re
import zlib
from collections import Counter

K1 = 1.5
B = 0.75
AVG_DOC_TOKENS = 256.0
TOKEN_PATTERN = re.compile(r"\b[a-z0-9]+\b")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens (keeps numbers and tickers, no stopword removal)."""
    return TOKEN_PATTERN.findall(text.lower())


def _token_id(token: str) -> int:
    return zlib.crc32(token.encode("utf-8"))


def encode_document(text: str) -> tuple[list[int], list[float]]:
    """Sparse BM25 term-frequency vector (indices, values) for an indexed chunk."""
    tokens = tokenize(text)
    length_norm = 1 - B + B * len(tokens) / AVG_DOC_TOKENS
    weights: dict[int, float] = {}
    for token, tf in Counter(tokens).items():
        index = _token_id(token)
        weights[index] = weights.get(index, 0.0) + tf * (K1 + 1) / (tf + K1 * length_norm)
    return list(weights), list(weights.values())


def encode_query(text: str) -> tuple[list[int], list[float]]:
    """Sparse query vector: one unit weight per unique token."""
    indices = list(dict.fromkeys(_token_id(token) for token in tokenize(text)))
    return indices, [1.0] * len(indices)
