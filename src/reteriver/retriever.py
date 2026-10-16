"""Dense semantic retrieval over embedded SEC filing chunks."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from sentence_transformers import SentenceTransformer

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS_DIRECTORY = ROOT / "data" / "chunks"
DEFAULT_MODEL_CACHE_DIRECTORY = ROOT / "models" / "sentence_transformers"


class SearchResult(NamedTuple):
    """A retrieved chunk paired with its cosine-similarity score."""

    score: float
    chunk: dict[str, Any]


class DenseRetriever:
    """Search normalized SEC filing embeddings using a BGE query model."""

    def __init__(
        self,
        run_directory: str | Path | None = None,
        *,
        model: SentenceTransformer | None = None,
        device: str | None = None,
        model_cache_directory: str | Path = DEFAULT_MODEL_CACHE_DIRECTORY,
    ) -> None:
        """Load a retrieval run and its matching sentence-transformers model.

        Args:
            run_directory: Directory containing chunks, embeddings, and the run
                manifest. When omitted, the latest complete run is selected.
            model: Optional preloaded model for reuse or testing.
            device: Explicit model device, such as ``cuda`` or ``cpu``.
            model_cache_directory: Directory used to cache model files.

        Raises:
            FileNotFoundError: If no complete default retrieval run exists.
            NotADirectoryError: If the requested run directory does not exist.
            ValueError: If chunks, embeddings, manifest, and model are inconsistent.
        """
        self.run_directory = self._resolve_run_directory(run_directory)
        self.manifest = self._load_manifest()
        self.chunks = self._load_chunks()
        self.embeddings = np.load(
            self.run_directory / "embeddings_gpu_base.npy",
            mmap_mode="r",
        )
        self.model_name = self.manifest["embedding_model"]["name"]
        self.model = model or SentenceTransformer(
            self.model_name, device=device, cache_folder=str(model_cache_directory)
        )
        self._validate_artifacts()


    @staticmethod
    def _resolve_run_directory(run_directory: str | Path | None) -> Path:
        """Resolve an explicit run directory or select the latest complete run."""
        if run_directory is not None:
            resolved = Path(run_directory).expanduser().resolve()
        else:
            candidates = sorted(
                path
                for path in DEFAULT_RUNS_DIRECTORY.glob("*")
                if path.is_dir()
                and (path / "run_manifest.json").is_file()
                and (path / "embeddings_gpu_base.npy").is_file()
            )
            if not candidates:
                raise FileNotFoundError(
                    f"No complete retrieval run found in {DEFAULT_RUNS_DIRECTORY}"
                )
            resolved = candidates[-1]
        if not resolved.is_dir():
            raise NotADirectoryError(f"Retrieval run does not exist: {resolved}")
        return resolved

    def _load_manifest(self) -> dict[str, Any]:
        """Load the selected run's JSON manifest."""
        manifest_path = self.run_directory / "run_manifest.json"
        with manifest_path.open(encoding="utf-8") as file:
            return json.load(file)

    def _load_chunks(self) -> list[dict[str, Any]]:
        """Load all run chunk files in deterministic filename order."""
        chunks: list[dict[str, Any]] = []
        for path in sorted(self.run_directory.glob("*.chunks.jsonl")):
            with path.open(encoding="utf-8") as file:
                chunks.extend(json.loads(line) for line in file if line.strip())
        return chunks

    def _validate_artifacts(self) -> None:
        """Validate chunk counts and embedding dimensions across run artifacts.

        Raises:
            ValueError: If the chunks, embeddings, manifest, or model disagree.
        """
        if self.embeddings.ndim != 2:
            raise ValueError(
                f"Expected a 2D embedding matrix, got shape {self.embeddings.shape}"
            )
        if len(self.chunks) != len(self.embeddings):
            raise ValueError(
                "Chunk and embedding counts differ: "
                f"{len(self.chunks)} chunks versus {len(self.embeddings)} embeddings"
            )
        expected_chunks = self.manifest.get("overall", {}).get("chunks")
        if expected_chunks is not None and expected_chunks != len(self.chunks):
            raise ValueError(
                f"Manifest declares {expected_chunks} chunks, loaded {len(self.chunks)}"
            )
        expected_dimension = self.manifest["embedding_model"].get("dimension")
        if expected_dimension is not None and expected_dimension != self.embeddings.shape[1]:
            raise ValueError(
                "Embedding dimension differs from manifest: "
                f"{self.embeddings.shape[1]} versus {expected_dimension}"
            )
        model_dimension = self.model.get_embedding_dimension()
        if model_dimension is not None and model_dimension != self.embeddings.shape[1]:
            raise ValueError(
                "Loaded model and stored embedding dimensions differ: "
                f"{model_dimension} versus {self.embeddings.shape[1]}"
            )

    def search(
        self,
        query: str,
        k: int = 5,
        *,
        doc_id: str | None = None,
        section: str | None = None,
        type_: str | None = None,
        filters: Mapping[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Return the highest-scoring chunks matching a semantic query and filters.

        Args:
            query: Natural-language search query.
            k: Maximum number of results to return.
            doc_id: Optional exact document-ID filter.
            section: Optional exact filing-section filter.
            type_: Optional exact chunk-type filter.
            filters: Additional exact-match chunk metadata filters. Explicit
                ``doc_id``, ``section``, and ``type_`` arguments take precedence.

        Returns:
            Results ordered by descending cosine-similarity score.

        Raises:
            ValueError: If ``query`` is empty or ``k`` is less than one.
        """
        if not query.strip():
            raise ValueError("Query must not be empty")
        if k < 1:
            raise ValueError("k must be at least 1")

        query_embedding = self.model.encode(
            BGE_QUERY_PREFIX + query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")
        scores = self.embeddings @ query_embedding
        matches = self._matching_indices(
            doc_id=doc_id,
            section=section,
            type_=type_,
            filters=filters,
        )
        if not matches:
            return []

        match_indices = np.asarray(matches, dtype=np.int64)
        match_scores = scores[match_indices]
        result_count = min(k, len(match_indices))
        if result_count == len(match_indices):
            ranked_positions = np.argsort(-match_scores)
        else:
            candidates = np.argpartition(-match_scores, result_count - 1)[:result_count]
            ranked_positions = candidates[np.argsort(-match_scores[candidates])]
        return [
            SearchResult(float(match_scores[position]), self.chunks[match_indices[position]])
            for position in ranked_positions[:result_count]
        ]

    def retrieve(
        self,
        query: str,
        k: int = 5,
        *,
        doc_id: str | None = None,
        section: str | None = None,
        type_: str | None = None,
        filters: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Return search results flattened into score-enriched chunk dictionaries.

        Arguments and validation are identical to :meth:`search`.
        """
        return [
            {"score": result.score, **result.chunk}
            for result in self.search(
                query, k=k, doc_id=doc_id, section=section, type_=type_, filters=filters
            )
        ]

    def _matching_indices(
        self,
        *,
        doc_id: str | None,
        section: str | None,
        type_: str | None,
        filters: Mapping[str, Any] | None,
    ) -> list[int]:
        """Return indices of chunks satisfying all requested metadata filters."""
        requested = dict(filters or {})
        if doc_id is not None:
            requested["doc_id"] = doc_id
        if section is not None:
            requested["section"] = section
        if type_ is not None:
            requested["type"] = type_
        return [
            index
            for index, chunk in enumerate(self.chunks)
            if all(chunk.get(key) == value for key, value in requested.items())
        ]

    def iter_chunks(self) -> Iterator[dict[str, Any]]:
        """Iterate over all loaded chunks in embedding-row order."""
        return iter(self.chunks)

    
def show(results: list[SearchResult], width: int = 350) -> None:
    """Print ranked search results with truncated single-line text previews.

    Args:
        results: Ranked search results to display.
        width: Maximum number of chunk-text characters shown per result.
    """
    for score, chunk in results:
        print(f"{score:.3f} | {chunk['breadcrumb']} [{chunk['type']}]")
        print("      ", chunk["text"][:width].replace("\n", " "))
        print(f"element Order {chunk["element_orders"]}")
        print("="*80)


def main() -> None:
    """Run the command-line interface for querying a retrieval run."""
    parser = argparse.ArgumentParser(description="Query the dense 10-K retriever.")
    parser.add_argument("query", nargs="?", default="What was Apple's total net sales in 2025?")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--doc-id")
    parser.add_argument("--section")
    parser.add_argument("--type", dest="type_")
    parser.add_argument("--run-directory")
    parser.add_argument("--device")
    parser.add_argument("--model-cache-dir", default=DEFAULT_MODEL_CACHE_DIRECTORY)
    parser.add_argument("--width", type=int, default=900)
    args = parser.parse_args()

    retriever = DenseRetriever(
        args.run_directory, device=args.device, model_cache_directory=args.model_cache_dir
    )
    show(
        retriever.search(
            args.query, k=args.k, doc_id=args.doc_id, section=args.section, type_=args.type_
        ),
        args.width,
    )


if __name__ == "__main__":
    main()
    


