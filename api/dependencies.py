"""
API Dependencies.

Handles initialization and dependency injection for the RAG pipeline.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

# Add src to path for imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from api.config import api_settings
from api.services.rag_service import RAGService

if TYPE_CHECKING:
    from generation import AnswerGenerator

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# MODULE-LEVEL STATE
# =============================================================================

_generator: "AnswerGenerator | None" = None
_rag_service: RAGService | None = None
_filings: list[dict] | None = None


# =============================================================================
# INITIALIZATION
# =============================================================================


def _load_filings_from_disk() -> list[dict]:
    """
    Load filing metadata from data/parsed/*.document.json files.

    Returns:
        List of filing metadata dictionaries
    """
    parsed_dir = api_settings.PARSED_DATA_DIR
    filings = []

    if not parsed_dir.exists():
        logger.warning(f"Parsed data directory not found: {parsed_dir}")
        return filings

    for doc_file in sorted(parsed_dir.glob("*.document.json")):
        try:
            with open(doc_file, "r", encoding="utf-8") as f:
                doc_meta = json.load(f)
                filings.append(doc_meta)
        except Exception as e:
            logger.warning(f"Failed to load {doc_file}: {e}")

    return filings


def _load_filings() -> list[dict]:
    """Filing catalog from Qdrant when it is the vector store, otherwise from disk."""
    from config.settings import settings

    if settings.VECTOR_STORE == "qdrant":
        from vector_store import QdrantChunkStore

        store = QdrantChunkStore.from_settings()
        try:
            filings = store.list_filings()
        finally:
            store.client.close()
        missing = [f["doc_id"] for f in filings if "company" not in f]
        if filings and not missing:
            return filings
        logger.warning(
            f"Filing metadata missing in Qdrant for {missing or 'all documents'}; filling from "
            "data/parsed. Re-run scripts/migrate_npy_to_qdrant.py to store it with the chunks."
        )
        on_disk = {f["doc_id"]: f for f in _load_filings_from_disk()}
        merged = [f if "company" in f else on_disk.get(f["doc_id"]) for f in filings]
        return [f for f in merged if f] or list(on_disk.values())
    return _load_filings_from_disk()


def init_generator() -> None:
    """
    Initialize the AnswerGenerator and load filings.

    Called once at startup via FastAPI lifespan.
    This loads embedding models, reranker, and BM25 index (~5-10 seconds).
    """
    global _generator, _rag_service, _filings

    # Load filings metadata
    logger.info("Loading filings metadata...")
    _filings = _load_filings()
    logger.info(f"  Loaded {len(_filings)} filings: {[f['doc_id'] for f in _filings]}")

    # Initialize the generator (this is the slow part)
    logger.info("Initializing AnswerGenerator...")
    from generation import AnswerGenerator, GenerationConfig

    config = GenerationConfig()
    _generator = AnswerGenerator(config, root=ROOT)

    # Create the service wrapper
    _rag_service = RAGService(_generator)

    logger.info(f"  Model: {_rag_service.model_name}")
    logger.info("Generator ready!")


def shutdown_generator() -> None:
    """
    Cleanup on shutdown.

    Called via FastAPI lifespan on app shutdown.
    """
    global _generator, _rag_service, _filings
    _generator = None
    _rag_service = None
    _filings = None
    logger.info("Generator shutdown complete")


# =============================================================================
# DEPENDENCY INJECTION
# =============================================================================


def get_generator() -> "AnswerGenerator":
    """
    FastAPI dependency to get the initialized AnswerGenerator.

    Raises:
        RuntimeError: If generator not initialized
    """
    if _generator is None:
        raise RuntimeError("Generator not initialized. Call init_generator() first.")
    return _generator


def get_rag_service() -> RAGService:
    """
    FastAPI dependency to get the RAG service.

    Raises:
        RuntimeError: If service not initialized
    """
    if _rag_service is None:
        raise RuntimeError("RAG service not initialized. Call init_generator() first.")
    return _rag_service


def get_filings() -> list[dict]:
    """
    FastAPI dependency to get the list of available filings.

    Raises:
        RuntimeError: If filings not loaded
    """
    if _filings is None:
        raise RuntimeError("Filings not loaded. Call init_generator() first.")
    return _filings


def get_valid_doc_ids() -> set[str]:
    """
    Get the set of valid document IDs.

    Returns:
        Set of valid doc_id strings
    """
    filings = get_filings()
    return {f["doc_id"] for f in filings}
