"""
API Dependencies.

Handles initialization and dependency injection for the RAG pipeline.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Add src to path for imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from api.config import api_settings
from api.services.rag_service import RAGService
from persistence import HistoryStore, NullHistoryStore, create_history_store

if TYPE_CHECKING:
    from agent.service import AgentService
    from facts.store import FactsStore
    from generation import AnswerGenerator
    from persistence import ThreadedPostgresSaver

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# MODULE-LEVEL STATE
# =============================================================================

_generator: "AnswerGenerator | None" = None
_rag_service: RAGService | None = None
_filings: list[dict] | None = None
_history_store: HistoryStore = NullHistoryStore()
_agent_service: "AgentService | None" = None
_checkpointer: "ThreadedPostgresSaver | None" = None


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
    filings: list[dict] = []

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


def init_history_store() -> None:
    """Create the history store for HISTORY_STORE; the database is reached lazily."""
    global _history_store
    from config.settings import settings

    _history_store = create_history_store(settings)
    _history_store.open()
    if not _history_store.enabled:
        logger.info("History store: DISABLED (HISTORY_STORE=none)")
    elif api_settings.supabase_auth_enabled:
        logger.info("History store: Postgres (owner = signed-in Supabase user)")
    elif api_settings.HISTORY_DEV_USER_ID:
        logger.info("History store: Postgres (owner = HISTORY_DEV_USER_ID, auth disabled)")
    else:
        logger.warning(
            "History store: Postgres, but auth is disabled and HISTORY_DEV_USER_ID is unset, "
            "so nothing is saved and history endpoints return 404"
        )


def shutdown_history_store() -> None:
    global _history_store
    _history_store.close()
    _history_store = NullHistoryStore()


def _load_facts_store(settings: Any) -> "FactsStore | None":
    """FACTS_STORE=postgres: the financial_facts table as an in-memory store.

    Fails open: any problem logs a WARNING and returns None, so research chat answers
    numbers from retrieval only (as with FACTS_STORE=off) instead of being disabled.
    """
    if settings.FACTS_STORE != "postgres":
        return None
    from facts.store import FactsStore
    from persistence.facts import read_facts
    from persistence.history import _error_summary

    try:
        records, load = read_facts(settings.HISTORY_DB_URL or "")
    except Exception as e:
        logger.warning(f"Facts store DISABLED: {_error_summary(e)}")
        return None
    if load is None or not records:
        logger.warning("Facts store DISABLED: financial_facts is empty (run scripts/load_facts.py)")
        return None
    if load.collection != settings.QDRANT_COLLECTION:
        logger.warning(
            f"Facts store DISABLED: facts were linked to collection {load.collection!r}, "
            f"the API uses {settings.QDRANT_COLLECTION!r} (re-run scripts/load_facts.py)"
        )
        return None
    return FactsStore(records, load.collection)


def init_agent() -> None:
    """
    Build the research agent for /chat on the generator's already-loaded retriever and
    reranker (no second model load). Run after init_generator and init_history_store.

    A failure (e.g. VECTOR_STORE is not qdrant, or AGENT_CHECKPOINTER=postgres and the
    database is unreachable) disables /chat with a 503 instead of stopping the API;
    /questions is unaffected. There is no fallback to the in-memory checkpointer: it would
    silently lose pauses and follow-up context on restart.
    """
    global _agent_service, _checkpointer
    from agent.graph import checkpoint_serializer
    from agent.service import AgentService
    from config.settings import settings
    from observability import QueryTracker
    from persistence import create_checkpointer

    try:
        if settings.AGENT_CHECKPOINTER == "postgres":
            # Settings validation guarantees the URL when AGENT_CHECKPOINTER=postgres.
            _checkpointer = create_checkpointer(
                settings.HISTORY_DB_URL or "", checkpoint_serializer()
            )
            _checkpointer.open()
        facts = _load_facts_store(settings)
        _agent_service = AgentService.from_generator(
            get_generator(),
            get_filings(),
            history=_history_store,
            writer=QueryTracker.from_settings().writer,
            checkpointer=_checkpointer,
            facts=facts,
        )
    except Exception as e:
        shutdown_agent()
        logger.error(f"Research chat DISABLED: {type(e).__name__}: {e}")
        return
    facts_text = f"postgres ({len(facts):,} facts, {facts.linked:,} citable)" if facts else "off"
    logger.info(
        f"Research chat ready: {_agent_service.provider}/{_agent_service.model}, "
        f"{len(_agent_service.catalog.filings())} filings, "
        f"checkpointer={settings.AGENT_CHECKPOINTER}, facts={facts_text}"
    )


def shutdown_agent() -> None:
    global _agent_service, _checkpointer
    _agent_service = None
    if _checkpointer is not None:
        _checkpointer.close()
        _checkpointer = None


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


def get_history_store() -> HistoryStore:
    """FastAPI dependency: the history store (the null store when history is off)."""
    return _history_store


def get_agent_service() -> "AgentService | None":
    """FastAPI dependency: the research agent, or None when /chat is unavailable."""
    return _agent_service


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
