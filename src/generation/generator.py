"""
Answer Generator - Main Generation Pipeline.

Orchestrates retrieval, context formatting, LLM generation, and validation.

Usage:
    from generation import AnswerGenerator, GenerationConfig

    config = GenerationConfig()
    generator = AnswerGenerator(config)
    result = generator.answer("What was Apple's total net sales in fiscal 2025?", doc_id="AAPL_10-K_2025")

    # Isolated generation on fixed chunks (retriever is never loaded)
    result = generator.generate(question, chunks, doc_id="AAPL_10-K_2025")
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from config.logging import get_logger

from .config import GenerationConfig
from .context import format_context
from .llm_client import LLMClient
from .prompts import build_user_prompt, get_system_prompt
from .schemas import GeneratedAnswer, GenerationResult
from .validation import parse_llm_response, validate_answer

logger = get_logger(__name__)

logger = logging.getLogger(__name__)


class AnswerGenerator:
    """
    Main generation pipeline for SEC 10-K question answering.

    Combines:
    1. Hybrid retrieval with cross-encoder reranking
    2. Context formatting with source IDs
    3. Grounded LLM generation
    4. Citation validation

    Attributes:
        config: Generation configuration
        retriever: HybridRetrieverWithReranker instance
        llm_client: LLM client instance
        system_prompt: System prompt for generation
    """

    def __init__(
        self,
        config: GenerationConfig | None = None,
        retriever: Any | None = None,
        llm_client: LLMClient | None = None,
        system_prompt: str | None = None,
        root: Path | str | None = None,
    ):
        """
        Initialize the generator.

        Args:
            config: Generation configuration (uses defaults if None)
            retriever: Pre-initialized retriever (creates one if None)
            llm_client: Pre-initialized LLM client (creates one if None)
            system_prompt: System prompt to use
            root: Project root for resolving paths
        """
        self.config = config or GenerationConfig()
        self.system_prompt = system_prompt or get_system_prompt(self.config.prompt_version)

        # Resolve root
        if root is None:
            root = Path(__file__).resolve().parents[2]
        self.root = Path(root)

        # Retriever is loaded lazily so isolated generation eval (generate() with
        # fixed chunks) does not load embedding/reranker models.
        self._retriever = retriever

        # Initialize LLM client if not provided
        if llm_client is not None:
            self.llm_client = llm_client
        else:
            self._init_llm_client()

    @property
    def retriever(self) -> Any:
        """Lazily-initialized hybrid retriever with reranker."""
        if self._retriever is None:
            self._init_retriever()
        return self._retriever

    def _init_retriever(self) -> None:
        """Initialize the hybrid retriever with reranker."""
        from config.settings import settings

        if settings.VECTOR_STORE == "qdrant":
            from retriever.qdrant_retriever import QdrantRetrieverWithReranker

            self._retriever = QdrantRetrieverWithReranker(
                alpha=self.config.alpha,
                retrieve_k=self.config.retrieve_k,
                rerank_k=self.config.context_k,
                reranker_model=self.config.reranker_model,
                device=self.config.device,
            )
            logger.info(f"Retriever ready with {self._retriever.chunk_count:,} chunks (qdrant)")
            return

        from retriever.hybrid_retriever import HybridRetrieverWithReranker

        run_directory = self.config.resolve_run_directory(self.root)

        self._retriever = HybridRetrieverWithReranker(
            run_directory=run_directory,
            alpha=self.config.alpha,
            retrieve_k=self.config.retrieve_k,
            rerank_k=self.config.context_k,
            reranker_model=self.config.reranker_model,
            device=self.config.device,
        )

        logger.info(f"Retriever ready with {len(self._retriever.chunks):,} chunks")

    def _init_llm_client(self) -> None:
        """Initialize the LLM client from the current configuration."""
        self.llm_client = LLMClient(
            provider=self.config.provider,
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

    def answer(
        self,
        question: str,
        *,
        doc_id: str | None = None,
        context_k: int | None = None,
        verbose: bool = False,
    ) -> GenerationResult:
        """
        Answer a question using the full RAG pipeline: retrieve() then generate().

        Args:
            question: The question to answer
            doc_id: Optional document filter (e.g., "AAPL_10-K_2025")
            context_k: Number of chunks to use (default: config.context_k)
            verbose: Print debug info

        Returns:
            GenerationResult with answer, sources, and metadata
        """
        results = self.retrieve(question, doc_id=doc_id, context_k=context_k, verbose=verbose)
        return self.generate(question, results, doc_id=doc_id, verbose=verbose)

    def retrieve(
        self,
        question: str,
        *,
        doc_id: str | None = None,
        context_k: int | None = None,
        verbose: bool = False,
    ) -> list:
        """
        Retrieve the top context_k reranked chunks for a question.

        Args:
            question: The question to retrieve evidence for
            doc_id: Optional document filter (e.g., "AAPL_10-K_2025")
            context_k: Override for config.context_k (default: None uses config value)
            verbose: Print retrieval progress

        Returns:
            List of SearchResult objects from the hybrid retriever
        """
        context_k = context_k or self.config.context_k
        logger.info("=== RETRIEVAL ===")
        logger.debug(f"Question: {question}")
        logger.info(f"Document filter: {doc_id or 'none'} | context_k={context_k}")
        start = time.perf_counter()
        results = self.retriever.search(question, k=context_k, doc_id=doc_id)
        logger.info(f"Retrieved {len(results)} chunks in {time.perf_counter() - start:.2f}s")
        for i, result in enumerate(results, start=1):
            chunk = result.chunk
            logger.info(
                f"[S{i}] score={result.score:.4f} chunk_id={chunk.get('chunk_id')} "
                f"doc={chunk.get('doc_id')} type={chunk.get('type')} "
                f"section={chunk.get('section_title') or chunk.get('section')!r} "
                f"elements={chunk.get('element_orders', [])} table_id={chunk.get('table_id')} "
                f"chars={len(chunk.get('text', ''))}"
            )
            logger.debug(f"[S{i}] text:\n{chunk.get('text', '')}")
        return results

    def generate(
        self,
        question: str,
        chunks: list,
        *,
        doc_id: str | None = None,
        verbose: bool = False,
    ) -> GenerationResult:
        """
        Generate an answer from a fixed list of chunks (no retrieval).

        Used directly for isolated generation evaluation with frozen context.

        Steps:
        1. Format context with source IDs
        2. Build prompt
        3. Generate answer
        4. Parse and validate

        Args:
            question: The question to answer
            chunks: SearchResult objects or chunk dicts, in prompt order (S1, S2, ...)
            doc_id: Optional document ID, recorded in the result
            verbose: Print debug info
        """
        timestamp = datetime.now().isoformat()
        context_k = len(chunks)

        logger.info("=== GENERATION ===")
        logger.info(f"Generating answer | doc={doc_id or 'none'} | context chunks={context_k}")

        if not chunks:
            logger.warning("No relevant context found in the filing")
            return GenerationResult(
                question=question,
                doc_id=doc_id,
                answer=GeneratedAnswer(
                    answer=None,
                    explanation="No relevant context found in the filing.",
                    citations=[],
                    calculation=None,
                    insufficient_context=True,
                ),
                sources=[],
                model=self.llm_client.model,
                context_k=context_k,
                prompt_tokens=0,
                completion_tokens=0,
                latency_seconds=0.0,
                timestamp=timestamp,
            )

        # Step 1: Format context
        context, sources = format_context(chunks)
        valid_source_ids = [s["source_id"] for s in sources]

        logger.info(f"Context: {len(context):,} chars, sources={valid_source_ids}")

        # Step 2: Build prompt
        user_prompt = build_user_prompt(question, context)
        logger.info(
            f"Prompt: system={len(self.system_prompt):,} chars "
            f"(version={self.config.prompt_version}), user={len(user_prompt):,} chars"
        )
        logger.debug(f"System prompt:\n{self.system_prompt}")
        logger.debug(f"User prompt:\n{user_prompt}")

        # Step 3: Generate
        logger.info(
            f"Calling LLM: provider={self.llm_client.provider} model={self.llm_client.model} "
            f"temperature={self.llm_client.temperature} max_tokens={self.llm_client.max_tokens}"
        )
        llm_response, llm_attempts = self._generate_with_retries(user_prompt)
        logger.info(
            f"LLM done: prompt_tokens={llm_response.prompt_tokens} "
            f"completion_tokens={llm_response.completion_tokens} "
            f"latency={llm_response.latency_seconds:.2f}s"
        )
        logger.debug(f"Raw LLM response:\n{llm_response.content}")

        # Step 4: Parse and validate
        answer = parse_llm_response(llm_response.content)
        answer = validate_answer(answer, valid_source_ids)

        logger.debug(
            f"Final answer: {answer.answer!r} | explanation={answer.explanation!r} | "
            f"citations={answer.citations} | calculation={answer.calculation!r} | "
            f"insufficient_context={answer.insufficient_context}"
        )
        if answer.is_valid:
            logger.info("Validation: PASSED")
        else:
            logger.warning(f"Validation: FAILED - {answer.validation_errors}")
        logger.info("=== GENERATION COMPLETE ===")

        return GenerationResult(
            question=question,
            doc_id=doc_id,
            answer=answer,
            sources=sources,
            model=self.llm_client.model,
            context_k=context_k,
            prompt_tokens=llm_response.prompt_tokens,
            completion_tokens=llm_response.completion_tokens,
            latency_seconds=llm_response.latency_seconds,
            timestamp=timestamp,
            context=context,
            raw_response=llm_response.content,
            llm_attempts=llm_attempts,
        )

    def _generate_with_retries(self, user_prompt: str) -> tuple[Any, int]:
        """Call the LLM, retrying transient failures. Returns (response, attempts)."""
        from config.settings import settings

        max_retries = settings.LLM_MAX_RETRIES
        attempts = 0
        while True:
            attempts += 1
            try:
                return self.llm_client.generate(self.system_prompt, user_prompt), attempts
            except (ValueError, ImportError, TypeError) as error:
                error.llm_attempts = attempts  # type: ignore[union-attr]
                raise
            except Exception as error:
                if attempts > max_retries:
                    error.llm_attempts = attempts  # type: ignore[attr-defined]
                    raise
                delay = settings.LLM_RETRY_BACKOFF_SECONDS * 2 ** (attempts - 1)
                logger.warning(
                    f"LLM call failed ({type(error).__name__}: {error}); "
                    f"retry {attempts}/{max_retries} in {delay:.1f}s"
                )
                time.sleep(delay)

    def answer_batch(
        self,
        questions: list[dict[str, Any]],
        *,
        delay_seconds: float = 0.0,
        verbose: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Answer a batch of questions.

        Args:
            questions: List of question dicts with 'query' and optional 'doc_id'
            delay_seconds: Delay between questions (for rate limiting)
            verbose: Print progress

        Returns:
            List of result dictionaries
        """
        import time

        results = []
        total = len(questions)

        for i, q in enumerate(questions, 1):
            if verbose:
                logger.info(f"[{i}/{total}] {q.get('id', 'Q')}: {q['query'][:60]}...")

            try:
                gen_result = self.answer(
                    q["query"],
                    doc_id=q.get("doc_id"),
                    verbose=False,
                )

                results.append(
                    {
                        # Golden data
                        "id": q.get("id"),
                        "query": q["query"],
                        "doc_id": q.get("doc_id"),
                        "category": q.get("category"),
                        "difficulty": q.get("difficulty"),
                        "expected_answer": q.get("expected_answer"),
                        "relevant_elements": q.get("relevant", []),
                        # Generated data
                        "generated_answer": gen_result.answer.answer,
                        "explanation": gen_result.answer.explanation,
                        "citations": gen_result.answer.citations,
                        "calculation": gen_result.answer.calculation,
                        "insufficient_context": gen_result.answer.insufficient_context,
                        "is_valid": gen_result.answer.is_valid,
                        "validation_errors": gen_result.answer.validation_errors,
                        # Sources
                        "sources": gen_result.sources,
                        # Metadata
                        "model": gen_result.model,
                        "context_k": gen_result.context_k,
                        "prompt_tokens": gen_result.prompt_tokens,
                        "completion_tokens": gen_result.completion_tokens,
                        "latency_seconds": gen_result.latency_seconds,
                        "timestamp": gen_result.timestamp,
                        # Status
                        "status": "success",
                        "error": None,
                    }
                )

            except Exception as e:
                if verbose:
                    logger.error(f"ERROR: {e}")
                results.append(
                    {
                        "id": q.get("id"),
                        "query": q["query"],
                        "doc_id": q.get("doc_id"),
                        "category": q.get("category"),
                        "difficulty": q.get("difficulty"),
                        "expected_answer": q.get("expected_answer"),
                        "status": "error",
                        "error": str(e),
                    }
                )

            # Rate limiting
            if delay_seconds > 0 and i < total:
                time.sleep(delay_seconds)

        return results


def main():
    """
    CLI entry point for testing the generation pipeline.

    Usage:
        uv run python -m generation.generator "What was Apple's total net sales in fiscal 2025?" --doc-id AAPL_10-K_2025
        uv run python -m generation.generator --interactive
    """
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="SEC 10-K RAG Answer Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="Question to answer",
    )
    parser.add_argument(
        "--doc-id",
        help="Document filter (e.g., AAPL_10-K_2025)",
    )
    parser.add_argument(
        "--context-k",
        type=int,
        default=None,
        help="Number of context chunks (default: CONTEXT_K from .env)",
    )
    parser.add_argument(
        "--provider",
        default=None,
        choices=["openai", "anthropic", "google", "ollama", "huggingface"],
        help="LLM provider (default: resolved from .env execution mode / LLM_PROVIDER)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name (default: provider default from .env)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive mode - ask multiple questions",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.question and not args.interactive:
        parser.error("Either provide a question or use --interactive mode")

    # Initialize generator
    logger.info("Initializing generator...")
    overrides = {"provider": args.provider, "model": args.model, "context_k": args.context_k}
    config = GenerationConfig(
        **{key: value for key, value in overrides.items() if value is not None}
    )
    generator = AnswerGenerator(config)

    def process_question(question: str, doc_id: str | None) -> None:
        """Process a single question and display results as text or JSON."""
        result = generator.answer(
            question,
            doc_id=doc_id,
            verbose=args.verbose,
        )

        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print("=" * 70)
            print(f"QUESTION: {result.question}")
            if result.doc_id:
                print(f"DOC: {result.doc_id}")
            print("=" * 70)
            print()
            print(f"ANSWER: {result.answer.answer}")
            print(f"EXPLANATION: {result.answer.explanation}")
            print(f"CITATIONS: {result.answer.citations}")
            if result.answer.calculation:
                print(f"CALCULATION: {result.answer.calculation}")
            print(f"INSUFFICIENT CONTEXT: {result.answer.insufficient_context}")
            print()
            print("--- Validation ---")
            print(f"Valid: {result.answer.is_valid}")
            if result.answer.validation_errors:
                print(f"Errors: {result.answer.validation_errors}")
            print()
            print("--- Metadata ---")
            print(f"Model: {result.model}")
            print(f"Context chunks: {result.context_k}")
            print(f"Tokens: {result.prompt_tokens} prompt + {result.completion_tokens} completion")
            print(f"Latency: {result.latency_seconds:.2f}s")
            print()
            print("--- Sources ---")
            for s in result.sources:
                print(f"  [{s['source_id']}] {s['doc_id']} | {s['type']} | score={s['score']:.4f}")
                print(f"       elements={s['element_orders']}")

    if args.interactive:
        print("Interactive mode. Type 'quit' or 'exit' to stop.")
        print("Format: <question> [--doc <doc_id>]")
        print()

        while True:
            try:
                user_input = input("Question> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break

            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                print("Exiting.")
                break

            # Parse doc_id from input
            doc_id = args.doc_id
            if "--doc" in user_input:
                parts = user_input.split("--doc")
                user_input = parts[0].strip()
                if len(parts) > 1:
                    doc_id = parts[1].strip().split()[0]

            process_question(user_input, doc_id)
            print()
    else:
        process_question(args.question, args.doc_id)


if __name__ == "__main__":
    main()
