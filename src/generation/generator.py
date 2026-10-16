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

from datetime import datetime
from pathlib import Path
from typing import Any

from .config import GenerationConfig
from .context import format_context
from .prompts import SYSTEM_PROMPT, build_user_prompt
from .schemas import GeneratedAnswer, GenerationResult
from .llm_client import LLMClient
from .validation import parse_llm_response, validate_answer


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
        system_prompt: str = SYSTEM_PROMPT,
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
        self.system_prompt = system_prompt
        
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
        from reteriver.hybrid_retriever import HybridRetrieverWithReranker
        
        run_directory = self.config.resolve_run_directory(self.root)
        
        self._retriever = HybridRetrieverWithReranker(
            run_directory=run_directory,
            alpha=self.config.alpha,
            retrieve_k=self.config.retrieve_k,
            rerank_k=self.config.context_k,
            reranker_model=self.config.reranker_model,
            device=self.config.device,
        )
        
        print(f"Retriever ready with {len(self._retriever.chunks):,} chunks")
    
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
        if verbose:
            print(f"Retrieving top {context_k} chunks...")
        return self.retriever.search(question, k=context_k, doc_id=doc_id)
    
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
        
        if not chunks:
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
        
        if verbose:
            print(f"Context: {len(context):,} characters, {len(sources)} sources")
        
        # Step 2: Build prompt
        user_prompt = build_user_prompt(question, context)
        
        if verbose:
            print(f"Prompt: {len(self.system_prompt) + len(user_prompt):,} characters")
        
        # Step 3: Generate
        if verbose:
            print(f"Generating with {self.llm_client.model}...")
        
        llm_response = self.llm_client.generate(self.system_prompt, user_prompt)
        
        if verbose:
            print(f"Response: {llm_response.completion_tokens} tokens, {llm_response.latency_seconds:.2f}s")
        
        # Step 4: Parse and validate
        answer = parse_llm_response(llm_response.content)
        answer = validate_answer(answer, valid_source_ids)
        
        if verbose:
            if answer.is_valid:
                print("Validation: PASSED")
            else:
                print(f"Validation: FAILED - {answer.validation_errors}")
        
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
        )
    
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
                print(f"[{i}/{total}] {q.get('id', 'Q')}: {q['query'][:60]}...")
            
            try:
                gen_result = self.answer(
                    q["query"],
                    doc_id=q.get("doc_id"),
                    verbose=False,
                )
                
                results.append({
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
                })
                
            except Exception as e:
                if verbose:
                    print(f"  ERROR: {e}")
                results.append({
                    "id": q.get("id"),
                    "query": q["query"],
                    "doc_id": q.get("doc_id"),
                    "category": q.get("category"),
                    "difficulty": q.get("difficulty"),
                    "expected_answer": q.get("expected_answer"),
                    "status": "error",
                    "error": str(e),
                })
            
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
    import sys
    
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
        default=5,
        help="Number of context chunks (default: 5)",
    )
    parser.add_argument(
        "--provider",
        default="ollama",
        choices=["openai", "anthropic", "google", "ollama"],
        help="LLM provider (default: ollama)",
    )
    parser.add_argument(
        "--model",
        default="qwen2.5-7b-instruct:latest",
        help="Model name (default: qwen2.5-7b-instruct:latest)",
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
    print("Initializing generator...")
    config = GenerationConfig(
        provider=args.provider,
        model=args.model,
        context_k=args.context_k,
    )
    generator = AnswerGenerator(config)
    print()
    
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
            print(f"--- Validation ---")
            print(f"Valid: {result.answer.is_valid}")
            if result.answer.validation_errors:
                print(f"Errors: {result.answer.validation_errors}")
            print()
            print(f"--- Metadata ---")
            print(f"Model: {result.model}")
            print(f"Context chunks: {result.context_k}")
            print(f"Tokens: {result.prompt_tokens} prompt + {result.completion_tokens} completion")
            print(f"Latency: {result.latency_seconds:.2f}s")
            print()
            print(f"--- Sources ---")
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
