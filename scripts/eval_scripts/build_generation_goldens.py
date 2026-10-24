#!/usr/bin/env python3
"""
Build the generation golden dataset from the retrieval golden dataset and fixed chunks.

This script creates goldens/golden_generation_dataset.jsonl for isolated generator evaluation.
Each case contains the question, structured ground truth, and exact frozen chunk dictionaries.

Usage:
    uv run python scripts/eval_scripts/build_generation_goldens.py

Output:
    - goldens/golden_generation_dataset.jsonl: All generated cases (needs_review)
    - goldens/generation_dataset_review.jsonl: Ambiguity and manual review items
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# Fixed chunk run for reproducibility
CHUNK_RUN = "2026-10-16_17-03-28"

# Random seed for deterministic distractor selection and ordering
RANDOM_SEED = 42

# Target number of chunks for oracle_plus_distractors variant
TARGET_CONTEXT_SIZE = 5


@dataclass
class ReviewItem:
    """An item flagged for manual review."""
    source_qid: str
    issue_type: str
    description: str
    details: dict = field(default_factory=dict)


def sha256_text(text: str) -> str:
    """Compute SHA-256 hash of UTF-8 encoded text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_retrieval_goldens(path: Path) -> list[dict]:
    """Load the retrieval golden dataset."""
    questions = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    return questions


def load_chunks(chunk_dir: Path) -> tuple[dict[str, list[dict]], dict[str, dict[int, list[dict]]]]:
    """
    Load all chunks from the fixed run.

    Returns:
        chunks_by_doc: {doc_id: [chunk, ...]}
        element_to_chunks: {doc_id: {element_order: [chunk, ...]}}
    """
    chunks_by_doc = {}
    element_to_chunks = {}

    for chunk_file in chunk_dir.glob("*.chunks.jsonl"):
        doc_id = chunk_file.stem.replace(".chunks", "")
        chunks_by_doc[doc_id] = []
        element_to_chunks[doc_id] = defaultdict(list)

        with open(chunk_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunk = json.loads(line)
                    chunks_by_doc[doc_id].append(chunk)
                    for eo in chunk.get("element_orders", []):
                        element_to_chunks[doc_id][eo].append(chunk)

    return chunks_by_doc, element_to_chunks


def extract_fiscal_period(text: str) -> str | None:
    """Extract fiscal period from text (e.g., 'fiscal 2025', 'FY2026', '2025')."""
    patterns = [
        r"fiscal\s*(?:year\s*)?(\d{4})",
        r"FY\s*(\d{4})",
        r"(?:in|for)\s+(\d{4})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return f"FY{match.group(1)}"
    return None


def parse_expected_answer(expected_answer: str, category: str, query: str = "") -> tuple[list[dict], list[str]]:
    """
    Parse expected_answer into structured expected_values and required_facts.

    Returns:
        expected_values: List of numeric value dicts
        required_facts: List of atomic fact strings
    """
    expected_values = []
    required_facts = []

    if category == "negative":
        # Negative questions have no expected values
        return [], []

    # Extract fiscal period from query or answer
    period = extract_fiscal_period(query) or extract_fiscal_period(expected_answer)

    # Try to extract numeric values with units
    # Use a single comprehensive pattern to avoid duplicates
    # Match: $416,161 million, $94.83 billion, 1.64 million, etc.
    pattern = r"\$?([\d,]+(?:\.\d+)?)\s*(million|billion|M|B|%|percent)?"

    seen_values = set()
    matches = re.findall(pattern, expected_answer, re.IGNORECASE)

    for match in matches:
        value_str = match[0].replace(",", "")
        try:
            value = float(value_str)

            # Skip very small numbers that are likely not the main value
            if value < 0.01:
                continue

            unit = None
            currency = None

            if len(match) > 1 and match[1]:
                unit_str = match[1].lower()
                if unit_str in ("million", "millions", "m"):
                    unit = "million"
                elif unit_str in ("billion", "billions", "b"):
                    unit = "billion"
                elif unit_str in ("%", "percent"):
                    unit = "percent"

            # Detect currency from $ prefix
            if "$" in expected_answer:
                currency = "USD"

            # Create a key for deduplication
            value_key = (value, unit, currency)
            if value_key not in seen_values:
                seen_values.add(value_key)
                expected_values.append({
                    "value": value,
                    "unit": unit,
                    "currency": currency,
                    "period": period,
                })
        except ValueError:
            pass

    # For non-numeric categories, split into atomic facts
    if category in ("trend_explanation", "risk_factor", "business_description",
                    "accounting_policy", "cross_section", "cross_document"):
        # Split on sentence-ending periods (not decimal points) and semicolons
        # Use negative lookbehind to avoid splitting on numbers like $2.69
        sentences = re.split(r"(?<![0-9])\.(?:\s|$)|;", expected_answer)
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:
                required_facts.append(sentence)

    # For exact_numeric, the answer itself is the required fact
    if category == "exact_numeric" and expected_answer:
        required_facts.append(expected_answer)

    return expected_values, required_facts


def find_gold_chunks(
    question: dict,
    element_to_chunks: dict[str, dict[int, list[dict]]],
    review_items: list[ReviewItem],
) -> tuple[list[dict], list[str]]:
    """
    Find gold chunks that support the expected answer.

    Returns:
        gold_chunks: List of chunk dicts that support the answer
        review_notes: List of notes about ambiguous mappings
    """
    gold_chunks = []
    review_notes = []
    seen_chunk_ids = set()

    doc_id = question.get("doc_id")
    relevant = question.get("relevant", [])
    expected_answer = question.get("expected_answer", "")

    for rel in relevant:
        if rel.get("match_by") != "element_orders":
            continue

        element_orders = rel.get("value", [])
        rel_doc_id = rel.get("doc_id", doc_id)  # cross-document questions

        if not rel_doc_id or rel_doc_id not in element_to_chunks:
            review_items.append(ReviewItem(
                source_qid=question["id"],
                issue_type="missing_doc",
                description=f"Document {rel_doc_id} not found in chunks",
            ))
            continue

        for eo in element_orders:
            candidate_chunks = element_to_chunks[rel_doc_id].get(eo, [])

            if not candidate_chunks:
                review_items.append(ReviewItem(
                    source_qid=question["id"],
                    issue_type="no_chunks_for_element",
                    description=f"Element {eo} in {rel_doc_id} has no chunks",
                ))
                continue

            if len(candidate_chunks) == 1:
                # Single chunk - straightforward mapping
                chunk = candidate_chunks[0]
                if chunk["chunk_id"] not in seen_chunk_ids:
                    gold_chunks.append(chunk)
                    seen_chunk_ids.add(chunk["chunk_id"])
            else:
                # Multiple chunks - need to verify which contains the answer
                # This is where we must be careful not to auto-label
                matching_chunks = []

                for chunk in candidate_chunks:
                    chunk_text = chunk.get("text", "").lower()

                    # Check if chunk text contains key parts of expected answer
                    # Extract numbers and key terms from expected answer
                    answer_lower = expected_answer.lower()

                    # Look for numeric matches
                    numbers_in_answer = re.findall(r"[\d,]+(?:\.\d+)?", expected_answer)
                    numbers_in_chunk = re.findall(r"[\d,]+(?:\.\d+)?", chunk.get("text", ""))

                    # Check for number overlap
                    number_match = any(
                        n.replace(",", "") in [c.replace(",", "") for c in numbers_in_chunk]
                        for n in numbers_in_answer
                        if len(n) > 2  # Skip very short numbers
                    )

                    # Check for key term overlap
                    key_terms = ["total", "net sales", "revenue", "income", "expense",
                                 "margin", "increase", "decrease", "million", "billion"]
                    term_match = sum(1 for t in key_terms if t in answer_lower and t in chunk_text)

                    if number_match or term_match >= 2:
                        matching_chunks.append(chunk)

                if len(matching_chunks) == 1:
                    chunk = matching_chunks[0]
                    if chunk["chunk_id"] not in seen_chunk_ids:
                        gold_chunks.append(chunk)
                        seen_chunk_ids.add(chunk["chunk_id"])
                elif len(matching_chunks) > 1:
                    # Multiple chunks match - flag for review but include all
                    review_notes.append(
                        f"Element {eo} maps to {len(matching_chunks)} chunks that may contain answer"
                    )
                    review_items.append(ReviewItem(
                        source_qid=question["id"],
                        issue_type="ambiguous_element_mapping",
                        description=f"Element {eo} maps to {len(matching_chunks)} chunks",
                        details={
                            "element_order": eo,
                            "chunk_ids": [c["chunk_id"] for c in matching_chunks],
                        },
                    ))
                    # Include all matching chunks but flag for review
                    for chunk in matching_chunks:
                        if chunk["chunk_id"] not in seen_chunk_ids:
                            gold_chunks.append(chunk)
                            seen_chunk_ids.add(chunk["chunk_id"])
                else:
                    # No chunks match expected answer - include all and flag
                    review_notes.append(
                        f"Element {eo} has {len(candidate_chunks)} chunks but none clearly match answer"
                    )
                    review_items.append(ReviewItem(
                        source_qid=question["id"],
                        issue_type="no_matching_chunk",
                        description=f"Element {eo} chunks don't clearly contain answer",
                        details={
                            "element_order": eo,
                            "chunk_ids": [c["chunk_id"] for c in candidate_chunks],
                            "expected_answer": expected_answer[:100],
                        },
                    ))
                    # Include first chunk but flag for review
                    chunk = candidate_chunks[0]
                    if chunk["chunk_id"] not in seen_chunk_ids:
                        gold_chunks.append(chunk)
                        seen_chunk_ids.add(chunk["chunk_id"])

    return gold_chunks, review_notes


def select_distractors(
    question: dict,
    gold_chunks: list[dict],
    chunks_by_doc: dict[str, list[dict]],
    target_total: int,
    rng: random.Random,
) -> list[dict]:
    """
    Select plausible distractor chunks from the same document.

    Distractors should be:
    - From the same document
    - Related to the topic but not containing the answer
    - Plausible wrong-year, wrong-row, wrong-segment, etc.
    """
    doc_id = question.get("doc_id")
    if not doc_id or doc_id not in chunks_by_doc:
        return []

    gold_chunk_ids = {c["chunk_id"] for c in gold_chunks}
    all_chunks = chunks_by_doc[doc_id]

    # Filter out gold chunks
    candidate_distractors = [c for c in all_chunks if c["chunk_id"] not in gold_chunk_ids]

    if not candidate_distractors:
        return []

    # Score candidates by relevance to the question topic
    query_lower = question.get("query", "").lower()
    category = question.get("category", "")

    scored_candidates = []
    for chunk in candidate_distractors:
        chunk_text = chunk.get("text", "").lower()
        score = 0

        # Prefer chunks from similar sections
        gold_sections = {c.get("section", "") for c in gold_chunks}
        if chunk.get("section", "") in gold_sections:
            score += 3

        # Prefer chunks with similar type (table vs text)
        gold_types = {c.get("type", "") for c in gold_chunks}
        if chunk.get("type", "") in gold_types:
            score += 2

        # Prefer chunks mentioning similar terms
        query_terms = set(query_lower.split())
        chunk_terms = set(chunk_text.split())
        overlap = len(query_terms & chunk_terms)
        score += min(overlap, 5)

        # Prefer chunks with numbers for numeric questions
        if category == "exact_numeric" and re.search(r"\d", chunk_text):
            score += 2

        # Add some randomness to avoid always picking the same distractors
        score += rng.random() * 2

        scored_candidates.append((score, chunk))

    # Sort by score descending
    scored_candidates.sort(key=lambda x: -x[0])

    # Select top distractors
    num_needed = max(0, target_total - len(gold_chunks))
    distractors = [c for _, c in scored_candidates[:num_needed]]

    return distractors


def build_context_entry(chunk: dict, is_gold: bool) -> dict:
    """Build a context entry for a chunk."""
    return {
        "chunk_id": chunk["chunk_id"],
        "is_gold": is_gold,
        "text_sha256": sha256_text(chunk.get("text", "")),
        "chunk": chunk,
    }


def build_oracle_case(
    question: dict,
    gold_chunks: list[dict],
    expected_values: list[dict],
    required_facts: list[str],
    review_notes: list[str],
) -> dict:
    """Build the oracle variant case."""
    return {
        "id": f"{question['id']}_oracle",
        "source_qid": question["id"],
        "query": question["query"],
        "doc_id": question.get("doc_id"),
        "category": question.get("category"),
        "difficulty": question.get("difficulty"),
        "context_variant": "oracle",
        "chunk_run": CHUNK_RUN,
        "answerable": True,
        "expected_answer": question.get("expected_answer"),
        "expected_values": expected_values,
        "required_facts": required_facts,
        "gold_chunk_ids": [c["chunk_id"] for c in gold_chunks],
        "context": [build_context_entry(c, is_gold=True) for c in gold_chunks],
        "unanswerable_reason": None,
        "review_status": "needs_review",
        "review_notes": review_notes,
    }


def build_oracle_plus_distractors_case(
    question: dict,
    gold_chunks: list[dict],
    distractors: list[dict],
    expected_values: list[dict],
    required_facts: list[str],
    review_notes: list[str],
    rng: random.Random,
) -> dict:
    """Build the oracle_plus_distractors variant case."""
    # Combine gold and distractors, then shuffle
    all_chunks = [(c, True) for c in gold_chunks] + [(c, False) for c in distractors]
    rng.shuffle(all_chunks)

    context = [build_context_entry(c, is_gold) for c, is_gold in all_chunks]

    return {
        "id": f"{question['id']}_oracle_plus_distractors",
        "source_qid": question["id"],
        "query": question["query"],
        "doc_id": question.get("doc_id"),
        "category": question.get("category"),
        "difficulty": question.get("difficulty"),
        "context_variant": "oracle_plus_distractors",
        "chunk_run": CHUNK_RUN,
        "answerable": True,
        "expected_answer": question.get("expected_answer"),
        "expected_values": expected_values,
        "required_facts": required_facts,
        "gold_chunk_ids": [c["chunk_id"] for c in gold_chunks],
        "context": context,
        "unanswerable_reason": None,
        "review_status": "needs_review",
        "review_notes": review_notes,
    }


def build_no_gold_case(
    question: dict,
    distractors: list[dict],
    review_notes: list[str],
) -> dict:
    """Build the no_gold variant case."""
    return {
        "id": f"{question['id']}_no_gold",
        "source_qid": question["id"],
        "query": question["query"],
        "doc_id": question.get("doc_id"),
        "category": question.get("category"),
        "difficulty": question.get("difficulty"),
        "context_variant": "no_gold",
        "chunk_run": CHUNK_RUN,
        "answerable": False,
        "expected_answer": None,
        "expected_values": [],
        "required_facts": [],
        "gold_chunk_ids": [],
        "context": [build_context_entry(c, is_gold=False) for c in distractors],
        "unanswerable_reason": "The supplied context does not contain the required evidence.",
        "review_status": "needs_review",
        "review_notes": review_notes,
    }


def build_trap_cases(
    chunks_by_doc: dict[str, list[dict]],
    rng: random.Random,
) -> list[dict]:
    """Build hand-written trap cases."""
    trap_cases = []

    # Trap 1: Wrong fiscal period (FY2020 asked against FY2025 filing)
    trap_cases.append({
        "id": "trap_001",
        "source_qid": None,
        "query": "What was Apple's total net sales in fiscal year 2020?",
        "doc_id": "AAPL_10-K_2025",
        "category": "trap",
        "difficulty": "medium",
        "context_variant": "trap",
        "chunk_run": CHUNK_RUN,
        "answerable": False,
        "expected_answer": None,
        "expected_values": [],
        "required_facts": [],
        "gold_chunk_ids": [],
        "context": _select_trap_context(chunks_by_doc, "AAPL_10-K_2025", rng, 3),
        "unanswerable_reason": "The supplied context is from fiscal 2025, not fiscal 2020.",
        "review_status": "needs_review",
        "review_notes": ["Hand-written trap: wrong fiscal period"],
    })

    # Trap 2: Metric not reported
    trap_cases.append({
        "id": "trap_002",
        "source_qid": None,
        "query": "What was Tesla's customer acquisition cost per vehicle in 2025?",
        "doc_id": "TSLA_10-K_2025",
        "category": "trap",
        "difficulty": "hard",
        "context_variant": "trap",
        "chunk_run": CHUNK_RUN,
        "answerable": False,
        "expected_answer": None,
        "expected_values": [],
        "required_facts": [],
        "gold_chunk_ids": [],
        "context": _select_trap_context(chunks_by_doc, "TSLA_10-K_2025", rng, 3),
        "unanswerable_reason": "Customer acquisition cost per vehicle is not disclosed in the filing.",
        "review_status": "needs_review",
        "review_notes": ["Hand-written trap: metric not reported"],
    })

    # Trap 3: Cross-company mismatch
    trap_cases.append({
        "id": "trap_003",
        "source_qid": None,
        "query": "What was Microsoft's iPhone revenue in fiscal 2026?",
        "doc_id": "MSFT_10-K_2026",
        "category": "trap",
        "difficulty": "easy",
        "context_variant": "trap",
        "chunk_run": CHUNK_RUN,
        "answerable": False,
        "expected_answer": None,
        "expected_values": [],
        "required_facts": [],
        "gold_chunk_ids": [],
        "context": _select_trap_context(chunks_by_doc, "MSFT_10-K_2026", rng, 3),
        "unanswerable_reason": "Microsoft does not sell iPhones; iPhone is an Apple product.",
        "review_status": "needs_review",
        "review_notes": ["Hand-written trap: cross-company product mismatch"],
    })

    # Trap 4: Incomplete calculation (one input missing)
    trap_cases.append({
        "id": "trap_004",
        "source_qid": None,
        "query": "What was Apple's year-over-year revenue growth rate from fiscal 2024 to 2025?",
        "doc_id": "AAPL_10-K_2025",
        "category": "trap",
        "difficulty": "medium",
        "context_variant": "trap",
        "chunk_run": CHUNK_RUN,
        "answerable": False,
        "expected_answer": None,
        "expected_values": [],
        "required_facts": [],
        "gold_chunk_ids": [],
        "context": _select_trap_context_partial(chunks_by_doc, "AAPL_10-K_2025", rng),
        "unanswerable_reason": "The context contains only fiscal 2025 revenue; fiscal 2024 revenue is needed for the calculation.",
        "review_status": "needs_review",
        "review_notes": ["Hand-written trap: incomplete calculation inputs"],
    })

    # Trap 5: Contradictory premise
    trap_cases.append({
        "id": "trap_005",
        "source_qid": None,
        "query": "Why did Tesla's total revenue decrease in 2025 compared to 2024?",
        "doc_id": "TSLA_10-K_2025",
        "category": "trap",
        "difficulty": "medium",
        "context_variant": "trap",
        "chunk_run": CHUNK_RUN,
        "answerable": False,
        "expected_answer": None,
        "expected_values": [],
        "required_facts": [],
        "gold_chunk_ids": [],
        "context": _select_trap_context(chunks_by_doc, "TSLA_10-K_2025", rng, 3),
        "unanswerable_reason": "The premise is incorrect; Tesla's total revenue did not decrease in 2025.",
        "review_status": "needs_review",
        "review_notes": ["Hand-written trap: contradictory premise"],
    })

    # Trap 6: Wrong company context
    trap_cases.append({
        "id": "trap_006",
        "source_qid": None,
        "query": "What was Apple's Azure revenue in fiscal 2025?",
        "doc_id": "AAPL_10-K_2025",
        "category": "trap",
        "difficulty": "easy",
        "context_variant": "trap",
        "chunk_run": CHUNK_RUN,
        "answerable": False,
        "expected_answer": None,
        "expected_values": [],
        "required_facts": [],
        "gold_chunk_ids": [],
        "context": _select_trap_context(chunks_by_doc, "AAPL_10-K_2025", rng, 3),
        "unanswerable_reason": "Azure is a Microsoft product, not an Apple product.",
        "review_status": "needs_review",
        "review_notes": ["Hand-written trap: wrong company product"],
    })

    # Trap 7: Future period not in filing
    trap_cases.append({
        "id": "trap_007",
        "source_qid": None,
        "query": "What is Microsoft's projected revenue for fiscal year 2030?",
        "doc_id": "MSFT_10-K_2026",
        "category": "trap",
        "difficulty": "medium",
        "context_variant": "trap",
        "chunk_run": CHUNK_RUN,
        "answerable": False,
        "expected_answer": None,
        "expected_values": [],
        "required_facts": [],
        "gold_chunk_ids": [],
        "context": _select_trap_context(chunks_by_doc, "MSFT_10-K_2026", rng, 3),
        "unanswerable_reason": "The filing does not contain revenue projections for fiscal 2030.",
        "review_status": "needs_review",
        "review_notes": ["Hand-written trap: future period not disclosed"],
    })

    # Trap 8: Segment not reported separately
    trap_cases.append({
        "id": "trap_008",
        "source_qid": None,
        "query": "What was Tesla's revenue from the European market specifically in 2025?",
        "doc_id": "TSLA_10-K_2025",
        "category": "trap",
        "difficulty": "hard",
        "context_variant": "trap",
        "chunk_run": CHUNK_RUN,
        "answerable": False,
        "expected_answer": None,
        "expected_values": [],
        "required_facts": [],
        "gold_chunk_ids": [],
        "context": _select_trap_context(chunks_by_doc, "TSLA_10-K_2025", rng, 3),
        "unanswerable_reason": "Tesla does not report revenue by European market separately in this filing.",
        "review_status": "needs_review",
        "review_notes": ["Hand-written trap: segment not reported separately"],
    })

    # Trap 9: Metric exists but wrong unit asked
    trap_cases.append({
        "id": "trap_009",
        "source_qid": None,
        "query": "What was Apple's total net sales in fiscal 2025 in euros?",
        "doc_id": "AAPL_10-K_2025",
        "category": "trap",
        "difficulty": "medium",
        "context_variant": "trap",
        "chunk_run": CHUNK_RUN,
        "answerable": False,
        "expected_answer": None,
        "expected_values": [],
        "required_facts": [],
        "gold_chunk_ids": [],
        "context": _select_trap_context(chunks_by_doc, "AAPL_10-K_2025", rng, 3),
        "unanswerable_reason": "Apple reports net sales in USD, not euros. Currency conversion is not provided.",
        "review_status": "needs_review",
        "review_notes": ["Hand-written trap: wrong currency unit requested"],
    })

    # Trap 10: Derived metric not directly stated
    trap_cases.append({
        "id": "trap_010",
        "source_qid": None,
        "query": "What was Microsoft's revenue per employee in fiscal 2026?",
        "doc_id": "MSFT_10-K_2026",
        "category": "trap",
        "difficulty": "hard",
        "context_variant": "trap",
        "chunk_run": CHUNK_RUN,
        "answerable": False,
        "expected_answer": None,
        "expected_values": [],
        "required_facts": [],
        "gold_chunk_ids": [],
        "context": _select_trap_context(chunks_by_doc, "MSFT_10-K_2026", rng, 3),
        "unanswerable_reason": "Revenue per employee is not disclosed; employee count is not in the provided context.",
        "review_status": "needs_review",
        "review_notes": ["Hand-written trap: derived metric with missing input"],
    })

    # Trap 11: Quarterly data asked but only annual available
    trap_cases.append({
        "id": "trap_011",
        "source_qid": None,
        "query": "What was Tesla's Q3 2025 automotive revenue?",
        "doc_id": "TSLA_10-K_2025",
        "category": "trap",
        "difficulty": "medium",
        "context_variant": "trap",
        "chunk_run": CHUNK_RUN,
        "answerable": False,
        "expected_answer": None,
        "expected_values": [],
        "required_facts": [],
        "gold_chunk_ids": [],
        "context": _select_trap_context(chunks_by_doc, "TSLA_10-K_2025", rng, 3),
        "unanswerable_reason": "The 10-K filing contains annual data; quarterly breakdown is not provided in this context.",
        "review_status": "needs_review",
        "review_notes": ["Hand-written trap: quarterly data not in annual filing"],
    })

    # Trap 12: Asking about subsidiary not mentioned
    trap_cases.append({
        "id": "trap_012",
        "source_qid": None,
        "query": "What was the revenue of Apple's Beats subsidiary in fiscal 2025?",
        "doc_id": "AAPL_10-K_2025",
        "category": "trap",
        "difficulty": "hard",
        "context_variant": "trap",
        "chunk_run": CHUNK_RUN,
        "answerable": False,
        "expected_answer": None,
        "expected_values": [],
        "required_facts": [],
        "gold_chunk_ids": [],
        "context": _select_trap_context(chunks_by_doc, "AAPL_10-K_2025", rng, 3),
        "unanswerable_reason": "Beats subsidiary revenue is not reported separately in the filing.",
        "review_status": "needs_review",
        "review_notes": ["Hand-written trap: subsidiary revenue not disclosed"],
    })

    return trap_cases


def _select_trap_context(
    chunks_by_doc: dict[str, list[dict]],
    doc_id: str,
    rng: random.Random,
    count: int,
) -> list[dict]:
    """Select random chunks for trap context."""
    if doc_id not in chunks_by_doc:
        return []

    all_chunks = chunks_by_doc[doc_id]
    selected = rng.sample(all_chunks, min(count, len(all_chunks)))
    return [build_context_entry(c, is_gold=False) for c in selected]


def _select_trap_context_partial(
    chunks_by_doc: dict[str, list[dict]],
    doc_id: str,
    rng: random.Random,
) -> list[dict]:
    """Select chunks that contain only partial information for calculation traps."""
    if doc_id not in chunks_by_doc:
        return []

    # For calculation traps, try to find chunks with numbers but not complete data
    all_chunks = chunks_by_doc[doc_id]
    numeric_chunks = [c for c in all_chunks if re.search(r"\d", c.get("text", ""))]

    if numeric_chunks:
        selected = rng.sample(numeric_chunks, min(3, len(numeric_chunks)))
    else:
        selected = rng.sample(all_chunks, min(3, len(all_chunks)))

    return [build_context_entry(c, is_gold=False) for c in selected]


def handle_negative_questions(
    question: dict,
    chunks_by_doc: dict[str, list[dict]],
    rng: random.Random,
) -> list[dict]:
    """
    Handle negative questions (those with no relevant elements).

    For negative questions, all variants should be unanswerable.
    """
    cases = []
    doc_id = question.get("doc_id")

    # Select some plausible context chunks
    context_chunks = []
    if doc_id and doc_id in chunks_by_doc:
        all_chunks = chunks_by_doc[doc_id]
        context_chunks = rng.sample(all_chunks, min(5, len(all_chunks)))

    context = [build_context_entry(c, is_gold=False) for c in context_chunks]

    # All three variants are unanswerable for negative questions
    for variant in ["oracle", "oracle_plus_distractors", "no_gold"]:
        cases.append({
            "id": f"{question['id']}_{variant}",
            "source_qid": question["id"],
            "query": question["query"],
            "doc_id": doc_id,
            "category": question.get("category"),
            "difficulty": question.get("difficulty"),
            "context_variant": variant,
            "chunk_run": CHUNK_RUN,
            "answerable": False,
            "expected_answer": None,
            "expected_values": [],
            "required_facts": [],
            "gold_chunk_ids": [],
            "context": context,
            "unanswerable_reason": question.get("expected_answer", "Not disclosed in these filings"),
            "review_status": "needs_review",
            "review_notes": ["Negative question: no relevant elements in source"],
        })

    return cases


def handle_cross_document_questions(
    question: dict,
    element_to_chunks: dict[str, dict[int, list[dict]]],
    chunks_by_doc: dict[str, list[dict]],
    review_items: list[ReviewItem],
    rng: random.Random,
) -> list[dict]:
    """
    Handle cross-document questions that span multiple filings.

    These questions have relevant elements from multiple documents.
    """
    cases = []

    # Collect gold chunks from all relevant documents
    all_gold_chunks = []
    review_notes = []

    relevant = question.get("relevant", [])
    expected_answer = question.get("expected_answer", "")

    for rel in relevant:
        if rel.get("match_by") != "element_orders":
            continue

        element_orders = rel.get("value", [])
        rel_doc_id = rel.get("doc_id")

        if not rel_doc_id or rel_doc_id not in element_to_chunks:
            review_items.append(ReviewItem(
                source_qid=question["id"],
                issue_type="missing_doc_cross",
                description=f"Cross-document: {rel_doc_id} not found",
            ))
            continue

        for eo in element_orders:
            candidate_chunks = element_to_chunks[rel_doc_id].get(eo, [])
            if candidate_chunks:
                # Take first chunk for simplicity, flag for review
                all_gold_chunks.append(candidate_chunks[0])
                if len(candidate_chunks) > 1:
                    review_notes.append(f"Element {eo} in {rel_doc_id} has {len(candidate_chunks)} chunks")

    if not all_gold_chunks:
        review_items.append(ReviewItem(
            source_qid=question["id"],
            issue_type="no_gold_chunks_cross",
            description="Cross-document question has no gold chunks",
        ))
        return []

    # Parse expected answer
    expected_values, required_facts = parse_expected_answer(
        expected_answer, question.get("category", ""), question.get("query", "")
    )

    # Build oracle case
    cases.append({
        "id": f"{question['id']}_oracle",
        "source_qid": question["id"],
        "query": question["query"],
        "doc_id": None,  # Cross-document
        "category": question.get("category"),
        "difficulty": question.get("difficulty"),
        "context_variant": "oracle",
        "chunk_run": CHUNK_RUN,
        "answerable": True,
        "expected_answer": expected_answer,
        "expected_values": expected_values,
        "required_facts": required_facts,
        "gold_chunk_ids": [c["chunk_id"] for c in all_gold_chunks],
        "context": [build_context_entry(c, is_gold=True) for c in all_gold_chunks],
        "unanswerable_reason": None,
        "review_status": "needs_review",
        "review_notes": review_notes + ["Cross-document question"],
    })

    # Build oracle_plus_distractors case
    # Add distractors from each document
    distractors = []
    gold_ids = {c["chunk_id"] for c in all_gold_chunks}

    for doc_id in chunks_by_doc:
        doc_chunks = [c for c in chunks_by_doc[doc_id] if c["chunk_id"] not in gold_ids]
        if doc_chunks:
            distractors.extend(rng.sample(doc_chunks, min(2, len(doc_chunks))))

    all_context = [(c, True) for c in all_gold_chunks] + [(c, False) for c in distractors[:5]]
    rng.shuffle(all_context)

    cases.append({
        "id": f"{question['id']}_oracle_plus_distractors",
        "source_qid": question["id"],
        "query": question["query"],
        "doc_id": None,
        "category": question.get("category"),
        "difficulty": question.get("difficulty"),
        "context_variant": "oracle_plus_distractors",
        "chunk_run": CHUNK_RUN,
        "answerable": True,
        "expected_answer": expected_answer,
        "expected_values": expected_values,
        "required_facts": required_facts,
        "gold_chunk_ids": [c["chunk_id"] for c in all_gold_chunks],
        "context": [build_context_entry(c, is_gold) for c, is_gold in all_context],
        "unanswerable_reason": None,
        "review_status": "needs_review",
        "review_notes": review_notes + ["Cross-document question"],
    })

    # Build no_gold case - distractors only
    no_gold_distractors = distractors[:5] if distractors else []
    cases.append({
        "id": f"{question['id']}_no_gold",
        "source_qid": question["id"],
        "query": question["query"],
        "doc_id": None,
        "category": question.get("category"),
        "difficulty": question.get("difficulty"),
        "context_variant": "no_gold",
        "chunk_run": CHUNK_RUN,
        "answerable": False,
        "expected_answer": None,
        "expected_values": [],
        "required_facts": [],
        "gold_chunk_ids": [],
        "context": [build_context_entry(c, is_gold=False) for c in no_gold_distractors],
        "unanswerable_reason": "The supplied context does not contain the required evidence from all documents.",
        "review_status": "needs_review",
        "review_notes": review_notes + ["Cross-document question"],
    })

    return cases


def build_generation_goldens(root: Path) -> tuple[list[dict], list[ReviewItem]]:
    """
    Build the complete generation golden dataset.

    Returns:
        cases: List of all generation cases
        review_items: List of items flagged for manual review
    """
    # Initialize RNG for reproducibility
    rng = random.Random(RANDOM_SEED)

    # Load data
    retrieval_goldens = load_retrieval_goldens(root / "goldens" / "golden_retriever_dataset.jsonl")
    chunk_dir = root / "data" / "chunks" / CHUNK_RUN
    chunks_by_doc, element_to_chunks = load_chunks(chunk_dir)

    cases = []
    review_items = []

    print(f"Processing {len(retrieval_goldens)} source questions...")

    for question in retrieval_goldens:
        qid = question["id"]
        category = question.get("category", "")
        relevant = question.get("relevant", [])
        doc_id = question.get("doc_id")

        # Handle negative questions (no relevant elements)
        if category == "negative" or not relevant:
            cases.extend(handle_negative_questions(question, chunks_by_doc, rng))
            continue

        # Handle cross-document questions
        if not doc_id:
            cases.extend(handle_cross_document_questions(
                question, element_to_chunks, chunks_by_doc, review_items, rng
            ))
            continue

        # Standard single-document questions
        gold_chunks, review_notes = find_gold_chunks(
            question, element_to_chunks, review_items
        )

        if not gold_chunks:
            review_items.append(ReviewItem(
                source_qid=qid,
                issue_type="no_gold_chunks",
                description="No gold chunks found for question",
            ))
            # Still create cases but flag them
            review_notes.append("WARNING: No gold chunks found")

        # Parse expected answer
        expected_values, required_facts = parse_expected_answer(
            question.get("expected_answer", ""), category, question.get("query", "")
        )

        # Select distractors
        distractors = select_distractors(
            question, gold_chunks, chunks_by_doc, TARGET_CONTEXT_SIZE, rng
        )

        # Build oracle case
        cases.append(build_oracle_case(
            question, gold_chunks, expected_values, required_facts, review_notes
        ))

        # Build oracle_plus_distractors case
        cases.append(build_oracle_plus_distractors_case(
            question, gold_chunks, distractors, expected_values, required_facts, review_notes, rng
        ))

        # Build no_gold case
        # Use different distractors to ensure gold chunks are not included
        no_gold_distractors = select_distractors(
            question, [], chunks_by_doc, TARGET_CONTEXT_SIZE, rng
        )
        # Filter out any gold chunks
        gold_ids = {c["chunk_id"] for c in gold_chunks}
        no_gold_distractors = [c for c in no_gold_distractors if c["chunk_id"] not in gold_ids]

        cases.append(build_no_gold_case(question, no_gold_distractors, review_notes))

    # Add trap cases
    print("Adding trap cases...")
    trap_cases = build_trap_cases(chunks_by_doc, rng)
    cases.extend(trap_cases)

    print(f"Generated {len(cases)} total cases")
    print(f"Flagged {len(review_items)} items for review")

    return cases, review_items


def main():
    """Main entry point."""
    root = Path(__file__).resolve().parents[2]

    print("Building generation golden dataset...")
    print(f"Chunk run: {CHUNK_RUN}")
    print(f"Random seed: {RANDOM_SEED}")
    print()

    cases, review_items = build_generation_goldens(root)

    # Write generation dataset
    output_path = root / "goldens" / "golden_generation_dataset.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")
    print(f"Wrote {len(cases)} cases to {output_path}")

    # Write review items
    review_path = root / "goldens" / "generation_dataset_review.jsonl"
    with open(review_path, "w", encoding="utf-8") as f:
        for item in review_items:
            f.write(json.dumps({
                "source_qid": item.source_qid,
                "issue_type": item.issue_type,
                "description": item.description,
                "details": item.details,
            }, ensure_ascii=False) + "\n")
    print(f"Wrote {len(review_items)} review items to {review_path}")

    # Print summary
    print()
    print("=== Summary ===")

    # Count by variant
    variant_counts = {}
    for case in cases:
        variant = case.get("context_variant", "unknown")
        variant_counts[variant] = variant_counts.get(variant, 0) + 1

    print("Cases by variant:")
    for variant, count in sorted(variant_counts.items()):
        print(f"  {variant}: {count}")

    # Count by category
    category_counts = {}
    for case in cases:
        cat = case.get("category", "unknown")
        category_counts[cat] = category_counts.get(cat, 0) + 1

    print()
    print("Cases by category:")
    for cat, count in sorted(category_counts.items()):
        print(f"  {cat}: {count}")

    # Count review issues by type
    if review_items:
        print()
        print("Review issues by type:")
        issue_counts = {}
        for item in review_items:
            issue_counts[item.issue_type] = issue_counts.get(item.issue_type, 0) + 1
        for issue_type, count in sorted(issue_counts.items()):
            print(f"  {issue_type}: {count}")


if __name__ == "__main__":
    main()
