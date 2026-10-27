#!/usr/bin/env python3
"""
Validate the generation golden dataset.

This script validates:
- JSONL parses correctly and IDs are unique
- Every source question has exactly one oracle, oracle_plus_distractors, and no_gold case
- Every context chunk belongs to the case's doc_id (except cross-company traps)
- Stored chunk_id agrees with chunk["chunk_id"]
- Stored SHA-256 agrees with exact chunk text
- answerable=true has at least one gold chunk and an expected answer/fact
- answerable=false has no gold chunks and has an unanswerable_reason
- gold_chunk_ids exactly equals context entries where is_gold=true
- Oracle-plus-distractor cases normally contain five chunks
- Expected total count is 190-195 cases

Usage:
    uv run python scripts/eval_scripts/validate_generation_goldens.py
    uv run python scripts/eval_scripts/validate_generation_goldens.py --strict  # Exit 1 on warnings
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

# Expected chunk run
EXPECTED_CHUNK_RUN = "2026-10-16_17-03-28"

# Expected number of source questions
EXPECTED_SOURCE_QUESTIONS = 60

# Expected variants per source question
EXPECTED_VARIANTS = {"oracle", "oracle_plus_distractors", "no_gold"}

# Expected total cases (60 * 3 + 10-15 traps)
MIN_EXPECTED_CASES = 190
MAX_EXPECTED_CASES = 200


class ValidationError:
    """A validation error."""

    def __init__(self, case_id: str, error_type: str, message: str):
        self.case_id = case_id
        self.error_type = error_type
        self.message = message

    def __str__(self):
        return f"[{self.case_id}] {self.error_type}: {self.message}"


class ValidationWarning:
    """A validation warning (non-fatal)."""

    def __init__(self, case_id: str, warning_type: str, message: str):
        self.case_id = case_id
        self.warning_type = warning_type
        self.message = message

    def __str__(self):
        return f"[{self.case_id}] {self.warning_type}: {self.message}"


def sha256_text(text: str) -> str:
    """Compute SHA-256 hash of UTF-8 encoded text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_case(
    case: dict, case_idx: int
) -> tuple[list[ValidationError], list[ValidationWarning]]:
    """Validate a single case."""
    errors = []
    warnings = []

    case_id = case.get("id", f"case_{case_idx}")

    # Required fields
    required_fields = [
        "id",
        "source_qid",
        "query",
        "category",
        "difficulty",
        "context_variant",
        "chunk_run",
        "answerable",
        "context",
        "gold_chunk_ids",
        "review_status",
        "review_notes",
    ]

    for field in required_fields:
        if field not in case:
            errors.append(
                ValidationError(case_id, "missing_field", f"Missing required field: {field}")
            )

    # Validate chunk_run
    if case.get("chunk_run") != EXPECTED_CHUNK_RUN:
        errors.append(
            ValidationError(
                case_id,
                "wrong_chunk_run",
                f"Expected chunk_run={EXPECTED_CHUNK_RUN}, got {case.get('chunk_run')}",
            )
        )

    # Validate context_variant
    variant = case.get("context_variant")
    if variant not in EXPECTED_VARIANTS and variant != "trap":
        errors.append(
            ValidationError(case_id, "invalid_variant", f"Invalid context_variant: {variant}")
        )

    # Validate context entries
    context = case.get("context", [])
    gold_chunk_ids_from_context = []

    for i, entry in enumerate(context):
        # Check required context entry fields
        if "chunk_id" not in entry:
            errors.append(
                ValidationError(
                    case_id, "context_missing_chunk_id", f"Context entry {i} missing chunk_id"
                )
            )
            continue

        if "is_gold" not in entry:
            errors.append(
                ValidationError(
                    case_id, "context_missing_is_gold", f"Context entry {i} missing is_gold"
                )
            )

        if "text_sha256" not in entry:
            errors.append(
                ValidationError(
                    case_id, "context_missing_hash", f"Context entry {i} missing text_sha256"
                )
            )

        if "chunk" not in entry:
            errors.append(
                ValidationError(
                    case_id, "context_missing_chunk", f"Context entry {i} missing chunk dict"
                )
            )
            continue

        chunk = entry.get("chunk", {})

        # Validate chunk_id consistency
        if entry.get("chunk_id") != chunk.get("chunk_id"):
            errors.append(
                ValidationError(
                    case_id,
                    "chunk_id_mismatch",
                    f"Context entry {i}: chunk_id mismatch ({entry.get('chunk_id')} vs {chunk.get('chunk_id')})",
                )
            )

        # Validate SHA-256 hash
        chunk_text = chunk.get("text", "")
        expected_hash = sha256_text(chunk_text)
        if entry.get("text_sha256") != expected_hash:
            errors.append(
                ValidationError(
                    case_id,
                    "hash_mismatch",
                    f"Context entry {i}: SHA-256 mismatch for chunk {entry.get('chunk_id')}",
                )
            )

        # Validate doc_id consistency (except for cross-document and trap cases)
        case_doc_id = case.get("doc_id")
        chunk_doc_id = chunk.get("doc_id")
        if case_doc_id and chunk_doc_id and case_doc_id != chunk_doc_id:
            if variant != "trap" and case.get("category") != "cross_document":
                warnings.append(
                    ValidationWarning(
                        case_id,
                        "doc_id_mismatch",
                        f"Context entry {i}: chunk doc_id ({chunk_doc_id}) differs from case doc_id ({case_doc_id})",
                    )
                )

        # Track gold chunks
        if entry.get("is_gold"):
            gold_chunk_ids_from_context.append(entry.get("chunk_id"))

    # Validate gold_chunk_ids consistency
    gold_chunk_ids = case.get("gold_chunk_ids", [])
    if set(gold_chunk_ids) != set(gold_chunk_ids_from_context):
        errors.append(
            ValidationError(
                case_id,
                "gold_chunk_ids_mismatch",
                f"gold_chunk_ids ({gold_chunk_ids}) doesn't match is_gold=true entries ({gold_chunk_ids_from_context})",
            )
        )

    # Validate answerability constraints
    answerable = case.get("answerable")

    if answerable:
        # Answerable cases must have gold chunks and expected answer
        if not gold_chunk_ids and variant != "trap":
            errors.append(
                ValidationError(case_id, "answerable_no_gold", "answerable=true but no gold chunks")
            )

        expected_answer = case.get("expected_answer")
        expected_values = case.get("expected_values", [])
        required_facts = case.get("required_facts", [])

        if not expected_answer and not expected_values and not required_facts:
            errors.append(
                ValidationError(
                    case_id,
                    "answerable_no_expected",
                    "answerable=true but no expected_answer, expected_values, or required_facts",
                )
            )

        if case.get("unanswerable_reason"):
            warnings.append(
                ValidationWarning(
                    case_id,
                    "answerable_has_reason",
                    "answerable=true but unanswerable_reason is set",
                )
            )
    else:
        # Unanswerable cases must have no gold chunks and have a reason
        if gold_chunk_ids:
            errors.append(
                ValidationError(
                    case_id,
                    "unanswerable_has_gold",
                    f"answerable=false but has gold chunks: {gold_chunk_ids}",
                )
            )

        if not case.get("unanswerable_reason"):
            errors.append(
                ValidationError(
                    case_id, "unanswerable_no_reason", "answerable=false but no unanswerable_reason"
                )
            )

    # Validate oracle_plus_distractors context size
    if variant == "oracle_plus_distractors":
        if len(context) < 3:
            warnings.append(
                ValidationWarning(
                    case_id,
                    "small_context",
                    f"oracle_plus_distractors has only {len(context)} chunks (expected ~5)",
                )
            )
        elif len(context) > 8:
            warnings.append(
                ValidationWarning(
                    case_id,
                    "large_context",
                    f"oracle_plus_distractors has {len(context)} chunks (expected ~5)",
                )
            )

    # Validate review_status
    if case.get("review_status") not in ("needs_review", "approved"):
        warnings.append(
            ValidationWarning(
                case_id,
                "unexpected_review_status",
                f"review_status should be 'needs_review' or 'approved', got '{case.get('review_status')}'",
            )
        )

    return errors, warnings


def validate_dataset(cases: list[dict]) -> tuple[list[ValidationError], list[ValidationWarning]]:
    """Validate the entire dataset."""
    all_errors = []
    all_warnings = []

    # Track IDs for uniqueness
    seen_ids = set()

    # Track source questions and their variants
    source_variants = defaultdict(set)

    # Validate each case
    for i, case in enumerate(cases):
        errors, warnings = validate_case(case, i)
        all_errors.extend(errors)
        all_warnings.extend(warnings)

        case_id = case.get("id", f"case_{i}")

        # Check ID uniqueness
        if case_id in seen_ids:
            all_errors.append(ValidationError(case_id, "duplicate_id", "Duplicate case ID"))
        seen_ids.add(case_id)

        # Track variants for source questions
        source_qid = case.get("source_qid")
        variant = case.get("context_variant")

        if source_qid and variant != "trap":
            source_variants[source_qid].add(variant)

    # Validate that each source question has all three variants
    for source_qid, variants in source_variants.items():
        missing = EXPECTED_VARIANTS - variants
        if missing:
            all_errors.append(
                ValidationError(
                    source_qid, "missing_variants", f"Source question missing variants: {missing}"
                )
            )

        extra = variants - EXPECTED_VARIANTS
        if extra:
            all_warnings.append(
                ValidationWarning(
                    source_qid,
                    "extra_variants",
                    f"Source question has unexpected variants: {extra}",
                )
            )

    # Validate total count
    total_cases = len(cases)
    trap_cases = sum(1 for c in cases if c.get("context_variant") == "trap")
    total_cases - trap_cases

    if total_cases < MIN_EXPECTED_CASES:
        all_warnings.append(
            ValidationWarning(
                "dataset",
                "low_count",
                f"Total cases ({total_cases}) below expected minimum ({MIN_EXPECTED_CASES})",
            )
        )
    elif total_cases > MAX_EXPECTED_CASES:
        all_warnings.append(
            ValidationWarning(
                "dataset",
                "high_count",
                f"Total cases ({total_cases}) above expected maximum ({MAX_EXPECTED_CASES})",
            )
        )

    # Validate source question count
    if len(source_variants) != EXPECTED_SOURCE_QUESTIONS:
        all_warnings.append(
            ValidationWarning(
                "dataset",
                "source_count",
                f"Found {len(source_variants)} source questions, expected {EXPECTED_SOURCE_QUESTIONS}",
            )
        )

    return all_errors, all_warnings


def load_dataset(path: Path) -> list[dict]:
    """Load the generation golden dataset."""
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    cases.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"ERROR: Invalid JSON on line {line_num}: {e}")
                    sys.exit(1)
    return cases


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Validate generation golden dataset")
    parser.add_argument(
        "--strict", action="store_true", help="Exit with code 1 on warnings (not just errors)"
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=None,
        help="Path to dataset (default: goldens/golden_generation_dataset.jsonl)",
    )
    args = parser.parse_args()

    # Determine path
    root = Path(__file__).resolve().parents[2]
    if args.path:
        dataset_path = args.path
    else:
        dataset_path = root / "goldens" / "golden_generation_dataset.jsonl"

    if not dataset_path.exists():
        print(f"ERROR: Dataset not found: {dataset_path}")
        sys.exit(1)

    print(f"Validating: {dataset_path}")
    print()

    # Load dataset
    cases = load_dataset(dataset_path)
    print(f"Loaded {len(cases)} cases")
    print()

    # Validate
    errors, warnings = validate_dataset(cases)

    # Print results
    if errors:
        print(f"=== ERRORS ({len(errors)}) ===")
        for error in errors[:50]:  # Limit output
            print(f"  {error}")
        if len(errors) > 50:
            print(f"  ... and {len(errors) - 50} more errors")
        print()

    if warnings:
        print(f"=== WARNINGS ({len(warnings)}) ===")
        for warning in warnings[:50]:  # Limit output
            print(f"  {warning}")
        if len(warnings) > 50:
            print(f"  ... and {len(warnings) - 50} more warnings")
        print()

    # Print summary
    print("=== SUMMARY ===")

    # Count by variant
    variant_counts = defaultdict(int)
    for case in cases:
        variant_counts[case.get("context_variant", "unknown")] += 1

    print("Cases by variant:")
    for variant, count in sorted(variant_counts.items()):
        print(f"  {variant}: {count}")

    # Count by category
    category_counts = defaultdict(int)
    for case in cases:
        category_counts[case.get("category", "unknown")] += 1

    print()
    print("Cases by category:")
    for cat, count in sorted(category_counts.items()):
        print(f"  {cat}: {count}")

    # Count answerable vs unanswerable
    answerable_count = sum(1 for c in cases if c.get("answerable"))
    unanswerable_count = len(cases) - answerable_count

    print()
    print(f"Answerable: {answerable_count}")
    print(f"Unanswerable: {unanswerable_count}")

    # Final status
    print()
    if errors:
        print(f"FAILED: {len(errors)} errors, {len(warnings)} warnings")
        sys.exit(1)
    elif warnings and args.strict:
        print(f"FAILED (strict): {len(warnings)} warnings")
        sys.exit(1)
    elif warnings:
        print(f"PASSED with {len(warnings)} warnings")
        sys.exit(0)
    else:
        print("PASSED: No errors or warnings")
        sys.exit(0)


if __name__ == "__main__":
    main()
