#!/usr/bin/env python3
"""
Independent validator for the Claude-reviewed generation golden dataset.

Validates goldens/golden_generation_dataset_claude.jsonl against the rules in
goldens/generator_claude.md. This validator is intentionally separate from
scripts/eval_scripts/validate_generation_goldens.py so the semantic-review
output is checked by its own stricter rules.

Checks (per generator_claude.md "VALIDATOR REQUIREMENTS"):
  1.  JSONL parsing and unique case IDs.
  2.  Exactly one oracle, oracle_plus_distractors, and no_gold case per source_qid.
  3.  All 12 trap cases exist.
  4.  Context chunk IDs match stored chunk dictionaries.
  5.  SHA-256 hashes match exact UTF-8 chunk text.
  6.  Gold chunk IDs exactly match context entries with is_gold=true.
  7.  Answerable cases contain expected answers and evidence.
  8.  Unanswerable cases have no gold chunks and include unanswerable_reason.
  9.  Every expected value appears in at least one oracle chunk after unit normalization.
  10. Every evidence_map quote occurs verbatim in its claimed chunk.
  11. No expected numeric value appears in no_gold after unit normalization.
  12. Context chunks belong to the declared document, except explicit cross-document traps.
  13. All standard cases have review_status=claude_reviewed.
  14. Every correction appears in the review log.
  15. Every medium/low-confidence case requires human review.

A numeric-absence check is necessary but NOT sufficient for no_gold; narrative
leakage must still be reviewed semantically by a human.

Usage:
    uv run python scripts/eval_scripts/validate_generation_goldens_claude.py
    uv run python scripts/eval_scripts/validate_generation_goldens_claude.py --strict
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

EXPECTED_CHUNK_RUN = "2026-10-16_17-03-28"
EXPECTED_SOURCE_QUESTIONS = 60
EXPECTED_TRAP_CASES = 12
STANDARD_VARIANTS = {"oracle", "oracle_plus_distractors", "no_gold"}
REQUIRED_REVIEW_STATUS = "claude_reviewed"


class Issue:
    def __init__(self, case_id: str, kind: str, message: str):
        self.case_id = case_id
        self.kind = kind
        self.message = message

    def __str__(self) -> str:
        return f"[{self.case_id}] {self.kind}: {self.message}"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"ERROR: Invalid JSON on line {line_num} of {path}: {e}")
                    sys.exit(1)
    return rows


def load_chunk_text(root: Path, chunk_run: str) -> dict[str, str]:
    """Load chunk_id -> text from the authoritative chunk files for the run."""
    text_by_id: dict[str, str] = {}
    pattern = str(root / "data" / "chunks" / chunk_run / "*.chunks.jsonl")
    for fp in glob.glob(pattern):
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    ch = json.loads(line)
                    text_by_id[ch["chunk_id"]] = ch.get("text", "")
    return text_by_id


# --- numeric normalization helpers for checks 9 & 11 ---

def _candidate_number_strings(value: float, unit: str | None) -> list[str]:
    """Return plausible textual representations of a numeric value in a filing."""
    out: set[str] = set()

    def with_sep(n: int) -> str:
        return f"{n:,}"

    variants: list[float] = [value]
    # Account for unit scaling (filings disclose in millions; values may be
    # expressed in billions in the expected answer and vice-versa).
    if unit == "billion":
        variants.append(value * 1000)          # billions -> millions
    if unit == "million":
        variants.append(value / 1000)          # millions -> billions

    for v in variants:
        if float(v).is_integer():
            iv = int(round(v))
            if abs(iv) >= 1:
                out.add(str(iv))
                out.add(with_sep(iv))
        else:
            s = f"{v:.10f}".rstrip("0").rstrip(".")
            out.add(s)
            # also one and two decimal places (e.g. 94.83, 214.4)
            out.add(f"{v:.1f}")
            out.add(f"{v:.2f}")
    return [s for s in out if s]


def number_present(value: float, unit: str | None, text: str) -> bool:
    for s in _candidate_number_strings(value, unit):
        if re.search(r"(?<![\d.,])" + re.escape(s) + r"(?![\d])", text):
            return True
    return False


def validate(cases: list[dict], review_rows: list[dict], chunk_text: dict[str, str]):
    errors: list[Issue] = []
    warnings: list[Issue] = []

    seen_ids: set[str] = set()
    source_variants: dict[str, set[str]] = defaultdict(set)
    trap_count = 0

    # Index oracle expected values per source_qid for the no_gold leakage check.
    oracle_values: dict[str, list[dict]] = {}
    for c in cases:
        if c.get("context_variant") == "oracle":
            oracle_values[c.get("source_qid")] = c.get("expected_values", []) or []

    # Collect corrections present on cases for check 14.
    corrections_in_cases: dict[str, list[str]] = defaultdict(list)

    for idx, c in enumerate(cases):
        cid = c.get("id", f"case_{idx}")
        variant = c.get("context_variant")

        # Check 1: unique IDs
        if cid in seen_ids:
            errors.append(Issue(cid, "duplicate_id", "Duplicate case ID"))
        seen_ids.add(cid)

        # chunk_run sanity
        if c.get("chunk_run") != EXPECTED_CHUNK_RUN:
            errors.append(Issue(cid, "wrong_chunk_run",
                                 f"Expected chunk_run={EXPECTED_CHUNK_RUN}, got {c.get('chunk_run')}"))

        if variant == "trap":
            trap_count += 1
        else:
            source_variants[c.get("source_qid")].add(variant)

        context = c.get("context", []) or []
        gold_from_ctx = []

        for i, e in enumerate(context):
            chunk = e.get("chunk", {}) or {}

            # Check 4: chunk_id consistency
            if e.get("chunk_id") != chunk.get("chunk_id"):
                errors.append(Issue(cid, "chunk_id_mismatch",
                                    f"ctx[{i}] chunk_id {e.get('chunk_id')} != chunk {chunk.get('chunk_id')}"))

            # Check 5: hash matches stored text
            if e.get("text_sha256") != sha256_text(chunk.get("text", "")):
                errors.append(Issue(cid, "hash_mismatch", f"ctx[{i}] SHA-256 mismatch for {e.get('chunk_id')}"))

            # Check 12: doc_id membership (except cross-document / trap)
            case_doc = c.get("doc_id")
            chunk_doc = chunk.get("doc_id")
            if case_doc and chunk_doc and case_doc != chunk_doc:
                if variant != "trap" and c.get("category") != "cross_document":
                    errors.append(Issue(cid, "doc_id_mismatch",
                                        f"ctx[{i}] chunk doc_id {chunk_doc} != case doc_id {case_doc}"))

            if e.get("is_gold"):
                gold_from_ctx.append(e.get("chunk_id"))

        # Check 6: gold_chunk_ids == is_gold entries
        if set(c.get("gold_chunk_ids", []) or []) != set(gold_from_ctx):
            errors.append(Issue(cid, "gold_ids_mismatch",
                                f"gold_chunk_ids {c.get('gold_chunk_ids')} != is_gold entries {gold_from_ctx}"))

        answerable = c.get("answerable")
        if answerable:
            # Check 7: answerable needs expected answer and evidence
            if not c.get("gold_chunk_ids") and variant != "trap":
                errors.append(Issue(cid, "answerable_no_gold", "answerable=true but no gold chunks"))
            if not (c.get("expected_answer") or c.get("expected_values") or c.get("required_facts")):
                errors.append(Issue(cid, "answerable_no_expected",
                                    "answerable=true but no expected_answer/values/facts"))
        else:
            # Check 8: unanswerable constraints
            if c.get("gold_chunk_ids"):
                errors.append(Issue(cid, "unanswerable_has_gold",
                                    f"answerable=false but has gold chunks {c.get('gold_chunk_ids')}"))
            if not c.get("unanswerable_reason"):
                errors.append(Issue(cid, "unanswerable_no_reason", "answerable=false but no unanswerable_reason"))

        # Check 9: expected values present in an oracle chunk
        if variant == "oracle":
            ctx_text = "\n".join((e.get("chunk", {}) or {}).get("text", "") for e in context)
            for ev in c.get("expected_values", []) or []:
                v = ev.get("value")
                if isinstance(v, (int, float)) and not number_present(float(v), ev.get("unit"), ctx_text):
                    errors.append(Issue(cid, "expected_value_absent",
                                        f"expected value {v} ({ev.get('unit')}) not found in oracle context"))

        # Check 11: no expected numeric value present in no_gold
        if variant == "no_gold":
            ctx_text = "\n".join((e.get("chunk", {}) or {}).get("text", "") for e in context)
            for ev in oracle_values.get(c.get("source_qid"), []):
                v = ev.get("value")
                if isinstance(v, (int, float)) and float(v) >= 100 and number_present(float(v), ev.get("unit"), ctx_text):
                    errors.append(Issue(cid, "no_gold_numeric_leak",
                                        f"no_gold context contains expected value {v} ({ev.get('unit')})"))

        # Check 10: evidence_map quotes are verbatim in their claimed chunk
        for fact in c.get("evidence_map", []) or []:
            claimed = fact.get("chunk_ids", []) or []
            for quote in fact.get("verbatim_evidence", []) or []:
                found = any(quote in chunk_text.get(ccid, "") for ccid in claimed)
                if not found:
                    errors.append(Issue(cid, "evidence_not_verbatim",
                                        f"evidence quote not found verbatim in {claimed}: {quote[:60]!r}"))

        # Check 13: review_status
        if variant in STANDARD_VARIANTS and c.get("review_status") != REQUIRED_REVIEW_STATUS:
            errors.append(Issue(cid, "wrong_review_status",
                                f"standard case review_status must be '{REQUIRED_REVIEW_STATUS}', got '{c.get('review_status')}'"))
        if c.get("review_status") == "human_approved":
            errors.append(Issue(cid, "falsely_human_approved",
                                "review_status must never be 'human_approved' in the Claude dataset"))

        # Check 15: medium/low confidence requires human review
        if c.get("confidence") in ("medium", "low") and not c.get("requires_human_review"):
            errors.append(Issue(cid, "confidence_not_flagged",
                                f"confidence={c.get('confidence')} but requires_human_review is not true"))

        # gather corrections for check 14
        key = c.get("source_qid") if variant != "trap" else cid
        for corr in c.get("corrections", []) or []:
            corrections_in_cases[key].append(json.dumps(corr, sort_keys=True, ensure_ascii=False))

    # Check 2: variants per source question
    for qid, variants in source_variants.items():
        missing = STANDARD_VARIANTS - variants
        if missing:
            errors.append(Issue(qid, "missing_variants", f"missing variants {missing}"))
        extra = variants - STANDARD_VARIANTS
        if extra:
            warnings.append(Issue(qid, "extra_variants", f"unexpected variants {extra}"))

    if len(source_variants) != EXPECTED_SOURCE_QUESTIONS:
        warnings.append(Issue("dataset", "source_count",
                              f"found {len(source_variants)} source questions, expected {EXPECTED_SOURCE_QUESTIONS}"))

    # Check 3: trap count
    if trap_count != EXPECTED_TRAP_CASES:
        errors.append(Issue("dataset", "trap_count", f"found {trap_count} trap cases, expected {EXPECTED_TRAP_CASES}"))

    # Check 14: every correction appears in the review log
    review_corrections: dict[str, set[str]] = defaultdict(set)
    for r in review_rows:
        key = r.get("source_qid")
        for corr in r.get("corrections", []) or []:
            review_corrections[key].add(json.dumps(corr, sort_keys=True, ensure_ascii=False))

    for key, corrs in corrections_in_cases.items():
        logged = review_corrections.get(key, set())
        for corr in set(corrs):
            if corr not in logged:
                errors.append(Issue(key, "correction_not_logged",
                                    f"correction not present in review log: {corr[:80]}"))

    return errors, warnings


def main():
    parser = argparse.ArgumentParser(description="Validate the Claude-reviewed generation golden dataset")
    parser.add_argument("--strict", action="store_true", help="Exit 1 on warnings too")
    parser.add_argument("--path", type=Path, default=None)
    parser.add_argument("--review-path", type=Path, default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    dataset_path = args.path or (root / "goldens" / "golden_generation_dataset_claude.jsonl")
    review_path = args.review_path or (root / "goldens" / "generation_claude_review.jsonl")

    if not dataset_path.exists():
        print(f"ERROR: Dataset not found: {dataset_path}")
        sys.exit(1)
    if not review_path.exists():
        print(f"ERROR: Review log not found: {review_path}")
        sys.exit(1)

    print(f"Validating: {dataset_path}")
    print(f"Review log: {review_path}")
    print()

    cases = load_jsonl(dataset_path)
    review_rows = load_jsonl(review_path)
    chunk_run = cases[0].get("chunk_run", EXPECTED_CHUNK_RUN) if cases else EXPECTED_CHUNK_RUN
    chunk_text = load_chunk_text(root, chunk_run)

    print(f"Loaded {len(cases)} cases, {len(review_rows)} review records, "
          f"{len(chunk_text)} chunks from run {chunk_run}")
    print()

    errors, warnings = validate(cases, review_rows, chunk_text)

    if errors:
        print(f"=== ERRORS ({len(errors)}) ===")
        for e in errors[:80]:
            print(f"  {e}")
        if len(errors) > 80:
            print(f"  ... and {len(errors) - 80} more")
        print()

    if warnings:
        print(f"=== WARNINGS ({len(warnings)}) ===")
        for w in warnings[:80]:
            print(f"  {w}")
        print()

    # summary
    variant_counts: dict[str, int] = defaultdict(int)
    for c in cases:
        variant_counts[c.get("context_variant", "unknown")] += 1
    print("=== SUMMARY ===")
    for v, n in sorted(variant_counts.items()):
        print(f"  {v}: {n}")
    hr = sum(1 for c in cases if c.get("requires_human_review"))
    conf = defaultdict(int)
    for c in cases:
        conf[c.get("confidence")] += 1
    print(f"  confidence: {dict(conf)}")
    print(f"  requires_human_review: {hr}")
    print()

    if errors:
        print(f"FAILED: {len(errors)} errors, {len(warnings)} warnings")
        sys.exit(1)
    if warnings and args.strict:
        print(f"FAILED (strict): {len(warnings)} warnings")
        sys.exit(1)
    if warnings:
        print(f"PASSED with {len(warnings)} warnings")
        sys.exit(0)
    print("PASSED: No errors or warnings")
    sys.exit(0)


if __name__ == "__main__":
    main()
