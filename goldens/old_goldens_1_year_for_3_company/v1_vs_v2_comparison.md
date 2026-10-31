# v1 vs v2 Golden Comparison

## v1 Best Score (archived)

**Config:** hybrid alpha 0.85 + ettin reranker, retrieve-k 50, 1200/900 chunks  
**File:** `eval/old/results/v1/2026-10-16_20-31-18_rerank_a0.85_ettin-reranker-150m-v1_eval_results.json`

| Metric | v1 |
|---|---|
| Hit@5 | 0.786 (78.6%) |
| MRR@5 | 0.587 |
| Recall@5 | 0.640 |
| Precision@5 | 0.225 |
| nDCG@5 | 0.546 |

## v2 Final Score (current production)

**Config:** hybrid alpha 0.85 + ettin reranker, retrieve-k 50, pseudo-heading chunks (1200/200/900)  
**File:** `eval/retriver_results/2026-10-20_16-35_rerank_a0.85_k50_ettin-reranker-150m-v1_eval_results.json`

| Metric | v2 |
|---|---|
| Hit@5 | 0.982 (98.2%) |
| MRR@5 | 0.905 |
| Recall@5 | 0.761 |
| Precision@5 | 0.482 |
| nDCG@5 | 0.758 |

## Apparent Improvement

| Metric | v1 | v2 | Difference |
|---|---|---|---|
| Hit@5 | 0.786 | 0.982 | +0.196 |
| MRR@5 | 0.587 | 0.905 | +0.318 |
| Recall@5 | 0.640 | 0.761 | +0.121 |
| Precision@5 | 0.225 | 0.482 | +0.257 |
| nDCG@5 | 0.546 | 0.758 | +0.212 |

## Why This Comparison Is Misleading

The v1 → v2 jump looks like a large improvement, but it is **not a fair comparison** because:

### 1. v1 Golden Was Unreliable
The v1 retrieval golden (`goldens/old_golden/golden_retriever_dataset.jsonl`) had ~25 of 60 queries wrong:
- **Invented / miscopied numbers:** e.g., Apple R&D "$33,284M" (real value $34,550M); wrong PP&E totals
- **Answerable "negatives":** 3 of 4 negative queries were actually answerable (Apple dividend, Tesla bitcoin, MSFT employee count)
- **Wrong element orders:** Several MSFT labels off by one (e.g., q014, q017, q043, q048)
- **Unlabeled duplicates:** Only one of several identical copies was labeled (same number in MD&A table, statement, and note)
- **Category violations:** risk_factor answered from Item 1/7, single-section cross_section

Some queries were easier than they should be, others harder. The 78.6% Hit@5 does not reflect true retriever quality.

### 2. v1 Scoring Bugs Inflated Metrics
Two bugs were fixed in the eval scripts:
- **Bug A:** A fact could be credited multiple times (nDCG could exceed 1, recall inflated). Cross-document recall ignored `doc_id`.
- **Bug B:** Precision counted facts instead of chunks (precision@1 could exceed hit@1).

The v1 precision of 0.225 is suspiciously low—this bug would have caused inconsistent precision values across runs.

### 3. v1 Results Are Archived
All v1-scored results are in `eval/old/results/v1/` and are **not comparable** with v2. They should not be used as a baseline.

## The Real Comparison: Within v2

The meaningful comparison is the v2 baseline-to-final progression, which uses the same verified golden and fixed scoring:

| Step | Config | Hit@5 | MRR@5 | Recall@5 | nDCG@5 |
|---|---|---|---|---|---|
| Baseline | dense, 1200/900 chunks | 0.946 | 0.804 | 0.689 | 0.659 |
| + BM25 | hybrid alpha 0.85 | 0.946 | 0.793 | 0.704 | 0.662 |
| + reranker | hybrid 0.85 + ettin, retrieve-k 50 | 0.964 | 0.887 | 0.743 | 0.746 |
| **+ chunking** | **same + pseudo-heading chunks** | **0.982** | **0.905** | **0.761** | **0.758** |
| **Total change vs baseline** | | +0.036 | +0.101 | +0.072 | +0.099 |

This shows the **actual improvement** from the retriever changes, measured correctly.

## What Each v2 Improvement Contributed

| Change | Main effect | Biggest gains |
|---|---|---|
| Hybrid (dense + 15% BM25) | Small recall/precision gain; ordering about the same | exact numbers, risk factors, cross-section |
| ettin cross-encoder reranker | Largest jump; fixes ordering | MRR +0.09, Hit@1 +0.14; exact_numeric, cross_document, tables |
| Pseudo-heading chunking | Fixes MSFT MD&A subsections; zero regressions with reranker | trend_explanation 0.62 → 0.75, MSFT recall +0.06 |

## Conclusion

- **Do not quote v1 scores** as a baseline. They are inflated by golden errors and scoring bugs.
- **Use v2 scores only** for all reporting and comparison.
- The v2 baseline-to-final progression (+0.036 Hit@5, +0.101 MRR) is the true measure of retriever improvement.
