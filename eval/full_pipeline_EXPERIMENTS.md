# Full Pipeline (RAG Triad) Experiments — Lab Notebook

A running log of end-to-end RAG triad experiments. **Newest entry on top.** Raw per-case
outputs and aggregate summaries live in `eval/rag_triad_results/<run_id>/`:

- `cases.jsonl` — per-query input, generation output, retrieved chunks, and all triad metrics;
- `manifest.json` — both golden paths, generation config, retrieval config, prompt hash, environment;
- `summary.json` — triad headlines, overall aggregates, breakdowns by category/difficulty/document.

This file records the *hypothesis, controlled change, results, observation, and decision* so the
reasoning is not lost.

## How to use

1. Before a run, record the **hypothesis** and the **single variable changed**.
2. Keep the retrieval golden, generation golden, context_k, prompt hash, temperature, maximum
   tokens, and fact threshold fixed when comparing models.
3. Run experiments sequentially when using local Ollama models; concurrent runs compete for
   GPU/CPU/RAM and invalidate latency comparisons.
4. Copy headline metrics below, inspect per-case failures, then record the decision and next step.

Standard commands:

```powershell
# Run full triad evaluation with defaults
uv run python eval/full_pipeline_triad_eval.py

# Run with specific model/provider
uv run python eval/full_pipeline_triad_eval.py `
  --provider ollama `
  --model <model>

# Quick subset for debugging
uv run python eval/full_pipeline_triad_eval.py --limit 5 --verbose

# Compare saved runs
uv run python eval/full_pipeline_triad_eval.py --compare
```

Fixed context unless an experiment explicitly says otherwise:

- Retrieval golden: `goldens/golden_retriver_dataset.jsonl` (60 queries, 56 positive / 4 negative).
- Generation golden: `goldens/golden_generation_dataset.jsonl` (oracle variants for answer expectations).
- Chunk run: `data/chunks/2026-10-16_17-03-28`.
- Retrieval pipeline: hybrid `alpha=0.85`, `retrieve_k=50`, reranker `cross-encoder/ettin-reranker-150m-v1`.
- Context_k: `5`.
- Prompt version: `v1_strict_grounded`.
- Temperature: `0.0`.
- Maximum output tokens: `1024`.
- Required-fact threshold: `0.6`.

---

<!-- ===== TEMPLATE: copy this block for each new experiment =====

## TRIAD-EXP-NNN — <short title>
- **Date:** YYYY-MM-DD
- **Result directory:**
  `eval/rag_triad_results/<run_id>`
- **Hypothesis:** <what you expect and why>
- **Change vs baseline:** <the one controlled variable changed>
- **Fixed context:** <dataset, context_k, prompt hash, threshold>

Triad headlines:

| Model/config | Cases | Context (nDCG) | Grounded (faithful) | Answer (correct) | Triad score | Avg. latency |
|---|---:|---:|---:|---:|---:|---:|
| | | | | | | |

Context leg detail:

| Model/config | Hit@k | Recall@k | MRR@k | Precision@k |
|---|---:|---:|---:|---:|
| | | | | |

Groundedness leg detail:

| Model/config | Cite valid | Cite precision | Cite recall |
|---|---:|---:|---:|
| | | | |

Answer leg detail:

| Model/config | Value cov. | Fact cov. | Abstention | Hallucinated | False abstain |
|---|---:|---:|---:|---:|---:|
| | | | | | |

- **Observation:** <did the hypothesis hold? which leg is the bottleneck?>
- **Decision:** <which configuration is selected and why?>
- **Failure analysis:** <which categories/difficulties failed and likely cause?>
- **Next step:** <one controlled follow-up experiment>

================================================================= -->

## TRIAD-EXP-001 — Qwen 2.5 7B baseline on full 60-query set

- **Date:** 2026-10-16
- **Result directory:**
  `eval/rag_triad_results/2026-10-16_22-53-55_ollama_qwen2.5-7b-instruct_latest`
- **Hypothesis:** The end-to-end triad will surface a different bottleneck than the component
  evals. Groundedness should remain high (consistent with GEN-EXP-001's 94% numeric
  faithfulness), but answer correctness may drop below the isolated generation eval because real
  retrieval introduces imperfect context.
- **Change vs baseline:** First triad run — establishes the end-to-end baseline for the
  selected retrieval pipeline (EXP-009) and generation model (GEN-EXP-001).
- **Fixed context:** All 60 retrieval golden queries, both golden sets, context_k=5,
  `v1_strict_grounded` prompt, temperature 0, max tokens 1024, fact threshold 0.6.

Triad headlines:

| Model | Cases | Context (nDCG) | Grounded (faithful) | Answer (correct) | Triad score | Avg. latency |
|---|---:|---:|---:|---:|---:|---:|
| **Qwen 2.5 7B Instruct** | 60 | **50.9%** | **94.2%** | **38.3%** | **61.2%** | 4.00s |

Context Relevance leg:

| Model | Hit@5 | Recall@5 | MRR@5 | Precision@5 |
|---|---:|---:|---:|---:|
| Qwen 2.5 7B Instruct | 73.3% | 59.7% | 54.8% | 21.0% |

Groundedness leg:

| Model | Cite valid | Cite precision | Cite recall |
|---|---:|---:|---:|
| Qwen 2.5 7B Instruct | 98.3% | 36.6% | 43.5% |

Answer Relevance leg:

| Model | Value cov. | Fact cov. | Abstention | Hallucinated | False abstain |
|---|---:|---:|---:|---:|---:|
| Qwen 2.5 7B Instruct | 41.2% | 62.9% | 100.0% | 0.0% | 0.0% |

Token and runtime totals:

| Model | Prompt tokens | Completion tokens | Total latency |
|---|---:|---:|---:|
| Qwen 2.5 7B Instruct | 119,731 | 8,346 | 239.9s |

Category breakdown (triad perspective):

| Category | Queries | Context nDCG | Faithful | Correct | Weakest leg |
|---|---:|---:|---:|---:|---|
| exact_numeric | 10 | 51.6% | 100.0% | **90.0%** | Context |
| trend_explanation | 8 | **87.5%** | 100.0% | 62.5% | Answer |
| business_description | 7 | 73.5% | — | 57.1% | Answer |
| risk_factor | 8 | 73.6% | 58.3% | 25.0% | Answer + Grounded |
| accounting_policy | 7 | 26.3% | 100.0% | 28.6% | Context |
| cross_section | 6 | 46.5% | 88.9% | 16.7% | Answer |
| cross_document | 6 | 27.8% | 100.0% | 0.0% | Context + Answer |
| table_structure | 4 | 27.2% | 100.0% | 0.0% | Context + Answer |
| negative | 4 | 0.0% | 81.3% | 0.0% | Context (by design) |

Difficulty breakdown:

| Difficulty | Queries | Context nDCG | Faithful | Correct |
|---|---:|---:|---:|---:|
| easy | 12 | 59.0% | 100.0% | **83.3%** |
| medium | 28 | 54.5% | 94.4% | 35.7% |
| hard | 20 | 41.2% | 88.0% | 15.0% |

Document breakdown:

| Document | Queries | Context nDCG | Faithful | Correct |
|---|---:|---:|---:|---:|
| AAPL_10-K_2025 | 23 | 60.8% | 84.0% | 43.5% |
| TSLA_10-K_2025 | 18 | 62.4% | 100.0% | 38.9% |
| MSFT_10-K_2026 | 13 | 28.4% | 100.0% | 46.2% |
| cross-document | 6 | 27.8% | 100.0% | 0.0% |

- **Observation:** The hypothesis held. The triad reveals a clear bottleneck hierarchy:
  1. **Answer Relevance (38.3%)** is the weakest leg. The model gets easy exact-numeric questions
     right (90%) but struggles with cross-section (16.7%), cross-document (0%), and table-structure
     (0%) categories. Hard queries drop to 15% correct.
  2. **Context Relevance (50.9% nDCG)** is the second bottleneck. Accounting policy (26.3%),
     cross-document (27.8%), and table-structure (27.2%) categories have poor retrieval. MSFT
     filing retrieval (28.4% nDCG) is notably weaker than AAPL (60.8%) or TSLA (62.4%).
  3. **Groundedness (94.2%)** is the strongest leg — the model rarely fabricates numbers. The
     exception is risk_factor (58.3%), where narrative answers include numbers not in cited text.
  
  Compared to isolated generation eval (GEN-EXP-001, 53.3% correct on oracle_plus_distractors),
  end-to-end correctness dropped to 38.3%. The 15-point gap quantifies the cost of real retrieval
  versus frozen golden context. The abstention behavior is perfect (100% correct, 0% hallucination,
  0% false abstention), confirming GEN-EXP-003's finding that Qwen handles insufficient context
  well.

- **Decision:** This is the baseline. The triad score (61.2%) and per-leg breakdown provide the
  reference point for all subsequent experiments. The three priority improvement areas are:
  1. **Retrieval for MSFT and cross-document queries** — nDCG is half of AAPL/TSLA.
  2. **Answer generation for multi-fact categories** — cross_section, table_structure, and
     risk_factor categories need prompt or post-processing improvements.
  3. **Citation precision/recall** — 36.6% precision and 43.5% recall mean the model cites real
     sources but not the gold-labeled ones.

- **Failure analysis by category:**
  - **cross_document (0% correct):** Retrieval only finds 38.9% of relevant chunks (recall), and
    `doc_id=None` queries search all filings without a document filter. The LLM cannot synthesize
    cross-filing comparisons from partial context.
  - **table_structure (0% correct):** Retrieval nDCG is 27.2%; the right table chunks are not
    surfacing. Even when they do, the model struggles to extract multi-cell values.
  - **accounting_policy (28.6% correct):** Context nDCG 26.3% — retrieval is the primary failure
    mode. Accounting policy text tends to be long-form narrative that fragments across chunks.
  - **risk_factor (25% correct, 58.3% faithful):** Retrieval is decent (73.6% nDCG), but the
    model includes numbers from explanation that aren't in the cited sources — the only category
    with a groundedness problem.
  - **negative (0% correct by design):** These 4 queries have no relevant chunks in the corpus.
    Context metrics are 0 by design. The model correctly abstains (100% abstention) but
    `answer_correct` is 0 because the scoring treats non-answerable queries as needing abstention,
    and the current `answer_correct` logic requires `predicted_abstention == True` which is exactly
    what happens — this 0% is a scoring artifact from how `answer_correct` interacts with the
    negative query gold labels (they have `answerable=True` from the oracle join, but retrieve no
    relevant context). This needs investigation.

- **Next step:** Run the same 60 queries with Llama 3.2 to establish the model comparison on the
  full pipeline, then investigate the negative-query scoring artifact.

---
