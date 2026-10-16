# Retrieval Experiments — Lab Notebook

A running log of retrieval experiments. **Newest entry on top.** The raw numbers and
per-query detail live in `eval/results/<run_timestamp>_eval_results.json`; repeated evaluations
of the same run use `_v2`, `_v3`, and so on. This file records the *hypothesis, change, and
takeaway* for each run so the reasoning isn't lost.

## How to use
1. Before a run, write down the **hypothesis** and the **change** you're testing.
2. Run the pipeline + eval (commands below) and copy the headline metrics into the table.
3. Fill in **Observation** and **Next step**. Keep one section per run.

Standard commands:
```powershell
# (re)build chunks + embeddings for a config
uv run python -m ingestion.processor --max-chars <N> --table-max-chars <N>
# evaluate the latest run against the golden set
uv run python .\eval\dense_reteriver_eval.py
# compare every logged run
uv run python .\eval\dense_reteriver_eval.py --compare
```

Fixed context (change only when noted): golden set = `goldens/golden_retriver_dataset.jsonl`
(60 queries, 56 positive / 4 negative) · model = `BAAI/bge-base-en-v1.5` · metrics reported @ k=5.

---

<!-- ===== TEMPLATE: copy this block for each new experiment =====

## EXP-NNN — <short title>
- **Date:** YYYY-MM-DD
- **Run dir:** data/chunks/<timestamp>
- **Result file:** eval/results/<run_timestamp>_eval_results.json (or `_vN` for a repeat)
- **Hypothesis:** <what you expect to happen and why>
- **Change vs baseline:** <the one thing you changed; link the EXP it builds on>

| MAX_CHARS | TBL_CHARS | Chunks | Hit@5 | MRR@5 | Recall@5 | Prec@5 | nDCG@5 |
|---|---|---|---|---|---|---|---|
| | | | | | | | |

- **Observation:** <did the hypothesis hold? notable category/difficulty breakdowns?>
- **Next step:** <what to try next>

================================================================= -->

## EXP-009 — Ettin candidate-depth tuning
- **Date:** 2026-10-16
- **Run dir:** data/chunks/2026-10-16_17-03-28
- **Result files:**
  - `eval/results/2026-10-16_20-42-50_rerank_a0.85_ettin-reranker-150m-v1_eval_results.json` (`retrieve_k=20`)
  - `eval/results/2026-10-16_20-45-05_rerank_a0.85_ettin-reranker-150m-v1_eval_results.json` (`retrieve_k=30`)
  - `eval/results/2026-10-16_20-31-18_rerank_a0.85_ettin-reranker-150m-v1_eval_results.json` (`retrieve_k=50`)
- **Hypothesis:** A smaller candidate pool will preserve most of Ettin's gain while reducing
  cross-encoder work and latency.
- **Change vs EXP-008:** Keep hybrid alpha and reranker fixed; vary only `retrieve_k`.

| Retrieve-k | Hit@5 | MRR@5 | Recall@5 | Prec@5 | nDCG@5 |
|---:|---:|---:|---:|---:|---:|
| 20 | 0.750 | 0.574 | 0.598 | 0.211 | 0.526 |
| 30 | 0.750 | 0.572 | 0.604 | 0.214 | 0.530 |
| **50** | **0.786** | **0.587** | **0.640** | **0.225** | **0.546** |

- **Observation:** The hypothesis **did not hold for quality**. Moving from 20 to 30 candidates
  produced only a small gain, while 50 improved every metric. Relevant evidence frequently lies
  between candidate ranks 31 and 50, and Ettin successfully promotes it into the final top five.
- **Decision:** Use `retrieve_k=50` for the quality-first configuration. Test 40 only if measured
  latency requires a compromise; otherwise additional candidate-depth tuning is unnecessary.
- **Selected pipeline:** 1200/900 chunks → BGE dense + BM25 hybrid (`alpha=0.85`) → top 50
  candidates → `cross-encoder/ettin-reranker-150m-v1` → final top 5.
- **Next step:** Fix empty category/difficulty breakdown reporting, measure warm-run latency, and
  confirm the selected configuration on an untouched held-out query set.

---

## EXP-008 — Cross-encoder reranker comparison
- **Date:** 2026-10-16
- **Run dir:** data/chunks/2026-10-16_17-03-28
- **Result files:**
  - `eval/results/2026-10-16_20-19-20_rerank_a0.8_ms-marco-MiniLM-L-6-v2_eval_results.json`
  - `eval/results/2026-10-16_20-24-59_rerank_a0.8_bge-reranker-base_eval_results.json`
  - `eval/results/2026-10-16_20-31-18_rerank_a0.85_ettin-reranker-150m-v1_eval_results.json`
- **Hypothesis:** Reranking a high-recall top-50 hybrid candidate pool will improve top-five
  ordering; models may differ substantially in their fit for SEC financial passages.
- **Change vs EXP-007:** Fix `alpha=0.85` and `retrieve_k=50`; vary only the cross-encoder.

| Reranker | Hit@5 | MRR@5 | Recall@5 | Prec@5 | nDCG@5 |
|---|---:|---:|---:|---:|---:|
| MiniLM-L-6-v2 | 0.661 | 0.468 | 0.521 | 0.175 | 0.431 |
| BGE reranker base | 0.750 | 0.509 | 0.577 | 0.196 | 0.466 |
| **Ettin reranker 150M** | **0.786** | **0.587** | **0.640** | **0.225** | **0.546** |

- **Observation:** Ettin was the only reranker to improve every metric over the hybrid input.
  MiniLM and BGE reordered good candidates incorrectly and were rejected. Against hybrid
  `alpha=0.85`, Ettin raised Hit 0.750→0.786, MRR 0.558→0.587, Recall 0.598→0.640, Precision
  0.204→0.225, and nDCG 0.520→0.546.
- **Decision:** Select `cross-encoder/ettin-reranker-150m-v1`.
- **Next step:** Tune candidate depth to determine whether 20 or 30 candidates retain the gain
  with less cross-encoder work (→ EXP-009).

---

## EXP-007 — Dense-heavy alpha and candidate-pool tuning
- **Date:** 2026-10-16
- **Run dir:** data/chunks/2026-10-16_17-03-28
- **Hypothesis:** BM25 is useful only as a small lexical contribution; a dense-heavy hybrid can
  improve ranking while retaining enough relevant candidates for reranking.
- **Change vs EXP-006:** Sweep hybrid alpha, then measure candidate recall at k=20 and k=50.

| Alpha | Hit@5 | MRR@5 | Recall@5 | Prec@5 | nDCG@5 | Recall@20 | Recall@50 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.80 | 0.750 | 0.540 | **0.616** | **0.207** | 0.509 | **0.771** | **0.932** |
| **0.85** | 0.750 | 0.558 | 0.598 | 0.204 | 0.520 | 0.759 | **0.932** |
| 0.90 | 0.750 | **0.575** | 0.589 | 0.200 | **0.524** | 0.759 | 0.911 |
| Dense (1.00) | 0.750 | 0.551 | 0.601 | **0.207** | 0.511 | — | — |

- **Observation:** `alpha=0.90` gave the best direct top-five ranking, while `0.80` maximized
  recall. At candidate depth 50, `0.80` and `0.85` tied on Recall (0.932) and Hit (0.982), but
  `0.85` had better candidate nDCG (0.636 vs 0.625). Alpha 0.90 missed more candidates at 50.
- **Decision:** Use `alpha=0.85` as the reranker candidate generator and begin with
  `retrieve_k=50`; ranking inside that pool will be replaced by the cross-encoder.
- **Next step:** Compare reranker models under the same alpha and candidate pool (→ EXP-008).

---

## EXP-006 — Dense vs BM25 vs equal-weight hybrid
- **Date:** 2026-10-16
- **Run dir:** data/chunks/2026-10-16_17-03-28
- **Result files:**
  - `eval/results/2026-10-16_20-00-51_dense_eval_results.json`
  - `eval/results/2026-10-16_20-01-13_bm25_eval_results.json`
  - `eval/results/2026-10-16_20-01-39_hybrid_a0.5_eval_results.json`
- **Hypothesis:** BM25's exact-term matching will complement semantic retrieval, so an equal-weight
  dense/BM25 hybrid will outperform the dense baseline on the frozen 1200/900 chunk run.
- **Change vs EXP-002:** Keep chunks and embeddings fixed; compare retrieval strategy only.

| Mode | Alpha | Hit@5 | MRR@5 | Recall@5 | Prec@5 | nDCG@5 |
|---|---:|---:|---:|---:|---:|---:|
| Dense | — | **0.750** | **0.551** | **0.601** | **0.207** | **0.511** |
| BM25 | — | 0.536 | 0.365 | 0.443 | 0.143 | 0.361 |
| Hybrid | 0.5 | 0.625 | 0.437 | 0.515 | 0.175 | 0.428 |

- **Observation:** The hypothesis **did not hold**. Dense retrieval won every metric. BM25 was
  substantially weaker, and assigning it 50% of the hybrid score pulled strong dense results
  downward: hybrid nDCG fell from 0.511 to 0.428, MRR from 0.551 to 0.437, and Recall from 0.601
  to 0.515. This rejects equal weighting, but does not yet reject a dense-heavy hybrid. Category
  and difficulty sections were unexpectedly empty in this evaluator output and should be fixed
  before using subgroup performance to choose the final configuration.
- **Next step:** Tune `alpha` from 0.0 to 1.0 on this same frozen run. Keep hybrid only if a
  dense-heavy weight beats dense nDCG@5=0.511 without materially reducing Recall@5=0.601;
  otherwise use dense (`alpha=1.0`) as the candidate generator for cross-encoder reranking.

---

## EXP-005 — Push budgets smaller (1000 / 900)
- **Date:** 2026-10-16
- **Run dir:** data/chunks/2026-10-16_19-18-24
- **Result file:** eval/results/2026-10-16_19-18-24_eval_results.json
- **Hypothesis:** If 1200 beat 1400, maybe even smaller (1000) keeps improving ranking quality.
- **Change vs EXP-002:** `MAX_CHARS` 1200 → 1000 (TBL_CHARS unchanged at 900).

| MAX_CHARS | TBL_CHARS | Chunks | Hit@5 | MRR@5 | Recall@5 | Prec@5 | nDCG@5 |
|---|---|---|---|---|---|---|---|
| 1000 | 900 | 1544 | 0.732 | 0.525 | 0.583 | 0.196 | 0.490 |

- **Observation:** Hypothesis **did not hold** — going below 1200 regressed vs EXP-002 on every
  metric (nDCG 0.511→0.490, Hit 0.750→0.732, Recall 0.601→0.583). Recall@5 is the lowest of all
  runs: smaller chunks carry less content, so fewer relevant elements fit in the top-5 (+147
  chunks vs EXP-002, yet worse recall). Still slightly ahead of the 1400 baseline on MRR/nDCG.
- **Next step:** Stop shrinking. 1200/900 is the peak; move on from chunk size to other levers
  (query prefix already set, try a reranker or hybrid lexical+dense for the weak categories).

---

## EXP-004 — Baseline reproducibility check (1400 / 1000)
- **Date:** 2026-10-16
- **Run dir:** data/chunks/2026-10-16_19-17-09
- **Result file:** eval/results/2026-10-16_19-17-09_eval_results.json
- **Hypothesis:** Re-running the exact baseline config should reproduce EXP-001 — confirms the
  parse→chunk→embed→eval pipeline is deterministic.
- **Change vs EXP-001:** none (same 1400 / 1000); fresh run to verify reproducibility.

| MAX_CHARS | TBL_CHARS | Chunks | Hit@5 | MRR@5 | Recall@5 | Prec@5 | nDCG@5 |
|---|---|---|---|---|---|---|---|
| 1400 | 1000 | 1254 | 0.732 | 0.521 | 0.592 | 0.200 | 0.476 |

- **Observation:** Identical to EXP-001 on all five metrics and chunk count (1254). Pipeline is
  deterministic, so metric deltas between experiments are real signal, not run-to-run noise.
- **Next step:** Safe to compare configs directly without averaging repeats.

---

## EXP-003 — Mild reduction (1300 / 1000)
- **Date:** 2026-10-16
- **Run dir:** data/chunks/2026-10-16_19-14-00
- **Result file:** eval/results/2026-10-16_19-14-00_eval_results.json
- **Hypothesis:** A gentler cut than EXP-002 (1300 instead of 1200) might capture most of the
  benefit while keeping chunk count low.
- **Change vs baseline (EXP-001):** `MAX_CHARS` 1400 → 1300 (TBL_CHARS unchanged at 1000).

| MAX_CHARS | TBL_CHARS | Chunks | Hit@5 | MRR@5 | Recall@5 | Prec@5 | nDCG@5 |
|---|---|---|---|---|---|---|---|
| 1300 | 1000 | 1307 | 0.732 | 0.494 | 0.598 | 0.204 | 0.478 |

- **Observation:** Hypothesis **did not hold** — 1300 is essentially flat vs the 1400 baseline on
  nDCG (0.478 vs 0.476) and actually **worse on MRR** (0.494 vs 0.521), i.e. the first relevant
  result lands lower on average. Clearly below EXP-002 (1200/900). Shows the gain from EXP-002 is
  not just "smaller is better" — the 900 table budget likely matters too, since EXP-003 kept
  TBL_CHARS at 1000.
- **Next step:** Isolate the table-budget effect — try 1200/1000 vs 1200/900 to see how much of
  EXP-002's win comes from the smaller table chunks specifically.

---

## EXP-002 — Smaller chunk + table budgets
- **Date:** 2026-10-16
- **Run dir:** data/chunks/2026-10-16_17-03-28
- **Result file:** eval/results/2026-10-16_17-03-28_eval_results.json
- **Hypothesis:** Tighter chunks make each embedding more topically focused, improving ranking
  quality (nDCG/MRR) at the cost of more chunks.
- **Change vs baseline (EXP-001):** `MAX_CHARS` 1400 → 1200, `TABLE_MAX_CHARS` 1000 → 900.

| MAX_CHARS | TBL_CHARS | Chunks | Hit@5 | MRR@5 | Recall@5 | Prec@5 | nDCG@5 |
|---|---|---|---|---|---|---|---|
| 1200 | 900 | 1397 | 0.750 | 0.551 | 0.601 | 0.207 | 0.511 |

- **Observation:** Every headline metric improved over EXP-001 (nDCG +0.035, MRR +0.030,
  Hit +0.018, Recall +0.009) for +143 chunks. Hypothesis held. By category, weakest areas are
  `cross_document` (Hit 0.50 / MRR 0.25) and `accounting_policy` (Hit 0.57 / MRR 0.33); strongest
  are `business_description` and `trend_explanation`. Recall drops on `hard` queries (0.47) vs
  easy/medium (0.67).
- **Next step:** Probe even smaller budgets (e.g. 1000 / 800) to find where gains plateau or
  recall starts dropping; separately investigate the cross-document and accounting-policy misses
  via `--analyze-failures`.

---

## EXP-001 — Baseline (default chunking)
- **Date:** 2026-10-16
- **Run dir:** data/chunks/2026-10-16_16-01-51
- **Result file:** eval/results/2026-10-16_16-01-51_eval_results.json
- **Hypothesis:** Establish a baseline with the default chunker constants.
- **Change vs baseline:** n/a (this is the baseline).

| MAX_CHARS | TBL_CHARS | Chunks | Hit@5 | MRR@5 | Recall@5 | Prec@5 | nDCG@5 |
|---|---|---|---|---|---|---|---|
| 1400 | 1000 | 1254 | 0.732 | 0.521 | 0.592 | 0.200 | 0.476 |

- **Observation:** Baseline reference point for all future experiments.
- **Next step:** Reduce chunk sizes to test whether tighter chunks improve ranking (→ EXP-002).
