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
uv run python .\eval\dense_retriever_eval.py
# compare every logged run
uv run python .\eval\dense_retriever_eval.py --compare
```

Fixed context (change only when noted): golden set = `goldens/golden_retriever_dataset.jsonl`
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

## EXP-011 — Dense-heavy RRF weight sweep
- **Date:** 2026-10-17
- **Run dir:** data/chunks/2026-10-16_17-03-28
- **Result files:**
  - `eval/results/2026-10-17_19-28-48_rrf_w2-1_k60_eval_results.json`
  - `eval/results/2026-10-17_19-29-10_rrf_w3-1_k60_eval_results.json`
  - `eval/results/2026-10-17_19-29-33_rrf_w4-1_k60_eval_results.json`
  - `eval/results/2026-10-17_19-29-55_rrf_w6-1_k60_eval_results.json`
  - `eval/results/2026-10-17_19-30-41_rrf_rerank_w4-1_k60_ettin-reranker-150m-v1_eval_results.json`
  - `eval/results/2026-10-17_19-35-57_rrf_rerank_w6-1_k60_ettin-reranker-150m-v1_eval_results.json`
  - `eval/results/2026-10-17_19-40-48_rrf_rerank_w6-1_k60_ettin-reranker-150m-v1_eval_results.json` (repeat)
- **Hypothesis:** Equal-weight RRF failed for the same reason as `alpha=0.5` (EXP-006): BM25 gets
  too much influence. Weighting dense ranks more heavily (6:1 ≈ `alpha=0.85`, since 0.85/0.15 ≈ 5.7)
  should match or beat the score-blended hybrid.
- **Change vs EXP-010:** Vary `--dense-weight` (2, 3, 4, 6) with `--bm25-weight 1`; keep
  `rrf_k=60`, `dense_k=bm25_k=50`. Added `--dense-weight` / `--bm25-weight` CLI flags and RRF
  settings to result filenames in `hybrid_rerank_eval.py`.

**Without reranker:**

| Config | Hit@5 | MRR@5 | Recall@5 | Prec@5 | nDCG@5 | Recall@20 | Recall@50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Hybrid α=0.85 (EXP-007) | 0.750 | 0.558 | 0.598 | 0.204 | 0.520 | 0.759 | **0.932** |
| Hybrid α=0.90 (EXP-007) | 0.750 | **0.575** | 0.589 | 0.200 | **0.524** | 0.759 | 0.911 |
| RRF 1:1 (EXP-010) | 0.643 | 0.493 | 0.533 | 0.182 | 0.461 | 0.750 | **0.943** |
| RRF 2:1 | 0.732 | 0.518 | 0.598 | 0.204 | 0.498 | 0.756 | 0.902 |
| RRF 3:1 | 0.732 | 0.548 | 0.589 | 0.200 | 0.509 | 0.744 | 0.902 |
| RRF 4:1 | 0.750 | 0.570 | 0.607 | 0.204 | 0.521 | **0.771** | 0.902 |
| RRF 6:1 | **0.768** | 0.562 | **0.616** | **0.207** | 0.523 | 0.759 | 0.902 |

**With Ettin reranker (`retrieve_k=50`):**

| Config | Hit@5 | MRR@5 | Recall@5 | Prec@5 | nDCG@5 |
|---|---:|---:|---:|---:|---:|
| **Hybrid α=0.85 + Ettin (EXP-009)** | **0.786** | **0.587** | **0.640** | **0.225** | **0.546** |
| RRF 1:1 + Ettin (EXP-010) | **0.786** | **0.587** | 0.631 | 0.221 | 0.542 |
| RRF 4:1 + Ettin | 0.768 | 0.581 | 0.619 | 0.214 | 0.537 |
| RRF 6:1 + Ettin | 0.768 | 0.581 | 0.619 | 0.214 | 0.537 |

- **Observation:** Without a reranker the hypothesis **held**: dense-heavy RRF caught up with
  the tuned hybrid. RRF 6:1 led on Hit, Recall and Precision, α=0.90 led on MRR, and nDCG was
  tied (0.523 vs 0.524), all within about one query out of 56. With the reranker it **did not
  hold**: leaning toward dense cut candidate Recall@50 from 0.943 (1:1) to 0.902, below hybrid
  α=0.85 (0.932). Ettin can't promote chunks missing from the pool, so RRF 4:1 and 6:1 + Ettin
  scored below both RRF 1:1 + Ettin and the production pipeline. 4:1 and 6:1 produced identical
  reranked metrics. The 6:1 rerank repeat reproduced exactly.
- **Decision:** Keep score-blended hybrid `alpha=0.85` + Ettin `retrieve_k=50` as the
  quality-first pipeline. RRF 6:1 is a valid alternative for the no-reranker profile but is not
  meaningfully better than hybrid α=0.90.
- **Why hybrid keeps the edge:** Score blending keeps how far apart chunk scores are; RRF keeps
  only rank order, and alpha is already tuned on this corpus, so RRF's main strength (no score
  calibration needed) adds nothing here. Hybrid also scores every filtered chunk, while RRF only
  sees the union of the dense and BM25 top-50 lists.
- **Next step:** Add paired bootstrap confidence intervals before reading deltas this small.
  Re-test RRF if the embedding model or corpus changes (score scales shift, which hurts min-max
  blending more than RRF). Optionally try `dense_k`/`bm25_k` = 100 with 4:1 weights to recover
  candidate recall.

---

## EXP-010 — Reciprocal rank fusion (BM25S + dense), equal weights
- **Date:** 2026-10-17
- **Run dir:** data/chunks/2026-10-16_17-03-28
- **Result files:**
  - `eval/results/2026-10-17_19-11-45_rrf_eval_results.json`
  - `eval/results/2026-10-17_19-12-42_rrf_rerank_ettin-reranker-150m-v1_eval_results.json`
  - Invalid: `eval/results/2026-10-17_17-10-17_rrf_eval_results.json`. It ran on the latest
    chunk directory (`2026-10-16_19-18-24`, 1000/900, 1544 chunks) instead of the frozen
    1200/900 run, so it is not comparable. Always pass `--run-directory` explicitly.
- **Hypothesis:** Rank-based fusion (`src/retriever/bm25s_rrf_retriever.py`) avoids min-max score
  calibration, so it may beat score blending as a direct ranker and as a reranker candidate pool.
- **Change vs EXP-007 / EXP-009:** Replace score blending with weighted RRF
  (`score = w_d/(rrf_k + rank_dense) + w_b/(rrf_k + rank_bm25)`) using `rrf_k=60`,
  `dense_k=bm25_k=50`, weights 1:1. Same chunks, embeddings, BM25 index and Ettin settings.

| Config | Hit@5 | MRR@5 | Recall@5 | Prec@5 | nDCG@5 | Recall@50 | Hit@50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Hybrid α=0.85 (EXP-007) | **0.750** | **0.558** | **0.598** | **0.204** | **0.520** | 0.932 | 0.982 |
| RRF 1:1 | 0.643 | 0.493 | 0.533 | 0.182 | 0.461 | **0.943** | **1.000** |
| Hybrid α=0.85 + Ettin (EXP-009) | **0.786** | **0.587** | **0.640** | **0.225** | **0.546** | — | — |
| RRF 1:1 + Ettin | **0.786** | **0.587** | 0.631 | 0.221 | 0.542 | — | — |

- **Observation:** As a direct ranker the hypothesis **did not hold**: equal-weight RRF was worse
  on every top-5 metric, matching the `alpha=0.5` failure in EXP-006 (BM25 ranks get too much
  say). As a candidate generator it produced the best pool measured so far (Recall@50 0.943,
  Hit@50 1.000), but after Ettin reranking it only tied the production pipeline on Hit/MRR and
  trailed slightly on Recall/Precision/nDCG (about one relevant chunk across 56 queries). Results
  at k=1 and k=3 were identical to hybrid + Ettin.
- **Next step:** Give dense ranks more weight in RRF (→ EXP-011).

---

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
