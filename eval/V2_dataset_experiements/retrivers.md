# Retriever Experiments (golden v2)

## Common setup

| Item | Value |
|---|---|
| Golden set | `goldens/golden_retriever_dataset_v2.jsonl` (60 queries: 56 positive, 4 negative) |
| Evaluator | `eval/hybrid_rerank_eval.py`, metrics @k=5, results in `eval/retriver_results/` |
| Embedding model | `BAAI/bge-base-en-v1.5` |
| Scoring | Each relevant item credited once (equivalent copies grouped), `doc_id`-aware; precision = relevant chunks / k |

Notes:
- All earlier results (golden v1 / old scoring) are archived in `eval/results/v1/` and are **not comparable**.
- One query = 0.018 Hit@5 (1/56). Differences below ~0.02 are within noise.

---

## Exp 1 - Chunk size (dense only)

**Hypothesis:** Smaller chunks give more focused embeddings (better precision/MRR); larger chunks keep
more context together (better recall). There is a middle size that balances both.

**Setup:** `--mode dense` on every run in `data/chunks/`.

| MaxCh / TblCh | Chunks | Hit@5 | MRR@5 | Recall@5 | Prec@5 | nDCG@5 |
|---|---|---|---|---|---|---|
| 1000 / 900 | 1544 | **0.946** | 0.793 | 0.676 | **0.375** | 0.656 |
| **1200 / 900** | 1397 | **0.946** | **0.804** | 0.689 | **0.375** | **0.659** |
| 1300 / 1000 | 1307 | 0.929 | 0.780 | 0.685 | 0.354 | 0.646 |
| 1400 / 1000 | 1254 | 0.929 | 0.780 | **0.694** | 0.354 | 0.639 |

**Observations:**
- 1200 leads on MRR, precision, nDCG and ties on Hit; recall is within 0.005 of the best.
- Gaps are small (Hit differs by 1 query): chunk size is a weak lever for this corpus.
- Category trade-off: 1400 is better on risk_factor (0.72 vs 0.62 recall) and exact_numeric
  (0.83 vs 0.78); 1200 is better on trend_explanation (0.77 vs 0.69) and cross_document (0.33 vs 0.27).
- The two 1400/1000 runs are identical, so chunking is deterministic.

**Decision:** Keep **1200 / 900** (`2026-10-16_17-03-28`) for all following experiments.

---

## Exp 2 - Dense vs BM25

**Hypothesis:** Dense wins on paraphrased/conceptual questions; BM25 helps exact terms
(company names, line items, numbers) but is weaker overall on natural-language queries.

**Setup:** 1200/900 run, `--mode dense` vs `--mode bm25`.

| Mode | Hit@5 | MRR@5 | Recall@5 | Prec@5 | nDCG@5 |
|---|---|---|---|---|---|
| **dense** | **0.946** | **0.804** | **0.689** | **0.375** | **0.659** |
| bm25 | 0.732 | 0.571 | 0.506 | 0.307 | 0.463 |

**Observations:**
- BM25 is far behind on every metric (-0.21 Hit, -0.23 MRR, -0.18 Recall).
- BM25 fails mostly on exact_numeric (Hit 0.40) and cross_document (Hit 0.33): numbers in tables
  and multi-company questions don't share enough exact words with the query.
- BM25 is competitive only on table_structure and trend_explanation (Hit 1.00).

**Decision:** Dense is the primary signal. BM25 is only worth keeping as a low-weight lexical
signal inside hybrid (Exp 3).

---

## Exp 3 - Hybrid alpha sweep

**Hypothesis:** Adding a small BM25 weight to dense recovers keyword-heavy misses without hurting
ranking; too much BM25 (alpha 0.5) drags results toward BM25's weaker quality.

**Setup:** 1200/900 run, `--mode hybrid --alpha {0.5, 0.8, 0.85, 0.9}`
(score = alpha * dense + (1 - alpha) * BM25, both normalized). Dense-only shown for reference.

| Mode | Alpha | Hit@5 | MRR@5 | Recall@5 | Prec@5 | nDCG@5 | Recall@10 |
|---|---|---|---|---|---|---|---|
| dense | - | 0.946 | 0.804 | 0.689 | 0.375 | 0.659 | 0.813 |
| hybrid | 0.50 | 0.875 | 0.697 | 0.653 | 0.379 | 0.578 | 0.779 |
| hybrid | **0.80** | **0.964** | 0.765 | **0.727** | **0.411** | 0.656 | 0.814 |
| hybrid | **0.85** | 0.946 | 0.793 | 0.704 | 0.400 | **0.662** | **0.818** |
| hybrid | 0.90 | 0.946 | **0.805** | 0.693 | 0.393 | **0.662** | 0.810 |

**Observations:**
- alpha 0.5 is worse than dense alone, which confirms BM25 should only be a minor signal.
- alpha 0.80 gives the best Hit, Recall@5 and Precision (+0.038 recall, +0.036 precision vs dense),
  but the first relevant chunk ranks lower (MRR 0.765 vs 0.804).
- alpha 0.85-0.90 keep dense-level MRR/nDCG with a smaller recall gain.
- Where hybrid 0.80 helps most vs dense (recall): cross_section 0.55 vs 0.46, exact_numeric
  0.83 vs 0.78, risk_factor 0.68 vs 0.62.
- Recall@10 is flat (0.81-0.82) across 0.8-0.9: alpha mostly reorders the top 10 rather than
  finding new evidence.

**Decision:** Hybrid beats dense; use **alpha in 0.80-0.85**.
- Without a reranker: 0.80 (best recall/hit/precision at k=5).
- With a reranker: the reranker fixes ordering, so the final pick between 0.80 and 0.85 is made in
  the reranker experiment (test both).

---

## Exp 4 - Cross-encoder reranker

**Hypothesis:** A cross-encoder reading query and chunk together ranks better than hybrid scores,
so reranking the hybrid top-50 should raise MRR/nDCG (ordering) and also Recall@5 (better chunks
move into the top 5). Larger/newer rerankers should help more.

**Setup:** 1200/900 run, `--mode rerank --retrieve-k 50`, rerankers MiniLM-L-6-v2,
bge-reranker-base, ettin-reranker-150m-v1, each on hybrid alpha 0.80 and 0.85.

| Mode | Reranker | Hit@5 | MRR@5 | Recall@5 | Prec@5 | nDCG@5 | Recall@10 |
|---|---|---|---|---|---|---|---|
| hybrid 0.80 | - (baseline) | 0.964 | 0.765 | 0.727 | 0.411 | 0.656 | 0.814 |
| rerank | ms-marco-MiniLM-L-6-v2 | 0.929 | 0.715 | 0.653 | 0.364 | 0.596 | 0.810 |
| rerank | bge-reranker-base | 0.911 | 0.698 | 0.660 | 0.361 | 0.605 | 0.769 |
| rerank | **ettin-reranker-150m-v1** | **0.964** | **0.887** | **0.743** | **0.479** | **0.746** | **0.862** |

Results are identical for alpha 0.80 and 0.85 with every reranker (table shows one row each).

**Observations:**
- ettin is the only reranker that helps: vs hybrid, MRR +0.122, nDCG +0.090, Precision +0.068,
  Recall@5 +0.016, Recall@10 +0.048. Hit@1 rises from 0.625 to 0.821.
- MiniLM and bge-reranker make every metric *worse* than no reranker; they are not suited to
  long financial chunks / tables.
- Alpha no longer matters: the two candidate pools differ by only 1-4 low-ranked chunks in 49/60
  queries, and the reranker never promotes those, so top-5 lists are the same.
- ettin gains by category (Recall@5 vs hybrid 0.80): exact_numeric 0.92 vs 0.83, cross_document
  0.52 vs 0.34, table_structure 0.88 vs 0.75, business_description 0.96 vs 0.88.
- ettin regression: trend_explanation 0.62 vs 0.83 (Hit 0.75 vs 1.00, i.e. 2 queries lost) and
  accounting_policy 0.74 vs 0.83. Worth inspecting those queries.

**Decision:** Use **hybrid (alpha 0.85) + ettin-reranker-150m-v1, retrieve-k 50**. This matches the
existing production config, now confirmed on golden v2. Alpha 0.85 is kept only for consistency
(0.80 gives the same results).

Final v2 retriever score: Hit@5 0.964, MRR@5 0.887, Recall@5 0.743, Prec@5 0.479, nDCG@5 0.746.

---

## Exp 4b - ettin regression analysis (trend_explanation / accounting_policy)

**Question:** Why does ettin lower recall on trend_explanation (0.83 -> 0.62) and accounting_policy
(0.83 -> 0.74) when it improves everything else?

**Method:** Per-query Recall@5 from the saved results (hybrid 0.80 vs rerank 0.85 + ettin), then the
full reranked top-50 for the worst queries, showing where each labeled chunk lands.

**Queries where ettin lowers recall:**

| Query | Category | Hybrid | ettin | Question |
|---|---|---|---|---|
| q012 | trend | 1.00 | 0.50 | Apple Services net sales increase |
| q014 | trend | 0.50 | **0.00** | MSFT revenue growth factors |
| q016 | trend | 0.67 | **0.00** | MSFT Intelligent Cloud growth |
| q017 | trend | 1.00 | 0.50 | MSFT gross margin % decrease |
| q029 | accounting | 0.67 | 0.33 | MSFT cloud revenue recognition |
| q033 | accounting | 1.00 | 0.67 | MSFT OpenAI accounting |
| q034 | accounting | 1.00 | 0.67 | MSFT standalone selling price |

6 of 7 are Microsoft queries; both total misses are MSFT MD&A trend questions.

**Drill-down:**
- **q014** (labeled #383 rel 3, #381 rel 2): ettin top 5 = cash from operations (wrong), revenue
  by geography x2 (weak), Productivity & Business Processes segment paragraphs #395-400 and Server
  products / Azure paragraphs #408-413 (both genuinely explain revenue growth). The chunk with #383
  is #6; it opens with the non-GAAP / OpenAI paragraph #382. Table #381 is #25.
- **q016** (labeled #408 rel 3, #407 and #409 rel 2): ettin #1 is chunk #382-386, which contains
  #383 "Intelligent Cloud revenue increased driven by Azure" (a correct answer, unlabeled for this
  query). Then MS Cloud gross margin, revenue by geography x2, metric-definitions table (wrong).
  The real answer #408 is #10 (hybrid had it #3). #407 "Intelligent Cloud: revenue +$31.5B" sits at
  the end of the Productivity chunk #402-407 and is #16.
- **q012**: the answer paragraph #290 is ranked #1; only the supporting table #279 fell to #7.
  Not a real problem.

**Root causes (three, mixed together):**
1. **Golden gaps:** Microsoft segment paragraphs legitimately explain total revenue growth (q014),
   and #383 directly answers q016, but they are not labeled. Part of the "regression" is ettin being
   penalized for correct choices.
2. **Chunk boundaries cross subsections:** Microsoft's MD&A uses one-line paragraphs, and the
   chunker packs them across subsection headings. The answer #383 shares a chunk that starts with
   unrelated non-GAAP text; the "Intelligent Cloud" sub-heading and revenue line #406-407 are glued to
   the end of the Productivity chunk, separated from #408.
3. **Genuine ettin errors:** cash-from-operations and the geographic revenue table rank top only on
   shared words ("increased ... revenue ... fiscal year 2026").

**Options (all recorded; none applied yet:**

- **A. Fix the labels (golden v2.1).** Add the segment-result paragraphs as rel 2 for q014
  (#396-397, #407-408, #416-419) and #383 as rel 2 for q016; then re-run the validator and the
  hybrid vs ettin comparison.
  - Pro: removes a measurement error; ettin's true quality is likely higher than reported.
  - Con: changing labels after seeing model output can bias the benchmark toward that model. Only
    add evidence that clearly answers the question on its own, and record the reason per label.
  - Effect: changes golden version -> all experiments would need a re-run to stay comparable.
- **B. Fix the chunker.** Start a new chunk at MD&A subsection headings (segment names such as
  "Productivity and Business Processes", "Intelligent Cloud", "More Personal Computing"), so a
  heading and its paragraphs stay together and unrelated topics are not mixed.
  - Pro: attacks the real cause; likely improves retrieval and generation for all MSFT MD&A queries
    (and any filing with short-paragraph sections).
  - Con: pipeline change; needs new chunk runs + re-embedding, then re-running chunk size, alpha and
    reranker experiments.
- **C. Document and move on.** ettin still wins overall by a wide margin (MRR +0.12, nDCG +0.09,
  Hit@1 0.625 -> 0.821); treat MSFT MD&A trend questions as a known weakness.
  - Pro: no extra work now; the production choice does not change.
  - Con: the weakness and the label gaps remain; trend_explanation scores stay understated/noisy.

**Status:** C is the current state (this entry). B is the preferred follow-up because it fixes the
cause; A should be done together with B (or before the next golden version) with documented reasons.

