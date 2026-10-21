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

---

## Exp 5 - Chunk boundaries at pseudo-headings (Option B, branch `chunker-experiment`)

**Hypothesis:** The parser emits many sub-headings as short paragraphs ("Intelligent Cloud",
"Greater China", "Revenue Recognition": 54 AAPL / 61 MSFT / 101 TSLA). Starting a new chunk at them
keeps each subsection together and fixes MSFT MD&A misses (Exp 4b) without hurting other queries.

**Setup:** New chunker flag `--pseudo-headings` (default off; flag off reproduces the baseline run
byte-for-byte). Same sizes as baseline (1200 / 200 / 900). A pseudo-heading = paragraph <= 80 chars,
no ending punctuation, not starting with a digit/bullet/"(", followed by body text in the same
section. No golden-labeled element is a pseudo-heading. Evaluated dense, hybrid 0.85 and hybrid
0.85 + ettin (retrieve-k 50) vs baseline run `2026-10-16_17-03-28`.

Two variants:
- **5a** (`2026-10-20_16-23-16`, 1436 chunks): pseudo-heading treated as a real heading (always
  breaks; text moves to the breadcrumb).
- **5b** (`2026-10-20_16-33-24`, 1447 chunks): breaks only if the current chunk already has
  >= 200 chars; heading text stays in the chunk body.

| Config | Run | Hit@5 | MRR@5 | Recall@5 | Prec@5 | nDCG@5 |
|---|---|---|---|---|---|---|
| dense | baseline | **0.946** | 0.804 | **0.689** | **0.375** | **0.659** |
| dense | 5a | 0.929 | 0.778 | 0.675 | **0.375** | 0.641 |
| dense | 5b | 0.911 | **0.807** | 0.646 | 0.371 | 0.646 |
| hybrid 0.85 | baseline | **0.946** | **0.793** | **0.704** | 0.400 | **0.662** |
| hybrid 0.85 | 5a | 0.875 | 0.731 | 0.634 | 0.389 | 0.607 |
| hybrid 0.85 | 5b | 0.911 | 0.769 | 0.675 | **0.404** | 0.638 |
| + ettin | baseline | 0.964 | 0.887 | 0.743 | 0.479 | 0.746 |
| + ettin | 5a | **0.982** | **0.917** | 0.752 | **0.482** | **0.765** |
| + ettin | 5b | **0.982** | 0.905 | **0.761** | **0.482** | 0.758 |

**Observations:**
- MSFT fix works: the "Intelligent Cloud" subsection is now one chunk (#406-412); q016 recall rises
  in every config.
- With ettin (production config): 5b improves every headline metric with **zero query regressions**
  (only q016 up; trend_explanation 0.62 -> 0.75, MSFT recall 0.62 -> 0.68). 5a also improves but
  loses q015 (Tesla energy revenue).
- Without a reranker, both variants regress on Apple (5b: AAPL recall 0.72 -> 0.60 in hybrid;
  q005, q012, q041 down). Apple's one-sentence region/product explanations become many small,
  similar chunks that push tables out of the top 5. 5a was worse: pseudo-headings in the breadcrumb
  mislabeled chunks (the Products & Services table got "Rest of Asia Pacific" as its heading).
- ettin absorbs the Apple fragmentation because the candidate pool (top 50) still contains the
  tables and the reranker ranks them correctly.
- Effect size is small: +1 query on Hit@5 (0.964 -> 0.982), so the gain is real but modest.

**Decision: adopted (5b).** Pseudo-heading boundaries are now the chunker default
(`--no-pseudo-headings` restores the old behavior) and `RUN_DIRECTORY` points to
`data/chunks/2026-10-20_16-33-24`. This requires the reranked pipeline (hybrid 0.85 + ettin);
do not use this chunking with a no-reranker setup. Possible next refinement: a larger break threshold
or keeping consecutive one-liners together, to remove the Apple regression for dense/hybrid.

---

## Summary - baseline to final

### Score progression (golden v2, k=5, 56 positive queries)

| Step | Config | Hit@1 | Hit@5 | MRR@5 | Recall@5 | Prec@5 | nDCG@5 | Recall@10 |
|---|---|---|---|---|---|---|---|---|
| Baseline | dense, 1200/900 chunks | 0.696 | 0.946 | 0.804 | 0.689 | 0.375 | 0.659 | 0.813 |
| + BM25 (Exp 3) | hybrid alpha 0.85 | 0.679 | 0.946 | 0.793 | 0.704 | 0.400 | 0.662 | 0.818 |
| + reranker (Exp 4) | hybrid 0.85 + ettin, retrieve-k 50 | 0.821 | 0.964 | 0.887 | 0.743 | 0.479 | 0.746 | 0.862 |
| **+ chunking (Exp 5b)** | **same + pseudo-heading chunks** | **0.839** | **0.982** | **0.905** | **0.761** | **0.482** | **0.758** | 0.859 |
| Total change vs baseline | | +0.143 | +0.036 | +0.101 | +0.072 | +0.107 | +0.099 | +0.046 |

**Final production config:** hybrid retrieval (alpha 0.85) -> top-50 candidates ->
`cross-encoder/ettin-reranker-150m-v1` -> top 5 chunks, chunk run `data/chunks/2026-10-20_16-33-24`
(1200 / 200 / 900 chars, pseudo-heading boundaries). Settings: `RETRIEVAL_ALPHA=0.85`,
`RETRIEVAL_K=50`, `CONTEXT_K=5`, `RERANKER_MODEL=cross-encoder/ettin-reranker-150m-v1`.

In plain terms: the right chunk is now ranked first for 84% of questions (was 70%), at least one
relevant chunk is in the top 5 for 55 of 56 questions, and nearly half of the 5 chunks passed to the
LLM are relevant (was 38%).

### What each improvement contributed

| Change | Main effect | Biggest gains |
|---|---|---|
| Hybrid (dense + 15% BM25) | Small recall/precision gain; ordering about the same | exact numbers, risk factors, cross-section |
| ettin cross-encoder reranker | Largest jump; fixes ordering | MRR +0.09, Hit@1 +0.14; exact_numeric, cross_document, tables |
| Pseudo-heading chunking | Fixes MSFT MD&A subsections; zero regressions with reranker | trend_explanation 0.62 -> 0.75, MSFT recall +0.06 |

Chunk size (Exp 1) was confirmed rather than changed: 1200/900 stayed best.
Rejected along the way: BM25 alone, hybrid alpha 0.5, MiniLM and bge-reranker (both worse than no
reranker), and chunking variant 5a.

### Evaluation improvements made (so the numbers are trustworthy)

These do not change the retriever; they make the scores correct and comparable.

- **Golden v2:** rebuilt 60-query golden after the v1 audit found ~25 wrong queries (invented
  numbers, off-by-one element orders, answerable "negatives", single-copy labels). Every order and
  number verified against the parsed elements; parsed data verified against the raw SEC HTML.
- **Equivalent copies grouped:** one relevant item = one fact, listing every place it appears
  (MD&A table, statements, notes), so retrieving any copy counts.
- **Scoring fixes:** each fact credited once and `doc_id`-aware (nDCG could exceed 1 before), and
  precision counted per chunk (precision@1 could exceed hit@1 before).
- **Validator:** `scripts/eval_scripts/validate_retriever_goldens.py` enforces the golden rules
  (0 errors on v2).
- **Tooling:** `--compare` shows chunk sizes, pseudo-heading flag and golden version; empty
  category/difficulty breakdowns fixed; v1 results archived so they are never mixed with v2.

### Known limitations

- Small test set: 1 query = 0.018 Hit@5, so differences under ~0.02 are noise.
- Weakest categories: cross_document (Recall@5 0.52) and cross_section (0.56). These need several
  pieces of evidence, which 5 chunks rarely cover.
- q014 (MSFT revenue growth factors) still misses, partly a golden label gap (Exp 4b, option A).
- The adopted chunking is tuned for the reranked pipeline; without a reranker it is worse on Apple.
- These are retrieval scores only; answer quality is measured separately by the generation and
  full-pipeline evaluations.

---

## Exp 6 - Document filtering and balanced cross-document retrieval

**Question:** Can document filtering improve the weak cross-document scores?

**Finding before implementation:** document filtering was already present. `HybridRetriever.search()`
accepts `doc_id`, and the evaluator already passes the golden query's `doc_id`. Therefore, all final
scores already use document filtering for single-document queries. Cross-document queries have
`doc_id: null` because their evidence intentionally spans multiple filings; filtering them to all
three documents would not change the search corpus.

**Experiment:** add opt-in `--balanced-docs` routing for reranked cross-document queries. It:

1. reads the distinct evidence document IDs from the golden query;
2. divides the same 50-candidate budget across those filings;
3. retrieves independently within each filing;
4. reranks the merged candidates with ettin; and
5. reserves at least one final top-5 result for each requested filing, then fills remaining slots by
   reranker score.

This uses golden evidence document IDs, so it is an **oracle upper-bound experiment**, not a deployable
query classifier. Production would need to detect company/document IDs from the user query.

| Metric | Final baseline | Balanced documents | Change |
|---|---:|---:|---:|
| Hit@5 | 0.982 | 0.982 | 0.000 |
| MRR@5 | 0.905 | 0.905 | 0.000 |
| Recall@5 | 0.761 | 0.764 | +0.003 |
| Precision@5 | 0.482 | 0.482 | 0.000 |
| nDCG@5 | 0.7584 | 0.7580 | -0.0004 |
| Cross-document Recall@5 | 0.517 | 0.542 | +0.025 |

Only two queries changed:

| Query | Recall@5 | nDCG@5 | Observation |
|---|---:|---:|---|
| q055 | 0.60 -> 0.60 | 0.645 -> 0.556 | Same evidence count, but the reserved result worsened graded ordering |
| q056 | 0.33 -> 0.50 | 0.374 -> 0.441 | One additional evidence item reached the top 5 |

**Decision:** do not adopt this version. The gain is negligible: overall Recall +0.003, nDCG slightly
worse, and even oracle document knowledge changes only two queries. The main cross-document bottleneck
is not unrelated filings entering the candidate pool. It is finding and fitting several distinct facts
from each requested filing into only five final slots. Better next experiments are query decomposition,
per-document/per-subquery evidence quotas greater than one, and adaptive context k=8-10.

---

## Exp 7 - Final context size (k=5 vs k=8 vs k=10)

**Question:** The pipeline already retrieves 50 hybrid candidates and reranks them with ettin. Does
passing more than the top 5 reranked chunks improve evidence coverage, especially for cross-document
and cross-section questions?

**Setup:** Final production retrieval configuration and chunk run:

- hybrid alpha 0.85;
- retrieve 50 candidates;
- `cross-encoder/ettin-reranker-150m-v1`;
- pseudo-heading chunk run `2026-10-20_16-33-24` (1447 chunks);
- evaluate the same reranked list at k=5, k=8, and k=10;
- golden v2, 56 positive queries.

Saved result: `eval/retriver_results/2026-10-21_14-48-26_rerank_a0.85_ettin-reranker-150m-v1_eval_results.json`.

### Overall results

| Metric | k=5 | k=8 | Change 5 -> 8 | k=10 | Change 8 -> 10 |
|---|---:|---:|---:|---:|---:|
| Hit | 0.982 | **1.000** | +0.018 | 1.000 | 0.000 |
| MRR | 0.905 | **0.908** | +0.003 | 0.908 | 0.000 |
| Recall | 0.761 | **0.833** | **+0.072** | 0.859 | +0.026 |
| Precision | **0.482** | 0.368 | -0.114 | 0.318 | -0.050 |
| nDCG | 0.758 | **0.785** | **+0.027** | 0.792 | +0.007 |

The precision decrease is expected because the denominator grows from 5 to 8 or 10. At k=8 the
retriever returns more relevant evidence in absolute terms, but also more irrelevant chunks. The
simultaneous Recall and nDCG increases show that the extra relevant evidence is useful rather than
only adding noise at low ranks.

### Recall by category

| Category | Recall@5 | Recall@8 | Change | Recall@10 |
|---|---:|---:|---:|---:|
| cross_section | 0.561 | **0.783** | **+0.222** | 0.858 |
| accounting_policy | 0.740 | **0.844** | **+0.104** | 0.844 |
| cross_document | 0.517 | **0.619** | **+0.102** | 0.675 |
| table_structure | 0.875 | **0.938** | +0.063 | 0.938 |
| trend_explanation | 0.750 | **0.812** | +0.062 | 0.812 |
| risk_factor | 0.725 | **0.787** | +0.062 | 0.875 |
| exact_numeric | 0.917 | 0.917 | 0.000 | 0.917 |
| business_description | 0.958 | 0.958 | 0.000 | 0.958 |

The gain is concentrated in the intended multi-evidence categories. Exact numeric and business
description do not improve because their required evidence is already in the top 5.

### Difficulty impact

| Difficulty | Recall@5 | Recall@8 | Change | nDCG@5 | nDCG@8 | Change |
|---|---:|---:|---:|---:|---:|---:|
| easy | 0.908 | 0.908 | 0.000 | 0.866 | 0.866 | 0.000 |
| medium | 0.718 | 0.798 | +0.080 | 0.746 | 0.765 | +0.019 |
| hard | 0.617 | **0.780** | **+0.163** | 0.624 | **0.698** | **+0.074** |

### Interpretation

- The original top-5 cutoff is a real evidence-coverage bottleneck: relevant chunks often occupy
  ranks 6-8 after reranking.
- k=8 captures most of the available gain: +0.072 Recall and +0.027 nDCG.
- Moving from k=8 to k=10 adds only +0.026 Recall and +0.007 nDCG while reducing precision by another
  0.050 and adding two more chunks to the LLM context.
- k=8 is therefore the best retrieval-quality/token-cost compromise among the tested values.

**Decision:** k=8 is a strong candidate, but do not change production from `CONTEXT_K=5` based on
retrieval metrics alone. First compare generation quality using identical questions with top-5 and
top-8 contexts. Preferred long-term design: k=5 for simple/exact questions and adaptive k=8 for
cross-document, cross-section, comparison, and other multi-evidence questions.

