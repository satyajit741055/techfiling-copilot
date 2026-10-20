# Retrieval configuration evolution and final selection

This document records how the retrieval configuration was selected through controlled evaluation.
It summarizes the experiment sequence, measured trade-offs, rejected alternatives, final parameter
values, and remaining validation work.

Detailed per-run notes are available in [`EXPERIMENTS.md`](./EXPERIMENTS.md). Evaluation outputs
are stored under `eval/results/`.

## Final quality-first configuration

| Component | Selected value |
|---|---|
| Text chunk budget | `1200` characters |
| Table chunk budget | `900` characters |
| Dense embedding model | `BAAI/bge-base-en-v1.5` |
| Sparse retriever | BM25 over breadcrumb and chunk text |
| Hybrid dense weight | `alpha=0.85` |
| Hybrid BM25 weight | `1 - alpha = 0.15` |
| Candidate count | `retrieve_k=50` |
| Cross-encoder | `cross-encoder/ettin-reranker-150m-v1` |
| Final result count | `k=5` |

```text
parsed SEC 10-K elements
  ↓
structure-aware chunking (text=1200, table=900)
  ↓
BGE dense retrieval + BM25 lexical retrieval
  ↓
min-max score normalization
  ↓
0.85 × dense score + 0.15 × BM25 score
  ↓
top 50 candidates
  ↓
Ettin 150M cross-encoder reranking
  ↓
final top 5 chunks
```

## Final measured result

| Configuration | Hit@5 | MRR@5 | Recall@5 | Precision@5 | nDCG@5 |
|---|---:|---:|---:|---:|---:|
| Original 1400/1000 dense baseline | 0.732 | 0.521 | 0.592 | 0.200 | 0.476 |
| Tuned 1200/900 dense baseline | 0.750 | 0.551 | 0.601 | 0.207 | 0.511 |
| **Final hybrid + Ettin pipeline** | **0.786** | **0.587** | **0.640** | **0.225** | **0.546** |

Absolute improvement over the original baseline:

```text
Hit@5:      +0.054
MRR@5:      +0.066
Recall@5:   +0.048
Precision:  +0.025
nDCG@5:     +0.070
```

Absolute improvement over the tuned dense baseline:

```text
Hit@5:      +0.036
MRR@5:      +0.036
Recall@5:   +0.039
Precision:  +0.018
nDCG@5:     +0.035
```

---

## 1. Evaluation basis

### 1.1 Golden dataset

The evaluation uses `goldens/golden_retriever_dataset.jsonl`:

- 60 labeled questions
- 56 positive queries
- 4 negative queries
- Apple, Microsoft, and Tesla 10-K filings
- Multiple categories and difficulty levels
- Relevance grades from 1 to 3

Relevance is linked to parser-level `element_orders`, not chunk IDs. A retrieved chunk is relevant
when its `element_orders` intersects the expected orders for a query.

This makes the labels stable across chunking experiments. Chunk IDs change when elements are split
or merged; parser element orders do not.

### 1.2 Relevance grades

| Grade | Meaning |
|---:|---|
| 3 | Directly answers the question |
| 2 | Supporting evidence |
| 1 | Useful context |
| 0 | Irrelevant |

Graded labels support nDCG, which rewards placing direct evidence above weaker supporting context.

### 1.3 Metrics

| Metric | Purpose |
|---|---|
| Hit Rate@k | Fraction of positive queries with at least one relevant result in the top k |
| MRR@k | Reciprocal rank of the first relevant result, averaged across queries |
| Recall@k | Fraction of labeled relevant evidence recovered in the top k |
| Precision@k | Fraction of retrieved results that are relevant |
| nDCG@k | Ranking quality with graded relevance and position discounting |

`nDCG@5` is the primary ranking metric. MRR and Recall are used as constraints because the final
few chunks dominate downstream generation, while evidence coverage remains important for complete
answers.

### 1.4 Controlled experiment policy

Experiments were run by changing one major variable at a time:

1. Chunk budgets
2. Retrieval mode
3. Hybrid fusion weight
4. Reranker model
5. Candidate depth

A reproducibility run repeated the original 1400/1000 setup and produced identical metrics and
chunk counts, confirming deterministic behavior for the evaluated pipeline.

---

## 2. Phase 1: chunk-budget selection

### Objective

Find a chunk size that improves semantic focus without removing context or spreading evidence
across too many chunks.

### Evaluated configurations

| Text budget | Table budget | Chunks | Hit@5 | MRR@5 | Recall@5 | Precision@5 | nDCG@5 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1400 | 1000 | 1254 | 0.732 | 0.521 | 0.592 | 0.200 | 0.476 |
| 1300 | 1000 | 1307 | 0.732 | 0.494 | 0.598 | 0.204 | 0.478 |
| **1200** | **900** | **1397** | **0.750** | **0.551** | **0.601** | **0.207** | **0.511** |
| 1000 | 900 | 1544 | 0.732 | 0.525 | 0.583 | 0.196 | 0.490 |

### Findings

The relationship between chunk size and retrieval quality was non-monotonic:

- Reducing text chunks from 1400 to 1300 produced little improvement and reduced MRR.
- The 1200/900 configuration improved every headline metric over the original baseline.
- Reducing text chunks further to 1000 increased chunk count but reduced retrieval quality.

Large chunks can mix topics and dilute embedding specificity. Very small chunks can remove
explanatory context and distribute relevant evidence across too many chunks. The selected
1200/900 configuration provided the best measured balance.

### Decision

Freeze the following values for subsequent retrieval experiments:

```text
MAX_CHARS = 1200
TABLE_MAX_CHARS = 900
```

---

## 3. Phase 2: retrieval-mode comparison

### Objective

Determine whether lexical retrieval adds useful signal to BGE dense retrieval.

### Evaluated modes

| Mode | Alpha | Hit@5 | MRR@5 | Recall@5 | Precision@5 | nDCG@5 |
|---|---:|---:|---:|---:|---:|---:|
| **Dense** | — | **0.750** | **0.551** | **0.601** | **0.207** | **0.511** |
| BM25 | — | 0.536 | 0.365 | 0.443 | 0.143 | 0.361 |
| Hybrid | 0.50 | 0.625 | 0.437 | 0.515 | 0.175 | 0.428 |

### Findings

BM25 alone was substantially weaker than dense retrieval on the evaluated query distribution.
Equal weighting also reduced quality because the weaker lexical signal received enough influence
to displace strong dense results.

The hybrid implementation combines min-max normalized scores:

```text
hybrid_score = alpha × normalized_dense + (1 - alpha) × normalized_BM25
```

An equal `alpha=0.50` therefore gives BM25 significant ranking influence even though its standalone
performance is lower.

### Decision

Reject equal weighting. Retain BM25 only as a lower-weight lexical signal and tune `alpha`.

---

## 4. Phase 3: hybrid-alpha selection

### Objective

Find a dense-heavy fusion weight that preserves exact-match signal without degrading semantic
retrieval.

### Key alpha results

| Alpha | Hit@5 | MRR@5 | Recall@5 | Precision@5 | nDCG@5 | Recall@20 | Recall@50 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.80 | 0.750 | 0.540 | **0.616** | **0.207** | 0.509 | **0.771** | **0.932** |
| **0.85** | 0.750 | 0.558 | 0.598 | 0.204 | 0.520 | 0.759 | **0.932** |
| 0.90 | 0.750 | **0.575** | 0.589 | 0.200 | **0.524** | 0.759 | 0.911 |
| Dense, 1.00 | 0.750 | 0.551 | 0.601 | **0.207** | 0.511 | — | — |

### Direct ranking versus candidate generation

Alpha 0.90 produced the strongest direct top-five hybrid ranking. It would be the preferred choice
if hybrid retrieval were the final output.

The production design adds a cross-encoder after hybrid retrieval. At that stage, candidate recall
is more important than initial ordering because the cross-encoder replaces the candidate ranking.
A reranker cannot recover evidence omitted from the candidate pool.

At 50 candidates:

- Alpha 0.80 and 0.85 both reached Recall@50 `0.932` and Hit@50 `0.982`.
- Alpha 0.85 had higher candidate nDCG: `0.636` versus `0.625`.
- Alpha 0.90 had lower Recall@50 `0.911` and Hit@50 `0.946`.

### Decision

Use `alpha=0.85` for candidate generation:

```text
dense weight = 0.85
BM25 weight = 0.15
```

This selection is specific to the two-stage pipeline. Alpha 0.90 remains the best measured
configuration when no cross-encoder is used.

---

## 5. Phase 4: cross-encoder selection

### Objective

Select a reranker that improves the ordering of a fixed high-recall candidate pool.

### Controlled settings

All reranker models used:

```text
chunk budgets = 1200 / 900
hybrid alpha = 0.85
retrieve_k = 50
final k = 5
```

### Model comparison

| Reranker | Hit@5 | MRR@5 | Recall@5 | Precision@5 | nDCG@5 |
|---|---:|---:|---:|---:|---:|
| MiniLM-L-6-v2 | 0.661 | 0.468 | 0.521 | 0.175 | 0.431 |
| BGE reranker base | 0.750 | 0.509 | 0.577 | 0.196 | 0.466 |
| **Ettin reranker 150M** | **0.786** | **0.587** | **0.640** | **0.225** | **0.546** |

### Findings

A cross-encoder did not automatically improve quality:

- `cross-encoder/ms-marco-MiniLM-L-6-v2` degraded every headline metric.
- `BAAI/bge-reranker-base` also underperformed the hybrid input.
- `cross-encoder/ettin-reranker-150m-v1` improved every metric.

The candidate pool had Recall@50 `0.932`, so poor MiniLM and BGE results were not caused by missing
candidates. Those models received relevant evidence but reordered it less effectively for this
workload.

### Decision

Select:

```text
cross-encoder/ettin-reranker-150m-v1
```

---

## 6. Phase 5: candidate-depth selection

### Objective

Reduce cross-encoder work if a smaller candidate pool preserves final quality.

### Results

| Retrieve-k | Hit@5 | MRR@5 | Recall@5 | Precision@5 | nDCG@5 |
|---:|---:|---:|---:|---:|---:|
| 20 | 0.750 | 0.574 | 0.598 | 0.211 | 0.526 |
| 30 | 0.750 | 0.572 | 0.604 | 0.214 | 0.530 |
| **50** | **0.786** | **0.587** | **0.640** | **0.225** | **0.546** |

### Findings

Twenty and thirty candidates produced similar results. Expanding the candidate pool to 50 improved
every metric, indicating that relevant evidence often appeared below candidate rank 30 and that
Ettin could successfully promote it into the final top five.

The experiment demonstrates the candidate ceiling:

```text
final reranker recall ≤ candidate-generator recall
```

### Decision

Use `retrieve_k=50` for the quality-first configuration. A latency-sensitive deployment may test
40 candidates or explicitly select a lower-cost 20/30 profile, but those variants have lower
measured retrieval quality.

---

## 7. Configuration evolution summary

| Stage | Configuration selected | Reason |
|---|---|---|
| Initial baseline | Dense, 1400/1000 chunks | Starting reference |
| Chunk tuning | Dense, 1200/900 chunks | Best chunking result across all metrics |
| Retrieval comparison | Dense over BM25 and alpha 0.50 hybrid | BM25 and equal fusion were weaker |
| Alpha tuning | Hybrid alpha 0.85 for candidates | Strong top-50 recall and candidate quality |
| Reranker selection | Ettin 150M | Only reranker to improve every metric |
| Candidate tuning | Retrieve 50, return 5 | Best final quality; evidence existed below rank 30 |

Rejected alternatives remain documented because they define the measured boundaries of the final
configuration:

- Text/table budgets of 1400/1000, 1300/1000, and 1000/900
- BM25-only retrieval
- Equal-weight hybrid retrieval
- MiniLM cross-encoder reranking
- BGE base reranking
- Candidate depths of 20 and 30

---

## 8. Reproducing the selected configuration

### 8.1 Build chunks and embeddings

```powershell
$env:PYTHONPATH = "$PWD\src"

uv run python -m ingestion.processor `
  --skip-parse `
  --max-chars 1200 `
  --table-max-chars 900
```

The command creates a timestamped directory under `data/chunks/` containing:

```text
<doc_id>.chunks.jsonl
embeddings_gpu_base.npy
run_manifest.json
```

### 8.2 Evaluate the selected retrieval pipeline

Set the run directory produced by the ingestion command:

```powershell
$RUN = ".\data\chunks\<run_timestamp>"
```

Run the final evaluation:

```powershell
uv run python .\eval\hybrid_rerank_eval.py `
  --mode rerank `
  --alpha 0.85 `
  --retrieve-k 50 `
  --reranker-model "cross-encoder/ettin-reranker-150m-v1" `
  --run-directory $RUN `
  --device cuda
```

Compare persisted evaluation runs:

```powershell
uv run python .\eval\hybrid_rerank_eval.py --compare
```

### 8.3 Lower-cost retrieval without a cross-encoder

When cross-encoder latency is not acceptable, the strongest tested direct hybrid configuration is:

```powershell
uv run python .\eval\hybrid_rerank_eval.py `
  --mode hybrid `
  --alpha 0.90 `
  --run-directory $RUN `
  --device cuda
```

Its measured top-five metrics were:

```text
Hit@5 = 0.750
MRR@5 = 0.575
Recall@5 = 0.589
Precision@5 = 0.200
nDCG@5 = 0.524
```

---

## 9. Operational profiles

The experiments support two explicit profiles.

### Quality-first profile

```text
alpha = 0.85
retrieve_k = 50
reranker = cross-encoder/ettin-reranker-150m-v1
final k = 5
```

Measured result:

```text
Hit@5 = 0.786
MRR@5 = 0.587
Recall@5 = 0.640
Precision@5 = 0.225
nDCG@5 = 0.546
```

### Lower-cost profile

```text
alpha = 0.90
no cross-encoder
final k = 5
```

Measured result:

```text
Hit@5 = 0.750
MRR@5 = 0.575
Recall@5 = 0.589
Precision@5 = 0.200
nDCG@5 = 0.524
```

A lower-cost Ettin profile with `retrieve_k=20` or `30` is also possible, but latency has not yet
been measured and those configurations have lower final quality than `retrieve_k=50`.

---

## 10. Known limitations

The selected configuration is the best result on the current evaluation set, subject to the
following limitations:

1. **Tuning and reporting use the same 60 queries.** A separate held-out test set is required for
   an unbiased final quality claim.
2. **Category and difficulty breakdowns are empty in the current hybrid evaluator output.** Overall
   metrics are available, but subgroup reporting must be repaired before claiming uniform gains.
3. **Latency and resource usage are not included in the comparison table.** Warm-query latency,
   throughput, model load time, and peak GPU memory must be measured before deployment.
4. **The golden set covers three companies.** Additional companies, filing years, and query types
   are needed to establish broader generalization.
5. **Retrieval metrics do not measure final answer quality.** The complete RAG pipeline still
   requires groundedness, citation, numeric-faithfulness, and answer-correctness evaluation.
6. **Metric confidence intervals are not calculated.** Paired bootstrap intervals would help
   determine whether small differences are statistically robust.

---

## 11. Follow-up work

- Repair category and difficulty breakdown generation in `hybrid_rerank_eval.py`.
- Create a stratified development/test split by category, difficulty, company, and answer type.
- Tune parameters only on the development split and report the selected configuration once on test.
- Measure p50/p95 warm latency, throughput, model load time, and peak GPU memory.
- Test `retrieve_k=40` only if latency measurements justify another quality/cost experiment.
- Add paired bootstrap confidence intervals for metric deltas.
- Evaluate the full retrieve → generate pipeline for groundedness and answer correctness.
- Add regression thresholds so retrieval quality changes can be checked in CI.

---

## 12. Related artifacts

| Artifact | Purpose |
|---|---|
| `eval/EXPERIMENTS.md` | Chronological hypothesis, result, and decision log |
| `eval/dense_retriever_eval.py` | Dense retrieval evaluation harness |
| `eval/hybrid_rerank_eval.py` | Dense, BM25, hybrid, and reranker evaluation harness |
| `eval/results/` | Persisted aggregate and per-query evaluation results |
| `goldens/golden_retriever_dataset.jsonl` | Labeled retrieval queries |
| `src/ingestion/chunking/chunker.py` | Structure-aware chunk generation |
| `src/ingestion/chunking/embedder.py` | BGE embedding generation and manifest creation |
| `src/retriever/retriever.py` | Dense retrieval implementation |
| `src/retriever/hybrid_retriever.py` | BM25, hybrid fusion, and cross-encoder pipeline |

The configuration in this document is the current quality-first selection for the evaluated
corpus and golden dataset. It should be revised when the corpus, labels, embedding model, or
operational constraints change.
