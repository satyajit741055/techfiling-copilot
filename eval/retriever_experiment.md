# Retriever Experiments

**Dataset:** `goldens/golden_retriever_15filings.jsonl`: 75 queries over AAPL, MSFT and TSLA 10-Ks (FY2021–2025). They are single-filing and open-corpus questions only.
**Metrics:** top 5 results. Results are saved in `eval/retriver_results/`; `--compare` prints them side by side.
**Embedder:** `BAAI/bge-base-en-v1.5`, running on a local GPU.

## Final configuration

| Setting | Value |
|---|---|
| Collection | `techfiling_20261031T133214Z` (1200/900 chunks, breadcrumb fix) |
| Mode | Hybrid (dense + BM25, weighted RRF) + reranker |
| Reranker | `cross-encoder/ettin-reranker-150m-v1` |
| Alpha | 0.7 |
| Retrieve-k | 50 |

**Hit@5 1.000 · MRR 0.967 · Recall 0.821 · nDCG 0.838**

## How we got there

| # | Step | Hypothesis | Change | Hit@5 | MRR | Recall | nDCG | Decision |
|---|---|---|---|---|---|---|---|---|
| 0 | Baseline | Dense search alone is good enough | Dense only, 1200/900 chunks | 0.920 | 0.792 | 0.680 | 0.652 | Baseline |
| 1 | Chunk size | Smaller chunks are more precise; bigger chunks carry more context | Dense, 800/600 | 0.907 | 0.769 | 0.632 | 0.636 | Rejected |
| | | | Dense, 1300/1000 | 0.933 | 0.780 | 0.683 | 0.640 | Tie with 1200/900 |
| | | | 1800/1200 | — | — | — | — | Rejected: chunks go over the 512-token limit |
| 2 | BM25 | Exact keyword search helps with numbers and tables | BM25 only | 0.960 | 0.822 | 0.678 | 0.669 | Better on tables and risk factors, worse on trend questions |
| 3 | Hybrid | Dense and BM25 are good at different questions, so combining them helps | Hybrid, alpha 0.85 | 0.987 | 0.862 | 0.760 | 0.733 | Kept |
| 4 | Reranker | A cross-encoder can fix the order of the candidates | Hybrid + ettin, alpha 0.85, k=50 | 0.987 | 0.953 | 0.803 | 0.822 | Kept (biggest gain) |
| 5 | Alpha | At 0.85, RRF lets dense crowd out BM25 almost completely | Tuned from 0.0 to 1.0 (hybrid only) | 0.987 | 0.856 | 0.774 | — | Similar results from 0.3 to 0.9; picked 0.7 |
| | | | Reranker, alpha 0.7 | 0.987 | 0.953 | 0.807 | 0.824 | Kept 0.7 (ties 0.5) |
| 6 | Retrieve-k | Fewer candidates may give the same quality | k=30 | 0.987 | 0.953 | 0.801 | 0.822 | Rejected |
| | | | k=20 | 0.987 | 0.953 | 0.792 | 0.822 | Rejected: recall drops; kept 50 |
| 7 | Other rerankers | A larger reranker could rank better | bge-reranker-base | 0.987 | 0.844 | 0.741 | 0.702 | Rejected |
| | | | ms-marco-MiniLM-L-6-v2 | 0.973 | 0.848 | 0.686 | 0.691 | Rejected: weak on tables |
| 8 | Breadcrumb fix | q012 misses because chunk breadcrumbs carry the wrong (stale) heading | Fixed chunker, re-indexed | **1.000** | **0.967** | **0.821** | **0.838** | **Final** |

## Key learnings

- **The reranker gave the biggest gain:** MRR went from 0.86 to 0.95.
- **Hybrid beats dense or BM25 alone:** recall went up by about 8 points.
- **Alpha barely matters once the reranker is on;** it only decides which 50 candidates the reranker sees.
- **The breadcrumb fix:** the chunker kept the last two headings without knowing their levels. A previous risk-factor title therefore ended up in the breadcrumbs of a new risk category. The fix changed 538 breadcrumbs in AAPL and TSLA risk factors and none in MSFT; chunk boundaries did not change.

## Open items

- Add the cross-year, cross-company and negative goldens (q076–q105), then test filtered and grouped retrieval.
- Run dense, BM25 and hybrid (no reranker) on the fixed collection.
- Results come from only 75 queries, so a gap of about one query is noise.
