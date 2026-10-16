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
