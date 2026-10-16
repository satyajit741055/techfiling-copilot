You are a senior dataset auditor creating a trustworthy golden dataset for isolated evaluation of an SEC 10-K RAG answer generator.

Repository root:
C:\Users\Satyajit_Khot\Downloads\techfiling-copilot

Your task is to REVIEW AND CORRECT the existing generation dataset. Do not generate a new benchmark from prior knowledge.

Read these files first:

1. goldens/generator_golden.md
2. goldens/golden_retriver_dataset.jsonl
3. goldens/golden_generation_dataset.jsonl
4. scripts/eval_scripts/build_generation_goldens.py
5. scripts/eval_scripts/validate_generation_goldens.py
6. data/chunks/2026-10-16_17-03-28/AAPL_10-K_2025.chunks.jsonl
7. data/chunks/2026-10-16_17-03-28/MSFT_10-K_2026.chunks.jsonl
8. data/chunks/2026-10-16_17-03-28/TSLA_10-K_2025.chunks.jsonl
9. src/generation/generator.py
10. data/parsed/AAPL_10-K_2025.elements.jsonl
11. data/parsed/MSFT_10-K_2026.elements.jsonl
12. data/parsed/TSLA_10-K_2025.elements.jsonl

Do not use web search, outside knowledge, or model memory. Treat the supplied filing chunks as the only evidence.

BACKGROUND

The existing dataset contains 60 source questions with three variants each:

- oracle
- oracle_plus_distractors
- no_gold

It also contains 12 trap cases.

The current dataset has known quality problems:

1. Some no_gold contexts contain the exact answer.
2. Some expected answers conflict with the supplied filing evidence.
3. Some oracle contexts do not support every expected value or required fact.
4. Element-order overlap was sometimes incorrectly treated as proof of answer support.
5. Narrative required facts may be too broad or non-atomic.

One confirmed example is q004:

- Existing expected answer: $33,284 million
- Supplied FY2025 Apple R&D evidence: $34,550 million

The expected answer must therefore be corrected if the supplied evidence confirms this.

OBJECTIVE

Produce a corrected, evidence-backed generation golden dataset for isolated generator evaluation.

Do not overwrite the current dataset initially.

Create:

1. goldens/golden_generation_dataset_claude.jsonl
2. goldens/generation_claude_review.jsonl
3. scripts/eval_scripts/validate_generation_goldens_claude.py

REVIEW UNIT

Review by source question, not by individual variant.

For each source_qid, examine together:

- oracle
- oracle_plus_distractors
- no_gold

Review all 60 source questions, then review the 12 trap cases.

FOR EACH SOURCE QUESTION

A. Validate the question

Identify:

- company;
- requested metric or subject;
- fiscal period;
- requested unit/currency;
- requested operation, such as lookup, comparison, explanation, or calculation.

If the question is ambiguous, do not guess. Mark it for human review.

B. Validate the expected answer

The existing expected answer is a proposal, not authoritative ground truth.

Verify every claim against the supplied filing chunks.

Correct:

- values;
- signs;
- units;
- currencies;
- fiscal periods;
- company names;
- metric names;
- calculations;
- unsupported narrative details.

For every correction, record:

- original label;
- corrected label;
- reason;
- exact evidence;
- supporting chunk ID.

C. Create structured expected values

For every numeric fact, record:

{
  "value": 34550,
  "unit": "million",
  "currency": "USD",
  "period": "FY2025",
  "metric": "research and development expense"
}

Preserve the filing's exact value and disclosed unit.

Do not incorrectly label percentages or counts as currency.

D. Create atomic required facts

For narrative answers, split expected content into independently verifiable atomic facts.

Bad required fact:

"Revenue increased due to Azure and Microsoft 365 while margins fell because of AI infrastructure."

Better:

[
  "Revenue increased due to Azure growth",
  "Revenue increased due to Microsoft 365 Commercial cloud growth",
  "Cloud gross margin percentage decreased due to AI infrastructure investment"
]

Every required fact must have explicit evidence.

Do not include optional details merely to make the answer longer.

E. Select oracle evidence

Choose the minimum sufficient chunk set that supports every expected value and required fact.

Do not select a chunk only because:

- its element_orders overlap;
- it discusses a related subject;
- it contains a similar number;
- it belongs to the same table or section.

For every expected value/fact, provide:

- supporting chunk IDs;
- an exact verbatim evidence excerpt.

If the complete answer cannot be supported by the available chunks, mark the question for human review or revise the expected answer to only what is supported.

F. Build oracle_plus_distractors

Include the complete oracle evidence and plausible same-document distractors, normally five chunks total.

Good distractors include:

- same metric for another year;
- related but different metric;
- another segment/product/geography;
- subtotal versus total;
- amount versus percentage;
- nearby explanatory material.

A distractor must not be marked non-gold if it independently supports an essential answer fact.

Use deterministic ordering and do not always place gold chunks first.

G. Build no_gold

Create context that is topically plausible but genuinely insufficient.

Remove every chunk that contains or enables derivation of:

- the requested value;
- an equivalent value in another unit;
- all required calculation inputs;
- any essential narrative answer fact.

Evaluate the combined context, not only chunks individually.

The no_gold context may mention the subject, but it must not answer the question.

If any expected answer evidence remains, reject that no_gold proposal and select different chunks.

H. Review trap cases

For every trap case:

- confirm that the context is insufficient;
- confirm that the requested period/company/metric/calculation cannot be answered;
- record a precise unanswerable_reason;
- ensure that no combination of supplied chunks answers it.

OUTPUT RECORD REQUIREMENTS

Each corrected JSONL case must preserve the existing schema and add:

{
  "labeling_method": "claude_semantic_review",
  "review_model": "<exact Claude model identifier>",
  "review_prompt_version": "v1",
  "confidence": "high|medium|low",
  "requires_human_review": false,
  "evidence_map": [
    {
      "fact": "atomic expected fact",
      "chunk_ids": ["chunk_id"],
      "verbatim_evidence": ["exact excerpt copied from chunk"]
    }
  ],
  "corrections": [
    {
      "field": "expected_answer",
      "original": "$33,284 million",
      "corrected": "$34,550 million",
      "reason": "The supplied FY2025 R&D table reports 34,550 million."
    }
  ]
}

Never mark a case human-approved. Use:

"review_status": "claude_reviewed"

A human will approve cases later.

CRITICAL RULES

1. Never rewrite, summarize, repair, or normalize chunk text.
2. Store the complete original frozen chunk dictionaries.
3. Never use prior knowledge or web search.
4. Never assume the existing expected answer is correct.
5. Never assume element overlap means the chunk supports the answer.
6. Never include supporting evidence in no_gold.
7. Never hide uncertainty.
8. Never invent a value or required fact.
9. Exact evidence excerpts must be copied verbatim from supplied chunks.
10. Do not use generated answers from eval results to define ground truth.
11. Do not optimize labels to make any evaluated model score better.
12. Do not modify the original dataset or chunk files.

REVIEW LOG

Write one record per source question to:

goldens/generation_claude_review.jsonl

Use this schema:

{
  "source_qid": "q004",
  "question_valid": true,
  "expected_answer_valid": false,
  "original_expected_answer": "$33,284 million",
  "corrected_expected_answer": "$34,550 million",
  "oracle_sufficient": true,
  "no_gold_answerable": false,
  "corrections": [],
  "confidence": "high",
  "requires_human_review": false,
  "review_notes": []
}

VALIDATOR REQUIREMENTS

Create scripts/eval_scripts/validate_generation_goldens_claude.py.

It must check:

1. JSONL parsing and unique case IDs.
2. Exactly one oracle, oracle_plus_distractors, and no_gold case per source_qid.
3. All 12 trap cases exist.
4. Context chunk IDs match stored chunk dictionaries.
5. SHA-256 hashes match exact UTF-8 chunk text.
6. Gold chunk IDs exactly match context entries with is_gold=true.
7. Answerable cases contain expected answers and evidence.
8. Unanswerable cases have no gold chunks and include unanswerable_reason.
9. Every expected value appears in at least one oracle chunk after unit normalization.
10. Every evidence_map quote occurs verbatim in its claimed chunk.
11. No expected numeric value appears in no_gold after unit normalization.
12. Context chunks belong to the declared document, except explicit cross-document traps.
13. All standard cases have review_status=claude_reviewed.
14. Every correction appears in the review log.
15. Every medium/low-confidence case requires human review.

Important limitation:
A numeric absence check is necessary but not sufficient for no_gold. Narrative leakage must also be reviewed semantically.

WORKFLOW

Before editing:

1. Report:
   - case counts;
   - source-question counts;
   - category distribution;
   - current oracle-support failures;
   - current no_gold leakage count;
   - proposed review approach.

Then:

2. Review and correct all source questions.
3. Write the corrected dataset and review log.
4. Write the independent validator.
5. Run the validator.
6. Inspect all corrections and all medium/low-confidence cases.
7. Do not silently resolve ambiguity.

FINAL REPORT

Report:

- exact Claude model used;
- number of source questions reviewed;
- expected-answer corrections;
- expected-value corrections;
- oracle-context corrections;
- no-gold contexts replaced;
- distractor corrections;
- trap corrections;
- high/medium/low-confidence counts;
- unresolved human-review cases;
- validator results;
- files created and modified.

Do not claim the dataset is human-approved.