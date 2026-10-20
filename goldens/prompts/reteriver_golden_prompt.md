# Prompt for the retrieval golden dataset (v2)

You are a financial-domain retrieval evaluation expert working inside this repository:

C:\Users\Satyajit_Khot\Downloads\techfiling-copilot

I am building a RAG system over SEC 10-K filings. Generate a golden evaluation dataset that
will be used to programmatically measure retrieval quality (Hit@k, MRR@k, Recall@k,
Precision@k, nDCG@k) with `eval/hybrid_rerank_eval.py` and `eval/dense_retriever_eval.py`.

You have tool access (file reading, search, shell). Use it. Every order number, number, and
claim in the dataset must be verified against the source files before it is written.

────────────────────────────────────────────────────────────────────────

## SOURCE FILES (the only authoritative source)

data\parsed\AAPL_10-K_2025.elements.jsonl   (Apple, 624 elements)
data\parsed\MSFT_10-K_2026.elements.jsonl   (Microsoft, 967 elements)
data\parsed\TSLA_10-K_2025.elements.jsonl   (Tesla, 957 elements)

The original SEC HTML filings are in data\raw_html\ (mapping in data\parsed\*.document.json).
The parsed files have been verified against the HTML; use the parsed files for all work.

Do not use external knowledge, model memory, web searches, or any existing golden file as a
source of facts. If a fact is not in the parsed elements, it does not exist for this task.

Each line of an elements file is one element:

{
  "doc_id":        "AAPL_10-K_2025",
  "order":         266,                         // stable position; this is what you cite
  "type":          "table" | "paragraph" | "heading",
  "part":          "Part I" | "Part II" | ...,
  "section":       "Item 1" | "Item 1A" | "Item 7" | "Item 8" | ...,
  "section_title": "Risk Factors" | ...,
  "text":          "...",                       // paragraphs/headings (null for tables)
  "table": {                                    // null for non-tables
    "table_id":  "AAPL_10-K_2025_t016",
    "title":     "Products and Services Performance",
    "units":     "dollars in millions",
    "markdown":  "| Line item | 2025 | ... |",
    "rows":      [{"label": "Total net sales", "cells": [{"text": "416,161", ...}]}]
  }
}

`order` is the element's `order` field, never the physical line number in the file.

────────────────────────────────────────────────────────────────────────

## OUTPUT

Write exactly 60 lines of JSONL to:

goldens\golden_retriever_dataset_v2.jsonl

If that file already exists, DO NOT overwrite it. Write to
goldens\golden_retriever_dataset_v2_new.jsonl instead.

One complete JSON object per line. No array wrapper, no Markdown fences, no commentary.

### Record schema

{
  "id":              "q001",                    // sequential q001..q060, no gaps
  "query":           "What was Apple's total net sales in fiscal year 2025?",
  "doc_id":          "AAPL_10-K_2025",          // null ONLY for cross_document
  "category":        "exact_numeric",
  "difficulty":      "easy",                    // easy | medium | hard
  "expected_answer": "$416,161 million, up 6% from $391,035 million in fiscal 2024.",
  "relevant": [
    {
      "match_by":  "element_orders",            // always "element_orders"
      "value":     [279, 357, 406, 266, 533, 529],
      "relevance": 3,
      "reason":    "Total net sales row in Products and Services Performance (AAPL_10-K_2025_t016), Statements of Operations (AAPL_10-K_2025_t023), ...",
      "doc_id":    "AAPL_10-K_2025"             // REQUIRED on every item, every query
    }
  ]
}

Optional field:
- "derived_values": ["67.9%"]  - list any number in expected_answer that is calculated
  from cited operands rather than printed in the filing (e.g., a ratio). Use sparingly;
  every operand must be in a cited element.

────────────────────────────────────────────────────────────────────────

## THE KEY CONCEPT: ONE RELEVANT ITEM = ONE PIECE OF INFORMATION

A relevant item represents one fact or one piece of evidence, NOT one element. 10-Ks repeat
the same information in several places, and the evaluator credits an item if ANY of its
listed elements is retrieved (each item is credited at most once). Therefore:

1. Put ALL equivalent copies of the same fact into the SAME item's `value` list:
   - the same number in several tables (MD&A table, financial statement, note table,
     segment note, revenue disaggregation, GAAP/non-GAAP reconciliation, etc.)
   - verbatim or near-verbatim paragraphs repeated across sections (MD&A "Critical
     Accounting Estimates" usually duplicates Note 1/Note 2 text; Item 1 often duplicates
     Item 1A or Item 8 text)
   - an MD&A paragraph that states the same figure as a table (e.g., "$14.75 billion"
     alongside a table showing 14,747)
   - a paragraph that the parser split across a page break (the second part starts
     mid-sentence) - list both halves in one item

2. Put DIFFERENT pieces of information in SEPARATE items, each with its own grade.

3. Never cite the same element in two items of the same query.

If you leave out an equivalent copy, a retriever that finds that copy is wrongly scored as a
miss. Finding every copy is mandatory (see workflow step 4).

### Relevance grades
- 3 = directly and completely answers the question (or a required part of it)
- 2 = important supporting evidence
- 1 = useful background context
Every non-negative query needs at least one grade-3 item. Headings are never grade 3.

────────────────────────────────────────────────────────────────────────

## CATEGORIES (exact counts)

1. exact_numeric (10) - a specific number. Prefer numbers printed in tables.
2. trend_explanation (8) - why a metric changed; grade 3 = the explaining paragraph.
3. risk_factor (8) - a specific disclosed risk. At least one grade-3 element MUST be in
   section "Item 1A". (Paragraphs from Item 1 or Item 7 may only be supporting.)
4. accounting_policy (8) - how something is accounted for. Grade 3 should be the policy
   text in the notes (Item 8) and/or its MD&A critical-estimate duplicate, not an
   unrelated paragraph that merely mentions the topic.
5. business_description (6) - products, segments, operations, properties.
6. cross_section (6) - needs evidence from at least TWO different `section` values
   (e.g., Item 1A + Item 7, Item 1 + Item 7, Item 1C + Item 1A). Two elements from the
   same section do not qualify.
7. cross_document (6) - compares companies. "doc_id": null; items from at least two
   doc_ids; every item carries its own doc_id.
8. negative (4) - the answer is NOT in any filing. "relevant": [] and
   "expected_answer": "Not disclosed in these filings" (exactly this string).
9. table_structure (4) - needs reading a multi-row/multi-column table (grade 3 = that
   table element).

## DISTRIBUTION (exact targets)
- Difficulty: 20 easy, 25 medium, 15 hard.
- Companies (excluding cross_document and negative): 17 / 17 / 16 across AAPL, MSFT, TSLA
  (any assignment of 17/17/16).
- Sections: each of Item 1, Item 1A, Item 7, Item 8 cited by at least 2 queries; at least
  one query must cite Item 1C, Item 2, or Item 15.
- At least 20 queries with a table as a grade-3 element and at least 20 with a paragraph
  as a grade-3 element.
- No near-duplicate questions across companies (e.g., not "total revenue" for all three;
  segment names for two companies counts as a repeat). Cross-document queries are the only
  place where the same metric is asked for several companies.

────────────────────────────────────────────────────────────────────────

## MANDATORY WORKFLOW (one query at a time)

Do NOT draft many queries and verify them later. For each query, complete all steps before
starting the next. Discard any candidate that fails a step and pick different evidence.

1. Pick the evidence. Open the full JSON of each candidate element (by doc_id + order).
   Read the complete text or table markdown, not a snippet.

2. Write the query: natural analyst wording, standalone, names the company and period,
   one clearly supported answer, no internal identifiers ("order 357", "table t016").

3. Write the expected answer from the element text only:
   - copy numbers exactly as printed (with units: "$416,161 million", "46.7 GWh", "18.0%")
   - no rounding, no unit conversion, no facts missing from the cited elements
   - no causal claims unless the filing states the cause
   - do not echo dates the filing doesn't support
   - percentages and growth rates must come from the filing, not your arithmetic
     (otherwise list them in derived_values with every operand cited)

4. Find every copy. Search the WHOLE document for:
   - each key number in the answer (search "416,161" and "416.2" style variants)
   - a distinctive 6-10 word phrase from each cited paragraph
   Add every element that states the same fact to the same item. Add genuinely different
   supporting evidence as separate items with lower grades.

5. Verify each citation:
   - the order exists in that doc_id's file
   - the element's type, section and content match your `reason`
   - for tables, the cited row/column really contains the value; cite the table_id in
     the reason
   - no heading at grade 3
   - category rules above (Item 1A for risk_factor, ≥2 sections for cross_section, etc.)

6. For negative queries: search all three documents with several phrasings and synonyms
   (e.g., "employees", "headcount", "full-time"; "bitcoin", "digital assets", "crypto").
   If ANYTHING in any filing answers the question, it is not a negative - pick another.
   Prefer questions that are specific and plausible but genuinely unanswerable (e.g., a
   product-level unit count the company never discloses).

7. Append the record only after steps 1-6 pass.

────────────────────────────────────────────────────────────────────────

## KNOWN PITFALLS FROM THE PREVIOUS (v1) DATASET - DO NOT REPEAT

- Off-by-one orders: the cited order pointed to the neighboring paragraph (e.g., "cost of
  revenue increased" instead of "revenue increased"). Always re-open the cited order.
- Invented or miscopied numbers (an R&D figure that appears nowhere; wrong accumulated
  depreciation and net PP&E; 75.3% instead of 75.4%; $94,830M instead of $94,827M).
- "Negative" questions that the filings actually answer (dividend per share, bitcoin
  holdings, employee count).
- Expected answers mentioning facts absent from the filing (e.g., a product feature the
  filing never names; a "Level 3" category the table does not contain).
- Grade-3 elements that only touch the topic (basis-of-presentation text for a revenue
  recognition question; a metric-definition table for a revenue-breakdown question; an
  interest-rate-sensitivity table for a fair-value-levels question).
- risk_factor queries whose grade-3 evidence was in Item 1 or Item 7.
- cross_section queries whose evidence came from a single section.
- Only one of several identical copies labeled (financial statement vs MD&A table vs
  note; MD&A critical estimates vs Note 1 text).
- Cross-document questions with one company missing evidence.
- `reason` text describing the wrong element type (calling a paragraph a "heading").

────────────────────────────────────────────────────────────────────────

## FINAL VALIDATION (required before you finish)

Run:

uv run python scripts/eval_scripts/validate_retriever_goldens.py --golden <your output file>

It checks: schema and ids, orders exist, doc_id rules, no heading at grade 3, every number
in expected_answer appears in a cited element (or derived_values), cross_section spans ≥2
sections, risk_factor has grade 3 in Item 1A, category/difficulty/company/section/type
distribution, cited elements appear verbatim in the raw HTML, and it warns about unlabeled
near-duplicate paragraphs and unlabeled tables containing an answer value.

- Fix every ERROR and re-run until it reports `ERRORS (0)`.
- Review every WARNING. Add the flagged element to the matching item if it is a true copy
  of the cited fact. Leave it only if it is a false positive (e.g., a prior-year value, a
  table that shows only a total but not the asked breakdown, a similar-looking paragraph
  about a different metric), and list those decisions in your final message.

Finish with a short summary: output path, validator result, distribution, and the list of
warnings you deliberately left.
