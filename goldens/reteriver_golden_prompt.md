# prompt for retrival golden dataset 
You are a financial-domain retrieval evaluation expert. I am building a RAG system
over SEC 10-K filings. I need you to generate a golden evaluation dataset that I will
use to programmatically measure retrieval quality (Hit Rate@k, Recall@k, MRR@k, nDCG@k).

I will provide you with three parsed element files (JSONL). Each line is one parsed
element from a 10-K filing with this schema:

{
  "doc_id":        "AAPL_10-K_2025",           // unique document identifier
  "order":         266,                         // stable sequential position in the document
  "type":          "table" | "paragraph" | "heading",
  "part":          "Part I" | "Part II" | "Front Matter" | ...,
  "section":       "Item 1" | "Item 1A" | "Item 7" | "Item 8" | ...,
  "section_title": "Risk Factors" | "Management's Discussion and Analysis..." | ...,
  "text":          "...",                       // paragraph/heading text (null for tables)
  "table": {                                   // null for non-table elements
    "table_id":  "AAPL_10-K_2025_t016",
    "kind":      "financial" | "textual",
    "title":     "Products and Services Performance",
    "units":     "dollars in millions",
    "intro":     "The following table shows...",
    "markdown":  "| Line item | 2025 | ... |",
    "rows": [
      {
        "label": "Total net sales",
        "row_type": "data" | "heading" | "total",
        "is_total": true,
        "cells": [{"text": "$416,161", "value": 416161.0, ...}]
      }
    ]
  }
}

The three documents are:
- AAPL_10-K_2025  (Apple, 624 elements)
- MSFT_10-K_2026  (Microsoft, 967 elements)
- TSLA_10-K_2025  (Tesla, 957 elements)

────────────────────────────────────────────────────────────────────────

TASK: Generate exactly 60 evaluation queries as a JSONL file. Each line must be a
valid JSON object with this exact schema:

{
  "id":              "q001",                    // sequential, zero-padded
  "query":           "What was Apple's total net sales in fiscal year 2025?",
  "doc_id":          "AAPL_10-K_2025",          // null for cross-document queries
  "category":        "exact_numeric",           // see categories below
  "difficulty":      "easy",                    // "easy" | "medium" | "hard"
  "expected_answer": "$416,161 million",        // concise human-readable answer
  "relevant": [
    {
      "match_by":    "element_orders",          // always "element_orders"
      "value":       [279],                     // list of `order` values that answer this query
      "relevance":   3,                         // 3 = directly answers, 2 = supporting, 1 = context
      "reason":      "Products and Services Performance table, Total net sales row"
    },
    {
      "match_by":    "element_orders",
      "value":       [357],
      "relevance":   2,
      "reason":      "Consolidated Statements of Operations, Total net sales row"
    }
  ]
}

────────────────────────────────────────────────────────────────────────

QUERY CATEGORIES — generate the specified count for each:

1. "exact_numeric" (10 queries)
   Questions asking for a specific number from a table.
   Example: "What was Tesla's total revenue in 2025?"
   Relevant elements: the table element(s) containing that number.

2. "trend_explanation" (8 queries)
   Questions asking why a metric changed or what drove a trend.
   Example: "Why did Apple's Greater China net sales decrease in 2025?"
   Relevant elements: the paragraph(s) explaining the trend.

3. "risk_factor" (8 queries)
   Questions about specific risks disclosed in Item 1A.
   Example: "What supply chain risks does Apple describe?"
   Relevant elements: the paragraph(s) in Item 1A discussing that risk.

4. "accounting_policy" (8 queries)
   Questions about accounting policies, footnotes, or financial statement notes (Item 8).
   Example: "How does Microsoft recognize revenue?"
   Relevant elements: the paragraph(s)/table(s) in the relevant note.

5. "business_description" (6 queries)
   Questions about business operations, products, or strategy (Item 1, Item 7).
   Example: "What are Tesla's main business segments?"
   Relevant elements: the paragraph(s) describing the business.

6. "cross_section" (6 queries)
   Questions requiring information from multiple sections to answer fully.
   Example: "How does Apple's reported goodwill relate to its acquisition strategy?"
   Relevant elements: elements from two or more different sections.

7. "cross_document" (6 queries)
   Questions comparing across companies. Set doc_id to null.
   Example: "Which company had the highest R&D expense?"
   Relevant elements: include relevant entries from multiple doc_ids.
   For cross-document, use this format in "relevant":
   [
     {"match_by": "element_orders", "value": [357], "relevance": 3,
      "reason": "Apple consolidated income statement", "doc_id": "AAPL_10-K_2025"},
     {"match_by": "element_orders", "value": [412], "relevance": 3,
      "reason": "Microsoft consolidated income statement", "doc_id": "MSFT_10-K_2026"}
   ]

8. "negative" (4 queries)
   Questions where the answer is NOT in any of the three filings.
   Example: "What was Apple's cryptocurrency holdings in 2025?"
   Set "relevant": [] and "expected_answer": "Not disclosed in these filings"

9. "table_structure" (4 queries)
   Questions that require understanding a multi-row or multi-column table layout.
   Example: "What are the fair value levels for Tesla's financial instruments?"
   Relevant elements: the specific table element.

────────────────────────────────────────────────────────────────────────

DISTRIBUTION REQUIREMENTS:

- Company coverage: approximately 20 queries per company (AAPL, MSFT, TSLA),
  excluding cross-document and negative queries.
- Section coverage: at least 2 queries must target each of these sections
  across the full dataset: Item 1, Item 1A, Item 7, Item 8.
  Include at least 1 query targeting Item 1C, Item 2, or Item 15 if content exists.
- Difficulty: roughly 20 easy, 25 medium, 15 hard.
- Type coverage: at least 20 queries should have a table element as the
  primary (relevance=3) answer. At least 20 should have a paragraph element
  as the primary answer.

────────────────────────────────────────────────────────────────────────

CRITICAL RULES:

1. Every `order` value in "relevant" MUST exist in the provided elements file
   for that doc_id. Verify each order value against the actual elements.
   Do NOT hallucinate order numbers.

2. For table elements, check that the specific row/cell you reference actually
   exists in the table's rows array. Cite the table_id in the reason field.

3. The "expected_answer" must be factually correct based on the element content.
   For numeric answers, include the exact number from the filing.

4. Each query must be a natural question a financial analyst would ask.
   Do not use internal identifiers like "order 357" or "element 266" in queries.

5. Queries must be diverse — no two queries should ask essentially the same
   thing about different companies (e.g., do not have "What was X's total
   revenue?" for all three companies).

6. For relevance grades:
   - 3 = this element directly and completely answers the question
   - 2 = this element provides important supporting evidence
   - 1 = this element provides useful background context
   Each query must have at least one element with relevance 3 (except negative queries).

7. Do NOT include "heading" type elements as relevance 3 answers.
   Headings are structural; only paragraphs and tables contain answer content.

8. Output ONLY the JSONL lines. No commentary, no markdown fences, no explanation.
   One valid JSON object per line, 60 lines total.

────────────────────────────────────────────────────────────────────────

Here are the three element files:

=== AAPL_10-K_2025.elements.jsonl ===
C:\Users\Satyajit_Khot\Downloads\pr_RAG\data\parsed\AAPL_10-K_2025.elements.jsonl

=== MSFT_10-K_2026.elements.jsonl ===
C:\Users\Satyajit_Khot\Downloads\pr_RAG\data\parsed\MSFT_10-K_2026.elements.jsonl

=== TSLA_10-K_2025.elements.jsonl ===
C:\Users\Satyajit_Khot\Downloads\pr_RAG\data\parsed\TSLA_10-K_2025.elements.jsonl



save dataset here C:\Users\Satyajit_Khot\Downloads\pr_RAG\goldens  -> C:\Users\Satyajit_Khot\Downloads\pr_RAG\goldens\golden_eval_dataset.jsonl