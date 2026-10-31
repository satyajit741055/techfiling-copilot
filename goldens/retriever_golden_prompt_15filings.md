# Prompt: retrieval golden dataset for the 15-filing corpus (AAPL / MSFT / TSLA, FY2021–FY2025)

You are a financial-domain retrieval-evaluation expert working inside this repository:

C:\Users\Satyajit_Khot\Downloads\techfiling-copilot

The repository is a RAG system over SEC 10-K filings. Your job is to build a **golden dataset of
105 retrieval test cases**. It will be used to measure retrieval quality (Hit@k, MRR@k,
Recall@k, Precision@k, nDCG@k, group recall, filter violations) with
`eval/hybrid_rerank_eval.py` and `eval/dense_retriever_eval.py`, against Qdrant collections
built from these same filings.

The dataset will be trusted without a second human labelling pass. **Correctness matters more
than speed.** Every order number, number, quote and claim must come from tool output that you
have actually seen in this session. If you cannot verify something, drop the candidate and
choose different evidence. Never fill a gap from memory.

You have tool access (file reading, search, shell). Use it constantly.

---

## 0. Ground rules (read twice)

1. **One test case at a time.** Plan the whole dataset first (§7, Phase 1). Then build, verify,
   validate and log one record before you start the next. Never draft several records and
   "verify later".
2. **The parsed elements are the only source of facts:** `data/parsed/*.elements.jsonl`.
   - Do not use external knowledge, model memory, web search, or any existing golden file
     (including `goldens/old_goldens_1_year_for_3_company/`) as a source of facts.
   - `data/parsed/*.facts.jsonl` (XBRL facts) may help you *locate* a number, but it is never
     evidence and is never cited.
3. **Copy, never retype.** Copy orders, table_ids, numbers and quotes from tool output.
4. **Do not modify** any file outside these four:
   - the output JSONL;
   - the work log;
   - `goldens/_work/` (any scratch files you need);
   - nothing else. Don't change the validator, the lookup helper, the eval scripts or `data/`.
5. If a rule here conflicts with what you see in the data, stop, record the conflict in the work
   log, and choose a different candidate. Don't bend the rule.

---

## 1. The corpus

15 filings: 3 companies × 5 fiscal years. The fiscal year in each `doc_id` is the company's
fiscal year.

| doc_id | Company | Fiscal year ends | Elements |
|---|---|---|---|
| AAPL_10-K_2021 … AAPL_10-K_2025 | Apple | last Saturday of September (2021-09-25, 2022-09-24, 2023-09-30, 2024-09-28, 2025-09-27) | ~625–650 each |
| MSFT_10-K_2021 … MSFT_10-K_2025 | Microsoft | June 30 (MSFT fiscal 2025 = July 2024–June 2025) | ~975–1,080 each |
| TSLA_10-K_2021 … TSLA_10-K_2025 | Tesla | December 31 (calendar year) | ~945–1,060 each |

Files per filing in `data/parsed/`:
- `<doc_id>.elements.jsonl`: the elements you cite.
- `<doc_id>.document.json`: metadata (`ticker`, `fiscal_year`, `period_end`, `source_file`).
- Raw SEC HTML is in `data/raw_html/`. The validator checks your cited elements against it; you
  don't need to read it.

Each line of an elements file is one element:

```json
{
  "doc_id": "AAPL_10-K_2025",
  "order": 357,                     // stable position within the filing: THIS is what you cite
  "type": "table" | "paragraph" | "heading",
  "part": "Part II",
  "section": "Item 8",              // Item 1, 1A, 1B, 1C, 2, 3, 5, 7, 7A, 8, 9A, 15, Cover Page …
  "section_title": "...",
  "text": "...",                    // paragraphs/headings; null for tables
  "table": {                        // null for non-tables
    "table_id": "AAPL_10-K_2025_t023",
    "title": "CONSOLIDATED STATEMENTS OF OPERATIONS",
    "units": "...",
    "markdown": "| Line item | Years ended September 27, 2025 | ... |\n| Net sales > Total net sales | 416,161 | 391,035 | 383,285 |",
    "rows": [...]
  }
}
```

Things in the data you must know about:
- `order` is the element's `order` field, never a line number in the file. Orders are unique
  within a filing only. The same order number in two filings refers to unrelated elements.
- Table markdown rows are written `| Parent > Child | value | value |`. Column headers are
  usually years or period descriptions. Some tables have generic headers (`Column 1`,
  `Column 2`, …). For those, the year of each column is **not** stated in the table. Don't use
  such a table as the only relevance-3 evidence for a year-specific value, unless an adjacent
  paragraph or table in the same filing confirms the column order.
- Curly apostrophes, dashes and some symbols appear as `�` (e.g. `Company�s`, `Note 2 � Revenue`).
  Copy quotes exactly as the tool prints them. The validator compares words only, so `�` and
  punctuation are ignored, but the words must match.
- Every 10-K repeats prior-year numbers as comparative columns. The FY2024 income statement shows
  FY2024, FY2023 and FY2022, and the balance sheet shows two year-ends. This matters for
  multi-filing and open-corpus cases (§5).
- Item 1C (Cybersecurity) exists only from FY2023 onward (MSFT from FY2024).

### Lookup helper (use it for every lookup)

```
uv run python scripts/eval_scripts/golden_lookup.py show   AAPL_10-K_2025 357      # full element (text / table markdown, table_id)
uv run python scripts/eval_scripts/golden_lookup.py around AAPL_10-K_2025 357 3    # neighbours ±3, one line each
uv run python scripts/eval_scripts/golden_lookup.py find "Total net sales" --doc AAPL_10-K_2025
uv run python scripts/eval_scripts/golden_lookup.py find "416,161" --ticker AAPL   # all 5 filings of one company
uv run python scripts/eval_scripts/golden_lookup.py find "digital assets"          # all 15 filings
uv run python scripts/eval_scripts/golden_lookup.py toc TSLA_10-K_2023 "Item 1A"   # section headings
```

`find` is a case-insensitive substring search over paragraph text and table markdown. Use
`--max N` to see more hits; the last line always prints the total.

---

## 2. How the evaluator scores your records (why the rules exist)

- The indexed chunks list the parsed elements they contain (`element_orders`) and their
  `doc_id`. A retrieved chunk is relevant to an item when it is from the item's `doc_id` **and**
  shares at least one order with the item's `value` list.
- Each relevant item is credited **at most once**. A retriever that finds any one listed copy of
  a fact gets the credit. Recall = items found ÷ items listed. nDCG uses the grades.
- **Retrieval scope:** each record also tells the retriever where to search.
  - `doc_id`: search only that filing.
  - `filters`: a single value, a list (`["AAPL", "MSFT"]`, matches any) or a range
    (`{"gte": 2022, "lte": 2024}`).
  - `group_by` + `groups`: one search per group, e.g. one per company.
  - `group_by` without `groups`: the retriever picks the top `group_limit` groups itself.
- **Grouped scoring:** metrics are computed per group and averaged. A grouped query counts as a
  hit only if **every** expected group gets a hit, so every group must have relevance-3 evidence.
- **Negative records** (no relevant items) are counted but not scored.

Consequences:
- If you leave out an equivalent copy, a retriever that finds that copy is wrongly scored as a
  miss.
- If you list evidence from a filing outside the scope, that evidence can never be retrieved.
- If an expected group has no evidence, the query can never count as a hit.

---

## 3. Output files

1. **Dataset:** `goldens/golden_retriever_15filings.jsonl`
   - One JSON object per line. No array wrapper, no Markdown fences, no comments.
   - UTF-8. Ids are `q001` … `q105`, sequential, with no gaps.
   - If the file already exists when you start, check the work log first. If it's your own
     earlier progress on this task, resume after the last record whose verification is logged.
     Otherwise don't overwrite it: write to `goldens/golden_retriever_15filings_new.jsonl`
     instead, and use that path everywhere below.
2. **Work log:** `goldens/golden_retriever_15filings_worklog.md`. It holds the coverage plan,
   one verification entry per record, the final re-verification table and the warning
   decisions (format in §8).

---

## 4. Record schema

Required on every record: `id`, `query`, `scope_type`, `doc_id`, `category`, `difficulty`,
`expected_answer`, `relevant`.

Optional fields, only where the rules call for them: `filters`, `group_by`, `groups`,
`group_limit`, `derived_values`, `negative_search_terms`, `notes`. Omit them, or set them to
`null`, when not used. No other fields.

```json
{
  "id": "q001",
  "query": "What were Apple's total net sales in fiscal 2025?",
  "scope_type": "single_doc",
  "doc_id": "AAPL_10-K_2025",
  "category": "exact_numeric",
  "difficulty": "easy",
  "expected_answer": "$416,161 million.",
  "relevant": [
    {
      "match_by": "element_orders",
      "value": [357, 279, 406, 533, 266, 529],
      "relevance": 3,
      "doc_id": "AAPL_10-K_2025",
      "evidence": "Net sales > Total net sales | 416,161",
      "reason": "Total net sales row (2025 column) in Statements of Operations AAPL_10-K_2025_t023; same value in Products and Services Performance AAPL_10-K_2025_t016, Note 2 Revenue AAPL_10-K_2025_t028, Note 13 segment tables AAPL_10-K_2025_t052 and AAPL_10-K_2025_t049, Segment Operating Performance AAPL_10-K_2025_t015."
    }
  ]
}
```

(This example and the one in §5.3 pass the validator. **Don't reuse them as dataset records.**)

### Field rules

| Field | Rule |
|---|---|
| `query` | Natural analyst wording. Standalone. Names the company (Apple / Microsoft / Tesla) and the fiscal year(s). Never mentions orders, table ids, doc_ids, "chunk" or "element". |
| `scope_type` | One of `single_doc`, `open_corpus`, `cross_year`, `cross_company`, `cross_company_year`, `negative` (§5). |
| `doc_id` | The filing, for `single_doc` and doc-scoped `negative`. `null` for every other scope type. |
| `filters` | Object; keys only `ticker`, `fiscal_year`, `doc_id`, `form`. Tickers in upper case. `fiscal_year` values are **integers** (`2023`, not `"2023"`). Range: `{"gte": 2022, "lte": 2024}`. List: `[2021, 2025]`. |
| `group_by` | `"ticker"`, `"fiscal_year"` or `"doc_id"`. |
| `groups` | List of group values (never a bare string), at most 6, no duplicates. Integers for `fiscal_year`. Never also put the `group_by` field in `filters`. |
| `group_limit` | Only for discovery-mode grouped queries (`group_by` without `groups`). Positive integer, ≥ number of groups with evidence. |
| `category` | Content type (§6). `negative` only for negative records. |
| `difficulty` | `easy` / `medium` / `hard` (§6). |
| `expected_answer` | Written only from the cited elements (§7, step 4). |
| `relevant[].match_by` | Always `"element_orders"`. |
| `relevant[].value` | Integer orders, **all from the item's one `doc_id`**. All equivalent copies of the item's fact in that filing (§5.1). |
| `relevant[].relevance` | 3, 2 or 1 (§5.2). |
| `relevant[].doc_id` | **Required on every item**, including single-filing records. |
| `relevant[].evidence` | A verbatim quote of at least 4 words from one of the item's elements that shows the item's fact. For relevance 3 it should contain the answer value or the answering sentence. For tables, copy one markdown row exactly, e.g. `Net sales > Total net sales \| 416,161`. |
| `relevant[].reason` | What each cited element is and states: element type, section, **every cited table_id**, and the row/column for tables. |
| `derived_values` | Only for numbers in `expected_answer` that you calculated rather than copied (e.g. a difference between two years). Every operand must be in a cited element. Use rarely. |
| `negative_search_terms` | Negatives only. At least 4 terms or phrases you searched (§5.6). |
| `notes` | Optional short note, e.g. "Tables t016/t028 also contain the 2024 comparative; not cited because the query is about 2025". |

---

## 5. Key concepts and scope rules

### 5.1 One item = one piece of information, in one filing

An item is one fact (or one piece of evidence) **within one filing**, not one element. 10-Ks
repeat the same fact in several places, so:

1. Put **all equivalent copies in that filing** into the same item's `value`:
   - the same number in several tables (MD&A table, financial statement, note table, segment
     note, revenue disaggregation, reconciliations);
   - verbatim or near-verbatim paragraphs repeated across sections (MD&A Critical Accounting
     Estimates often duplicates Note 1/Note 2; Item 1 often duplicates Item 1A or Item 8);
   - an MD&A sentence that states the same figure as a table (e.g. "$14.7 billion" next to a
     table showing 14,747);
   - a paragraph the parser split at a page break (the second half starts mid-sentence). List
     both halves.
2. Put different facts in separate items, each with its own grade.
3. The same fact in **another filing** is a separate item with that filing's `doc_id`, and only
   when that filing is in scope and the rules below allow it.
4. Never cite the same element in two items of one record.

### 5.2 Relevance grades

- **3**: directly and completely answers the question, or a required part of it.
- **2**: important supporting evidence.
- **1**: useful background context.

Every positive record needs at least one grade-3 item. Every in-scope filing of a multi-filing
record needs its own grade-3 item. Headings are never grade 3.

### 5.3 Single-filing scope types

- **`single_doc`**: `doc_id` set; no `filters`/`group_by`. All items from that filing. This is
  how the app normally retrieves.
- **`open_corpus`**: `doc_id` null; no `filters`/`group_by`. The retriever searches all 15
  filings, so it must pick the right company **and year** by itself. Evidence comes from exactly
  one filing.
  - **Uniqueness rule:** the answer (its key number or its distinctive sentence) must appear in
    **no other filing**. Check with `find` across all 15 filings and log the hit counts.
  - Comparative columns break this rule for most financial-statement numbers. FY2023 revenue
    also appears in the FY2024 and FY2025 filings. Good open-corpus candidates are:
    - point-in-time facts reported only for the current year (e.g. headcount, holders of record,
      a dated event described only in that year's filing);
    - year-specific narrative;
    - FY2025 figures, which no later filing repeats. Still verify with `find`.

Multi-filing example (passes the validator; don't reuse it):

```json
{"id": "q002", "query": "How did Apple's total net sales in fiscal 2021 compare with fiscal 2025?",
 "scope_type": "cross_year", "doc_id": null, "filters": {"ticker": "AAPL"},
 "group_by": "fiscal_year", "groups": [2021, 2025], "category": "exact_numeric", "difficulty": "medium",
 "expected_answer": "Fiscal 2021: $365,817 million. Fiscal 2025: $416,161 million.",
 "relevant": [
  {"match_by": "element_orders", "value": [350, 259, 276, 411, 561], "relevance": 3, "doc_id": "AAPL_10-K_2021",
   "evidence": "Net sales > Total net sales | 365,817",
   "reason": "FY2021 filing: Total net sales (2021 column) in AAPL_10-K_2021_t023; copies in AAPL_10-K_2021_t015, AAPL_10-K_2021_t016, AAPL_10-K_2021_t029, AAPL_10-K_2021_t054."},
  {"match_by": "element_orders", "value": [357, 279, 406, 533, 266, 529], "relevance": 3, "doc_id": "AAPL_10-K_2025",
   "evidence": "Net sales > Total net sales | 416,161",
   "reason": "FY2025 filing: Total net sales (2025 column) in AAPL_10-K_2025_t023; copies in AAPL_10-K_2025_t016, AAPL_10-K_2025_t028, AAPL_10-K_2025_t052, AAPL_10-K_2025_t015, AAPL_10-K_2025_t049."}]}
```

(In the real file every record is on one line.)

### 5.4 Multi-filing scope types and mechanisms

`doc_id` is always `null`; the scope comes from `filters` and/or `group_by`.

**In-scope filings** are the filings that match `filters` and, when `groups` is given, belong to
one of the groups. **Every in-scope filing needs at least one grade-3 item, and no item may come
from outside the scope.** Choose the scope so it covers exactly the filings the question needs.

| scope_type | Covers | Allowed mechanisms (exact counts in §6) |
|---|---|---|
| `cross_year` | one company, ≥ 2 fiscal years | `grouped_fiscal_year`: `filters {"ticker": T}`, `group_by "fiscal_year"`, `groups [years]` · `range`: `filters {"ticker": T, "fiscal_year": {"gte": a, "lte": b}}` · `list`: `filters {"ticker": T, "fiscal_year": [y1, y2, …]}` |
| `cross_company` | ≥ 2 companies, one fiscal year | `grouped_ticker`: `filters {"fiscal_year": Y}`, `group_by "ticker"`, `groups [tickers]` · `grouped_doc_id`: `group_by "doc_id"`, `groups [doc_ids of one year]` · `list`: `filters {"ticker": [T1, T2, …], "fiscal_year": Y}` |
| `cross_company_year` | ≥ 2 companies and ≥ 2 years | `grouped_ticker` with a fiscal_year **range** filter · `grouped_doc_id` with 4–6 doc_ids · `discovery`: `group_by "ticker"`, **no** `groups`, `group_limit 3`, `filters` on `fiscal_year` |

### 5.5 Year rules for multi-filing records (comparative columns)

1. A fact about fiscal year Y is evidenced by **the FY-Y filing's own current-year value or
   text**. Read the column header to confirm you have the right column.
2. Don't cite a comparative column from another filing as evidence for Y. In grouped queries it
   would sit in the wrong group. In flat queries it would let one filing answer for several.
3. For `range` and `list` cross_year queries, pick facts that are **not** all present in a single
   in-scope filing. Otherwise one later filing answers the whole question while the grading
   expects one item per year. Use:
   - facts each filing reports only for its own year (headcount, point-in-time counts,
     year-specific narrative such as a risk factor or segment description that changed);
   - or years further apart than any one filing's comparatives (a 10-K shows 3 years of income
     statement and 2 year-ends of balance sheet; e.g. `[2021, 2025]`).

   Log the `find` result that proves this.
4. Grouped queries:
   - `groups` hold 2–6 values, and every group needs grade-3 evidence;
   - at least 2 of the 6 `grouped_fiscal_year` queries use all five years `[2021, 2022, 2023, 2024, 2025]`;
   - every fiscal year 2021–2025 appears in at least one `cross_company` query.
5. Questions must make sense for every company in scope. Use comparable metrics or disclosures
   (e.g. revenue, R&D expense, employees, cybersecurity governance, revenue recognition
   approach, share repurchases). Don't ask Microsoft about vehicle deliveries.
6. `expected_answer` gives one labelled part per filing: "Apple FY2023: …; Tesla 2023: …".

### 5.6 Negative records

- `scope_type` and `category` are both `"negative"`. `relevant` is `[]`.
- `expected_answer` is exactly `Not disclosed in these filings`.
- `negative_search_terms` holds ≥ 4 terms. Search them **all** with `find` (synonyms, units,
  abbreviations) and log the hit counts.
- 4 doc-scoped negatives (`doc_id` set, one per company at least, different years). The fact
  must be absent from **all five filings of that company**, not just from the scoped year.
  "Disclosed in 2022 but not in 2023" is a year-mismatch trick and isn't allowed.
- 2 open negatives (`doc_id` null, no filters): absent from all 15 filings.
- They must be specific and plausible, but genuinely unanswerable, e.g. a product-level unit
  count or a metric the company never discloses.
- Don't use anything that is disclosed, including dividend per share, employee counts,
  digital-asset holdings, store counts, cybersecurity incidents or customer concentration.
  Search first.
- If any element answers the question, even partially, it's not a negative. Pick another.

---

## 6. Dataset composition (exact targets; the validator checks them)

### 6.1 Scope types: 105 records

| scope_type | Count |
|---|---|
| single_doc | 69 |
| open_corpus | 6 (2 per company, different years) |
| cross_year | 12 (4 per company) |
| cross_company | 9 |
| cross_company_year | 3 |
| negative | 6 (4 doc-scoped, 2 open) |

### 6.2 Coverage of filings

`single_doc` + `open_corpus` = 75 records, **exactly 5 per filing**, so all 15 filings are
tested equally. Every filing must also appear in at least one multi-filing record. Check it in
your plan.

### 6.3 Categories for single_doc + open_corpus (75)

Recommended split per company (25): exact_numeric 5, trend_explanation 4, risk_factor 4,
accounting_policy 3, business_description 3, cross_section 3, table_structure 3.

| category | Total | Definition and rules |
|---|---|---|
| exact_numeric | 15 | A specific number. Prefer table values. Every copy of the value in the filing goes in the item. |
| trend_explanation | 12 | Why a metric changed. Grade 3 is the paragraph that states the reason; the figure's table can be grade 2. Don't state a cause the filing doesn't state. |
| risk_factor | 12 | A specific disclosed risk. At least one grade-3 element in **Item 1A**; Item 1/7 text can only be supporting. |
| accounting_policy | 9 | How something is accounted for. Grade 3 is the policy text in the notes (Item 8) and/or its MD&A critical-estimate duplicate, not a paragraph that only mentions the topic. |
| business_description | 9 | Products, segments, operations, properties, human capital, customers, competition. |
| cross_section | 9 | Needs evidence from **≥ 2 different `section` values within the same filing** (e.g. Item 1A + Item 7, Item 1C + Item 1A, Item 1 + Item 8). |
| table_structure | 9 | Needs reading a multi-row/multi-column table (a specific row × column, or several rows). Grade 3 is that table. |

Multi-filing records use the same category names, except `cross_section` and `negative`. Mix
them, with at least 8 of the 24 being exact_numeric or table_structure and at least 8 being
narrative (risk_factor, accounting_policy, business_description, trend_explanation).

### 6.4 Mechanisms (exact)

| scope_type | Mechanism counts |
|---|---|
| cross_year (12) | grouped_fiscal_year 6, range 3, list 3 |
| cross_company (9) | grouped_ticker 5, grouped_doc_id 2, list 2 |
| cross_company_year (3) | grouped_ticker (fiscal_year range) 1, grouped_doc_id (4–6 filings) 1, discovery 1 |

### 6.5 Difficulty: easy 30, medium 45, hard 30

Suggested split: single_doc+open_corpus 28/32/15, multi-filing 0/11/13, negatives 2/2/2.

- **easy:** the answer is stated in one element, and the query wording is close to the filing's
  wording.
- **medium:** paraphrased wording; or the answer needs a specific row/column; or the fact appears
  in several copies; or two groups.
- **hard:** wording differs from the filing (synonyms), multi-part, three or more filings or
  groups, or a detail buried in a long note.

### 6.6 Other coverage targets

- At least 30 records with a **table** at grade 3 and at least 30 with a **paragraph** at
  grade 3.
- Item 1, Item 1A, Item 7 and Item 8 are each cited by at least 5 records.
- At least 3 records cite Item 1C, Item 2, Item 7A or Item 15.
- No duplicate query text.
- No near-duplicates in single-filing records:
  - the same metric for the same company in two different years belongs in `cross_year`;
  - the same metric for different companies is allowed at most twice across all single-filing
    records;
  - comparisons belong in the multi-filing records.

---

## 7. Mandatory workflow

### Phase 1: plan (before writing any record)

Write a **coverage matrix** in the work log: one row per slot q001–q105 with
`id | scope_type | category | difficulty | filing(s) / scope | mechanism | topic idea`.

Order the slots as follows (it keeps related work together):

| Ids | Content |
|---|---|
| q001–q075 | single_doc and open_corpus, grouped by filing: AAPL 2021 … 2025, MSFT 2021 … 2025, TSLA 2021 … 2025, five per filing; open_corpus slots sit inside their filing's block |
| q076–q087 | cross_year |
| q088–q096 | cross_company |
| q097–q099 | cross_company_year |
| q100–q105 | negative |

Below the matrix, tally every target in §6 and show that the plan meets it exactly. Topic ideas
are placeholders; you'll change them when the evidence doesn't support them. Update the matrix
whenever you change a slot.

Before planning, browse each filing's section headings with `toc` so the topic ideas are
realistic. Don't invent facts while planning; topics only.

### Phase 2: build one record at a time

For each slot, in order, do all of these steps **before** starting the next slot. If a step
fails, discard the candidate and try other evidence or a different topic. Update the matrix and
note the change in the log.

1. **Find the evidence.**
   - Use `toc` / `find` to locate candidates.
   - Open every candidate with `show` and read the **full** text or table markdown.
   - Use `around` to check neighbouring elements; the answer is sometimes in the next
     paragraph.
   - For tables, confirm the column header (the year) and the row label of the value.
2. **Check the scope.** The evidence must sit in the filing(s) the slot's scope covers. For
   multi-filing records apply §5.4 and §5.5. For open_corpus apply the uniqueness rule (§5.3).
3. **Write the query** following the field rules (§4).
4. **Write `expected_answer` from the cited text only.**
   - Copy numbers exactly as printed, with units ("$416,161 million", "18.0%", "1.81 million
     vehicles"). Don't round or convert units.
   - Include no fact that isn't in a cited element.
   - State no cause the filing doesn't state.
   - Percentages and changes must be printed in the filing. Otherwise list them in
     `derived_values` with every operand cited.
   - Keep it short: 1–3 sentences, or one part per filing.
5. **Find every copy.** In each filing involved, search:
   - every key number in the answer, including variants (`416,161`, `416.2`, `416.161`);
   - a distinctive 6–10 word phrase from each cited paragraph.

   Add every element stating the same fact to the item, and genuinely different supporting
   evidence as separate lower-graded items. Log each search and its hit count.
6. **Write the items:** `doc_id`, `value`, `relevance`, `evidence` copied from `show` output,
   and `reason` naming every table_id.
7. **Self-check the record:**
   - each order exists in its filing, and its type/section match the reason;
   - the evidence appears in one of the item's elements;
   - every number in the answer appears in a cited element (correct year column);
   - no heading at grade 3;
   - category rules hold (Item 1A for risk_factor, ≥ 2 sections for cross_section, a table for
     table_structure);
   - for multi-filing records, every in-scope filing has grade 3 and nothing is out of scope.
8. **Append** the single JSON line to the dataset file.
9. **Validate the partial file.** It must report `ERRORS (0)`:
   ```
   uv run python scripts/eval_scripts/validate_retriever_goldens_v3.py --golden goldens/golden_retriever_15filings.jsonl --partial --skip-html --skip-loader
   ```
   - Fix any error in the record you just wrote before continuing.
   - Read every warning for that record. Add the flagged element if it's a true copy, otherwise
     note in the log why it's a false positive.
10. **Log** the record's verification entry (§8).

**Every 15 records**, run the full checks (raw-HTML and eval-loader):
```
uv run python scripts/eval_scripts/validate_retriever_goldens_v3.py --golden goldens/golden_retriever_15filings.jsonl --partial
```

### Phase 3: independent re-verification (mandatory, after all 105 exist)

Treat the file as if someone else wrote it. For **every** record, q001 to q105, one at a time:

1. Re-open every cited element with `show`.
2. Confirm that:
   - the query is answered by the grade-3 evidence;
   - each number in `expected_answer` is in the right row and year column;
   - each `evidence` quote is in its element;
   - each `reason` describes the element correctly (type, section, table_id).
3. Re-run `find` for each key number and phrase in every in-scope filing, to catch copies you
   missed.
4. For open_corpus records, re-run the uniqueness check across all 15 filings.
5. For negatives, search **two new synonyms** that aren't in `negative_search_terms`, and add
   them to the list.
6. For multi-filing records, re-check that the scope covers exactly the filings with evidence,
   and that no comparative column is cited as evidence for another year.
7. Fix anything wrong, and record `q### | verified | changes` in the re-verification table.

Then run the full validator, without `--partial`:
```
uv run python scripts/eval_scripts/validate_retriever_goldens_v3.py --golden goldens/golden_retriever_15filings.jsonl
```
- It must report **`ERRORS (0)`**. It also confirms that `eval/hybrid_rerank_eval.py` can load
  the file.
- Review **every** warning:
  - add flagged elements that are true copies of a cited fact;
  - fix query wording warnings;
  - fix distribution warnings by revising slots (then re-verify those records).

  You may leave a warning only when it's a false positive, e.g. a prior-year comparative, a
  table that shows only a total and not the asked breakdown, a similar-looking paragraph about a
  different metric, or a negative term that occurs in an unrelated context. List each one with
  its reason in the work log.
- Re-run until errors are 0 and every remaining warning is explained.

---

## 8. Work log format (`goldens/golden_retriever_15filings_worklog.md`)

```
# Work log: retrieval goldens (15 filings)

## Coverage plan
<matrix q001–q105> + <target tally>

## Records
### q007: single_doc AAPL_10-K_2022, accounting_policy, medium
- Candidates opened: AAPL_10-K_2022#412 (paragraph, Item 8), #301 (paragraph, Item 7) ...
- Searches: "deferred revenue" --doc AAPL_10-K_2022 -> 9 hits (#..., #...); "<phrase>" -> 2 hits
- Copies added: #301 (MD&A critical estimate duplicate of #412)
- Rejected: #415 (mentions topic, does not state the policy)
- Scope / uniqueness / comparative check: n/a | <result>
- Validator (partial): 0 errors; warnings: <none | handled: ...>

## Re-verification
| id | verified | changes |

## Final validator result and warning decisions
```

---

## 9. Known pitfalls from earlier datasets: don't repeat them

- **Off-by-one orders:** the cited order was the neighbouring paragraph (e.g. "cost of revenue
  increased" instead of "revenue increased"). Always `show` the exact order you cite.
- **Invented or miscopied numbers:** an R&D figure that appeared nowhere; 75.3% written instead
  of 75.4%; $94,830M instead of $94,827M.
- **Wrong year column:** a prior-year comparative copied as the current-year value. Read the
  header.
- **Fake negatives:** "negative" questions the filings actually answer (dividend per share,
  digital-asset holdings, employee count).
- **Unsupported answer text:** expected answers with facts absent from the filing (a product
  feature the filing never names; a "Level 3" category the table doesn't contain).
- **Off-topic grade 3:** grade-3 elements that only touch the topic (basis-of-presentation text
  for a revenue-recognition question; a metric-definition table for a revenue-breakdown
  question).
- **Category rule breaches:** risk_factor grade 3 outside Item 1A; cross_section evidence from a
  single section.
- **Missed copies:** only one of several identical copies labelled (financial statement vs MD&A
  table vs note; critical estimates vs Note 1).
- **Incomplete groups:** a cross-company or cross-year record with a company or year missing
  evidence, or evidence from a filing outside the scope.
- **Wrong element type:** a `reason` calling a paragraph a "heading", or naming the wrong
  table_id.
- **Reused identifiers:** an order number taken from one filing and cited for another filing of
  the same company. Orders shift between years.
- **Old corpus:** stale doc_ids from the old corpus (e.g. `MSFT_10-K_2026` doesn't exist). Only
  the 15 doc_ids in §1 are valid.

---

## 10. Final message

Finish with a short summary:
- the output path, and confirmation that every record was re-verified (Phase 3);
- the full validator output (distribution lines, `ERRORS (0)`, warning count);
- the warnings you deliberately left, with reasons;
- any slots you changed from the original plan, and why.
