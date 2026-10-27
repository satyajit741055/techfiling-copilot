# Client brief — TechFiling Copilot

> **Portfolio context:** This brief uses a representative fictional client scenario to define a realistic product. It does not claim a completed engagement with an actual client.

## The client

**Aperture Technology Research** is a specialist equity-research team that follows large US technology and technology-led public companies. Its analysts produce company reports, financial models, earnings previews, risk assessments, and cross-company comparisons for portfolio managers and internal investment committees.

The team competes on accurate interpretation of primary-source disclosures, not on faster access to general market news. SEC filings—especially annual 10-K reports—are therefore central to its research.

## How Aperture creates value

- Analysts develop deep knowledge of a focused group of technology companies.
- They turn lengthy regulatory filings into concise, decision-useful research.
- They compare financial performance, business segments, capital allocation, and risks across companies and reporting periods.
- They support every material conclusion with evidence that can be verified against the original filing.
- Their credibility depends on numerical accuracy and on clearly separating disclosed facts from interpretation.

## The problem

SEC 10-K filings are authoritative but slow to use. A single filing can run to hundreds of pages of narrative disclosures, complex financial tables, footnotes, repeated layout elements, and Inline XBRL metadata.

Every analyst spends a large share of each week on repetitive document-intake work before any original analysis can begin:

- locating the relevant Item, note, or table;
- searching for a specific metric or disclosure;
- reconstructing meaning from multi-row financial tables;
- verifying units, periods, and negative values;
- copying supporting passages into research notes; and
- confirming that a conclusion is actually stated in the filing rather than inferred.

This intake work is boring, necessary, and repeated across analysts—multiple people read the same Apple or Microsoft 10-K every year. It is the single biggest drag on analyst output, and hiring more analysts does not fix it because the bottleneck scales linearly with coverage.

Ordinary keyword search misses semantically related language. General-purpose chatbots sound fluent but may rely on outside knowledge, cite the wrong passage, or return a plausible but wrong number. In financial research, a confident wrong number is worse than an honest "not disclosed."

## Who this helps, and how much time it saves

| User | Current pain | Expected benefit |
|---|---|---|
| Technology equity analyst | Hours per filing spent locating and extracting figures and disclosures | Finds and verifies the same evidence in seconds instead of minutes |
| Senior analyst / reviewer | Re-checking that a junior's number traces back to the filing | One-click access to the cited source passage |
| Research team as a whole | The same filings re-read independently by several people | Shared, repeatable answers over a curated corpus |

The target is a **meaningful reduction in source-document intake time per analyst per week**, freeing that time for original analysis. The intake step is the explicit bottleneck being removed; the tool accelerates research but does not replace analyst judgment.

## What the client wants

An internal research assistant—**TechFiling Copilot**—where any analyst can:

- select a company and annual filing;
- ask a question in plain English about that filing;
- get a concise answer grounded only in that filing's evidence;
- see the citations and the underlying source passages used;
- see the company, fiscal year, filing section, table title, and units for each source; and
- get an explicit "not disclosed / insufficient evidence" response when the filing does not support an answer.

## Example analyst questions

The system should handle filing-grounded questions such as:

1. What were Apple's total net sales in fiscal 2025, split between products and services?
2. How much did Apple spend on research and development, and how did the filing describe the change?
3. What operating segments did Microsoft report, and what revenue or operating income was disclosed for them?
4. What did Microsoft disclose about cloud and AI infrastructure investment or capacity constraints?
5. What revenue change did Tesla report, and which factors did management identify?
6. What customer, supplier, manufacturing, regulatory, or cybersecurity risks did each company disclose?
7. Does the filing disclose the requested value directly, or would answering require an unsupported inference?

(Once historical filings are added, the system should also answer multi-year questions such as how R&D spend or AI/cloud language changed across five fiscal years.)

## What "trust" means here

This is a research product; its entire value is being right and verifiable. The assistant must:

- **never invent facts or numbers;**
- **ground every factual claim in the selected filing;**
- **always cite, and show the underlying passage** so an analyst can verify in one click;
- **preserve units, periods, signs, and company identity;**
- **never mix evidence across companies or filings;** and
- **say so when the answer is not in the filing.**

A wrong but confident answer is worse than no answer.

## Why the MVP is technology-focused

The first release deliberately covers a narrow technology-oriented corpus. This is a scoping and quality decision, not a limitation of the system.

Different sectors emphasize different concepts and disclosure structures—technology filings center on product and services revenue, cloud and AI investment, customer concentration, supply chain, and R&D, whereas banking centers on net interest margin, loan-loss provisions, and regulatory capital. Supporting a new sector well means adding representative filings and verified test questions, not just loading more documents. Focusing on technology first gives a coherent user workflow and a controlled corpus the quality can actually be proven against before expanding.

## Corpus

**Version 1 (pilot):** one recent annual 10-K each for Apple (FY2025), Microsoft (FY2026), and Tesla (FY2025). Apple and Microsoft are the core information-technology examples; Tesla is a technology-led company included to test that the pipeline generalizes beyond near-identical filings.

**Planned expansion:** roughly five fiscal years of filings per company, then additional technology companies (e.g. Alphabet, Amazon, NVIDIA), enabling year-over-year and cross-company questions. These are roadmap items, not Version 1 promises.

## Constraints

- Domain: SEC 10-K annual reports for a controlled technology-oriented corpus.
- Source: Inline-XBRL HTML from SEC EDGAR (public domain).
- Evidence boundary: answers must come from the selected indexed filings.
- Accuracy priority: an honest refusal is preferable to a confident unsupported answer.
- Hosting: a modest cloud footprint suitable for an internal MVP.

## Out of scope for Version 1

- Investment recommendations, price targets, or stock picks
- News, social media, earnings-call, or market-price data
- Legal, accounting, or personalized financial advice
- Unsupported causal conclusions
- Real-time SEC ingestion
- Multi-year comparison before historical filings are indexed
- Enterprise SSO, billing, subscriptions, or a mobile app

## Definition of done for the MVP

A pilot group of technology-research analysts uses it for a week over the supported filings and can:

1. select a filing and ask a plain-English question;
2. get a concise answer or an explicit insufficient-evidence response;
3. inspect valid citations and the underlying source passages; and
4. verify the company, period, section, units, and table context behind each answer.

The pilot is a success if analysts report that it meaningfully reduces source-intake time **and** that they trust the cited evidence enough to base downstream analysis on it. Time saved without reliable, verifiable evidence does not count.