"""Curated metrics: analyst wording -> ordered XBRL concept aliases.

A topic matches a metric only when the whole topic (after normalisation) is one of the
metric's synonyms, optionally followed by a derived-value word ("revenue growth"). Topics
such as "iPhone revenue" or "cloud revenue" do not match, so they go to retrieval instead
of being answered with a company total.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

MetricKind = Literal["flow", "instant", "per_share", "shares"]


@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    concepts: tuple[str, ...]  # first concept with a value wins
    synonyms: tuple[str, ...]
    kind: MetricKind = "flow"

    @property
    def unit(self) -> str:
        return {"per_share": "USD/share", "shares": "shares"}.get(self.kind, "USD")


def _gaap(*names: str) -> tuple[str, ...]:
    return tuple(f"us-gaap:{name}" for name in names)


METRICS: tuple[Metric, ...] = (
    Metric(
        "revenue",
        "Total revenue",
        _gaap("RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet"),
        (
            "revenue",
            "revenues",
            "net sales",
            "sales",
            "net revenue",
            "net revenues",
            "top line",
            "turnover",
        ),
    ),
    Metric(
        "cost_of_revenue",
        "Total cost of revenue",
        _gaap("CostOfRevenue", "CostOfGoodsAndServicesSold"),
        ("cost of revenue", "cost of revenues", "cost of sales", "cost of goods sold", "cogs"),
    ),
    Metric(
        "gross_profit",
        "Gross profit",
        _gaap("GrossProfit"),
        ("gross profit", "gross margin"),
    ),
    Metric(
        "rd_expense",
        "Research and development expense",
        _gaap(
            "ResearchAndDevelopmentExpense",
            "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost",
        ),
        (
            "r&d",
            "r&d expense",
            "r&d expenses",
            "r&d spending",
            "r&d spend",
            "r and d",
            "research and development",
            "research and development expense",
            "research and development expenses",
            "research and development spending",
            "research & development",
        ),
    ),
    Metric(
        "sga",
        "Selling, general and administrative expense",
        _gaap("SellingGeneralAndAdministrativeExpense"),
        (
            "sg&a",
            "sg&a expense",
            "sg&a expenses",
            "selling general and administrative",
            "selling general and administrative expense",
            "selling general and administrative expenses",
        ),
    ),
    Metric(
        "operating_expenses",
        "Total operating expenses",
        _gaap("OperatingExpenses"),
        ("operating expenses", "opex"),
    ),
    Metric(
        "operating_income",
        "Operating income",
        _gaap("OperatingIncomeLoss"),
        (
            "operating income",
            "income from operations",
            "operating profit",
            "operating income loss",
            "operating loss",
        ),
    ),
    Metric(
        "pretax_income",
        "Income before income taxes",
        _gaap(
            "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
            "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
        ),
        (
            "pretax income",
            "pre tax income",
            "income before income taxes",
            "income before taxes",
            "income before provision for income taxes",
        ),
    ),
    Metric(
        "income_tax",
        "Provision for income taxes",
        _gaap("IncomeTaxExpenseBenefit"),
        (
            "income tax",
            "income taxes",
            "income tax expense",
            "provision for income taxes",
            "tax expense",
            "income tax provision",
        ),
    ),
    Metric(
        "net_income",
        "Net income",
        _gaap("NetIncomeLoss"),
        (
            "net income",
            "net earnings",
            "net profit",
            "profit",
            "earnings",
            "bottom line",
            "net income loss",
            "net loss",
        ),
    ),
    Metric(
        "eps_basic",
        "Basic earnings per share",
        _gaap("EarningsPerShareBasic"),
        ("basic eps", "basic earnings per share", "eps basic"),
        "per_share",
    ),
    Metric(
        "eps_diluted",
        "Diluted earnings per share",
        _gaap("EarningsPerShareDiluted"),
        ("eps", "diluted eps", "earnings per share", "diluted earnings per share", "eps diluted"),
        "per_share",
    ),
    Metric(
        "shares_diluted",
        "Weighted-average diluted shares",
        _gaap("WeightedAverageNumberOfDilutedSharesOutstanding"),
        (
            "diluted shares",
            "diluted share count",
            "weighted average diluted shares",
            "diluted shares outstanding",
        ),
        "shares",
    ),
    Metric(
        "operating_cash_flow",
        "Cash generated by operating activities",
        _gaap("NetCashProvidedByUsedInOperatingActivities"),
        (
            "operating cash flow",
            "operating cash flows",
            "cash from operations",
            "cash flow from operations",
            "cash flows from operations",
            "cash generated by operating activities",
            "cash provided by operating activities",
            "net cash provided by operating activities",
            "cash flow from operating activities",
        ),
    ),
    Metric(
        "capex",
        "Capital expenditures",
        _gaap("PaymentsToAcquirePropertyPlantAndEquipment"),
        (
            "capex",
            "capital expenditures",
            "capital expenditure",
            "capital spending",
            "purchases of property and equipment",
            "payments for acquisition of property plant and equipment",
        ),
    ),
    Metric(
        "investing_cash_flow",
        "Cash used in investing activities",
        _gaap("NetCashProvidedByUsedInInvestingActivities"),
        (
            "investing cash flow",
            "cash from investing activities",
            "cash used in investing activities",
            "net cash used in investing activities",
        ),
    ),
    Metric(
        "financing_cash_flow",
        "Cash used in financing activities",
        _gaap("NetCashProvidedByUsedInFinancingActivities"),
        (
            "financing cash flow",
            "cash from financing activities",
            "cash used in financing activities",
            "net cash used in financing activities",
        ),
    ),
    Metric(
        "dividends_paid",
        "Dividends paid",
        _gaap("PaymentsOfDividends", "PaymentsOfDividendsCommonStock"),
        ("dividends paid", "dividends", "dividend payments", "cash dividends"),
    ),
    Metric(
        "share_repurchases",
        "Repurchases of common stock",
        _gaap("PaymentsForRepurchaseOfCommonStock"),
        (
            "share repurchases",
            "stock repurchases",
            "buybacks",
            "share buybacks",
            "stock buybacks",
            "repurchases of common stock",
        ),
    ),
    Metric(
        "cash",
        "Cash and cash equivalents",
        _gaap("CashAndCashEquivalentsAtCarryingValue"),
        ("cash", "cash and cash equivalents", "cash and equivalents"),
        "instant",
    ),
    Metric(
        "inventory",
        "Inventories",
        _gaap("InventoryNet"),
        ("inventory", "inventories"),
        "instant",
    ),
    Metric(
        "total_assets",
        "Total assets",
        _gaap("Assets"),
        ("assets", "total assets"),
        "instant",
    ),
    Metric(
        "total_liabilities",
        "Total liabilities",
        _gaap("Liabilities"),
        ("liabilities", "total liabilities"),
        "instant",
    ),
    Metric(
        "stockholders_equity",
        "Total shareholders' equity",
        _gaap("StockholdersEquity"),
        (
            "shareholders equity",
            "stockholders equity",
            "shareholder equity",
            "stockholder equity",
            "equity",
            "book value",
        ),
        "instant",
    ),
    Metric(
        "long_term_debt",
        "Long-term debt",
        _gaap("LongTermDebtNoncurrent", "LongTermDebt"),
        ("long term debt", "debt"),
        "instant",
    ),
)

_BY_KEY = {metric.key: metric for metric in METRICS}
_NON_WORD = re.compile(r"[^a-z0-9&]+")
_OWNER = re.compile(r"^\s*\S+(?:\s+\S+)?'s\s+")
_LEADING = {"total", "the", "annual", "reported", "company", "companys", "fiscal", "year"}
_GROWTH_WORDS = {
    "growth",
    "growth rate",
    "change",
    "changes",
    "trend",
    "cagr",
    "increase",
    "decrease",
    "decline",
    "yoy",
    "year over year",
    "year over year growth",
}
_MARGIN_PREFIX = {
    "operating": "operating_income",
    "net": "net_income",
    "net profit": "net_income",
    "profit": "net_income",
    "gross": "gross_profit",
    "pretax": "pretax_income",
}
# Words a purely numeric question may contain besides metrics, companies and years.
_FILLER = frozenset(
    """a about across all amount an and annual annualized are as at average between by
    cagr change changed changes companies company compare compared comparison compound
    compounded corp corporation decline declined decrease decreased did do does during
    each end ended fiscal for from fy give grew grow growth had has have how in inc
    increase increased is it its last latest me much of on over percent percentage period
    previous prior rate recent reported show spend spent tell the their them they to total
    value versus vs was were what whats when which year years yoy""".split()
)
_GROWTH_CUES = frozenset(
    """cagr change changed changes compound compounded decline declined decrease decreased
    grew grow growth increase increased yoy""".split()
)
_YEAR_TOKEN = re.compile(r"^(?:fy)?\d{2,4}$")
_MARGIN_PHRASE = re.compile(
    rf" ({'|'.join(sorted(_MARGIN_PREFIX, key=len, reverse=True))})(?: profit)? margins? "
)


def normalize_topic(text: str) -> str:
    """Lowercase, drop a leading owner ("Apple's"), possessives and punctuation (keeping
    "&"), and leading filler words."""
    value = _OWNER.sub("", text.casefold().replace("’", "'")).replace("'s", "")
    words = _NON_WORD.sub(" ", value).split()
    while words and words[0] in _LEADING:
        words = words[1:]
    return " ".join(words)


_SYNONYMS = {normalize_topic(synonym): metric for metric in METRICS for synonym in metric.synonyms}


@dataclass
class MetricMatch:
    metrics: list[Metric] = field(default_factory=list)
    unmatched: list[str] = field(default_factory=list)
    growth: bool = False
    margin: bool = False


def _match_one(topic: str) -> tuple[Metric | None, bool, bool]:
    """(metric, growth asked, margin asked) for one normalised topic."""
    if metric := _SYNONYMS.get(topic):
        return metric, False, False
    for word in sorted(_GROWTH_WORDS, key=len, reverse=True):
        if topic.endswith(f" {word}"):
            base = topic[: -len(word) - 1]
            if metric := _SYNONYMS.get(base):
                return metric, True, False
        if topic.startswith(f"{word} in ") or topic.startswith(f"{word} of "):
            base = topic.split(" ", 2)[2]
            if metric := _SYNONYMS.get(base):
                return metric, True, False
    for suffix in (" margin", " margins", " margin percentage", " margin %", " profit margin"):
        if topic.endswith(suffix):
            prefix = topic[: -len(suffix)].strip()
            if prefix in _MARGIN_PREFIX:
                return _BY_KEY[_MARGIN_PREFIX[prefix]], False, True
    if topic in {"margin", "margins", "profit margin", "profit margins"}:
        return _BY_KEY["net_income"], False, True
    return None, False, False


def _words(text: str) -> list[str]:
    return _NON_WORD.sub(" ", text.casefold().replace("’", "'").replace("'s", "")).split()


def match_metrics(topics: list[str], ignore: frozenset[str] = frozenset()) -> MetricMatch:
    """Metrics named by the analysis topics; any topic that is not a metric is unmatched.

    ``ignore`` holds company-name words ("tesla revenue" matches revenue). A topic made
    only of filler/derived-value words ("compound annual growth rate") asks for growth.
    """
    match = MetricMatch()
    for topic in topics:
        normalized = " ".join(word for word in normalize_topic(topic).split() if word not in ignore)
        if not normalized:
            continue
        if all(word in _FILLER for word in normalized.split()):
            match.growth |= any(word in _GROWTH_CUES for word in normalized.split())
            continue
        metric, growth, margin = _match_one(normalized)
        if metric is None:
            match.unmatched.append(topic)
            continue
        if metric not in match.metrics:
            match.metrics.append(metric)
        match.growth |= growth
        match.margin |= margin
    return match


def scan_question(question: str, ignore: frozenset[str] = frozenset()) -> MetricMatch:
    """Metrics named in the question text, for analyses that returned no topics.

    Matches only when metric phrases, company words (``ignore``), years and filler words
    account for every word; any other word ("iPhone", "risk factors") means the question
    is not purely numeric and nothing matches, so the agent behaves as without facts.
    """
    text = f" {' '.join(_words(question))} "
    match = MetricMatch()
    for found in _MARGIN_PHRASE.finditer(text):
        if (metric := _BY_KEY[_MARGIN_PREFIX[found.group(1)]]) not in match.metrics:
            match.metrics.append(metric)
        match.margin = True
    text = _MARGIN_PHRASE.sub(" ", text)
    for synonym in sorted(_SYNONYMS, key=len, reverse=True):
        if f" {synonym} " in text:
            if (metric := _SYNONYMS[synonym]) not in match.metrics:
                match.metrics.append(metric)
            text = text.replace(f" {synonym} ", " ")
    residual = [
        word
        for word in text.split()
        if word not in _FILLER and word not in ignore and not _YEAR_TOKEN.match(word)
    ]
    if residual or not match.metrics:
        return MetricMatch(unmatched=[question])
    match.growth = any(word in _GROWTH_CUES for word in _words(question))
    return match


def company_words(*names: str) -> frozenset[str]:
    """Words of company names and tickers, ignored when matching metrics."""
    return frozenset(word for name in names for word in _words(name))


def metric(key: str) -> Metric:
    return _BY_KEY[key]
