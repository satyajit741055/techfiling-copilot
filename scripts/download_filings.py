"""Download the latest six SEC 10-K filings for the configured company corpus."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib import error, request

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "data" / "raw_htm"
COMPANY_CIKS = {
    "AAPL": "0000320193",
    "MSFT": "0000789019",
    "NVDA": "0001045810",
    "AMZN": "0001018724",
    "GOOGL": "0001652044",
    "TSLA": "0001318605",
}
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions"
SEC_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data"
REQUEST_DELAY_SECONDS = 0.2
MAX_RETRIES = 3


def fetch(url: str, user_agent: str, accept: str) -> bytes:
    headers = {"Accept": accept, "User-Agent": user_agent}
    for attempt in range(MAX_RETRIES):
        try:
            with request.urlopen(request.Request(url, headers=headers), timeout=60) as response:
                return response.read()
        except error.HTTPError as exc:
            if exc.code not in {429, 500, 502, 503, 504} or attempt == MAX_RETRIES - 1:
                raise
        except error.URLError:
            if attempt == MAX_RETRIES - 1:
                raise
        time.sleep(2**attempt)
    raise RuntimeError(f"Unable to download {url}")


def get_json(url: str, user_agent: str) -> dict[str, Any]:
    return json.loads(fetch(url, user_agent, "application/json").decode("utf-8"))


def extract_10k_filings(submission: dict[str, Any]) -> list[dict[str, str]]:
    recent = submission["filings"]["recent"] if "filings" in submission else submission
    filings = []
    for form, accession, document, filing_date, report_date in zip(
        recent["form"],
        recent["accessionNumber"],
        recent["primaryDocument"],
        recent["filingDate"],
        recent["reportDate"],
        strict=True,
    ):
        if form == "10-K":
            filings.append(
                {
                    "fiscal_year": (report_date or filing_date)[:4],
                    "form": form,
                    "accession_number": accession,
                    "primary_document": document,
                    "filing_date": filing_date,
                    "report_date": report_date,
                }
            )
    return filings


def list_filings(cik: str, count: int, user_agent: str) -> list[dict[str, str]]:
    submission = get_json(f"{SEC_SUBMISSIONS_URL}/CIK{cik}.json", user_agent)
    filings = extract_10k_filings(submission)
    for historical_file in submission.get("filings", {}).get("files", []):
        if len(filings) >= count:
            break
        filings.extend(
            extract_10k_filings(
                get_json(f"{SEC_SUBMISSIONS_URL}/{historical_file['name']}", user_agent)
            )
        )
        time.sleep(REQUEST_DELAY_SECONDS)

    unique = {filing["accession_number"]: filing for filing in filings}
    return sorted(
        unique.values(),
        key=lambda filing: (filing["report_date"], filing["filing_date"]),
        reverse=True,
    )[:count]


def download_filings(
    output_dir: Path, filings_per_company: int, user_agent: str
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "source": "SEC EDGAR",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "form": "10-K",
        "filings_per_company": filings_per_company,
        "downloaded_count": 0,
        "skipped_count": 0,
        "filings": [],
    }

    for ticker, cik in COMPANY_CIKS.items():
        print(f"Finding {ticker} 10-K filings...")
        filings = list_filings(cik, filings_per_company, user_agent)
        if len(filings) < filings_per_company:
            raise RuntimeError(
                f"SEC returned only {len(filings)} 10-K filings for {ticker}; "
                f"expected {filings_per_company}"
            )

        for filing in filings:
            accession_path = filing["accession_number"].replace("-", "")
            source_url = (
                f"{SEC_ARCHIVES_URL}/{int(cik)}/{accession_path}/"
                f"{filing['primary_document']}"
            )
            suffix = Path(filing["primary_document"]).suffix or ".html"
            local_path = output_dir / f"{ticker}_10-K_{filing['fiscal_year']}{suffix}"
            status = "skipped"
            if not local_path.exists():
                local_path.write_bytes(
                    fetch(
                        source_url,
                        user_agent,
                        "text/html,application/xhtml+xml,application/xml",
                    )
                )
                manifest["downloaded_count"] += 1
                status = "downloaded"
                time.sleep(REQUEST_DELAY_SECONDS)
            else:
                manifest["skipped_count"] += 1

            manifest["filings"].append(
                {
                    "ticker": ticker,
                    "cik": cik,
                    **filing,
                    "source_url": source_url,
                    "local_path": str(local_path.relative_to(output_dir)),
                    "status": status,
                }
            )
            print(f"  {filing['fiscal_year']}: {status} {local_path.name}")

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download six years of 10-K filings for the agentic RAG corpus."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--filings-per-company", type=int, default=6)
    parser.add_argument(
        "--user-agent",
        default=os.getenv("SEC_USER_AGENT"),
        help="SEC identity, e.g. 'TechFiling Copilot name@example.com'; or set SEC_USER_AGENT.",
    )
    args = parser.parse_args()
    if args.filings_per_company < 1:
        parser.error("--filings-per-company must be at least 1")
    if not args.user_agent:
        parser.error("set SEC_USER_AGENT or pass --user-agent with your name and email")
    return args


def main() -> None:
    args = parse_args()
    manifest = download_filings(args.output_dir, args.filings_per_company, args.user_agent)
    print(
        f"Finished: {manifest['downloaded_count']} downloaded, "
        f"{manifest['skipped_count']} already present in {args.output_dir}"
    )
    print(f"Manifest: {args.output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
