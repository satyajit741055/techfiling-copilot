"""Latency, cost and reliability benchmark for the retrieve-then-generate pipeline.

Modes:
  direct   run the pipeline in-process (component timings, retriever cold start)
  api      call a running backend over HTTP (client E2E incl. auth/network, server timings via
           the Server-Timing header)
  analyze  summarize production logs (logs/metrics/queries-*.jsonl) with the same math

Latency percentiles (P50/P95/P99) are reported for warm requests; cold requests (first
--warmup requests, or any request that loaded the retriever) are reported separately.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import sys
import time
from datetime import datetime, timezone
from itertools import cycle, islice
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from config.settings import settings
from observability import QueryRecord, QueryTracker, compute_cost, load_records, summarize
from observability.records import is_timeout_error, utc_now_iso

DEFAULT_GOLDEN_PATH = ROOT / "goldens" / "golden_retriever_dataset_v3.jsonl"
DEFAULT_RESULTS_DIR = ROOT / "eval" / "performance" / "results"


def load_cases(path: Path, require_doc: bool) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        cases = [json.loads(line) for line in handle if line.strip()]
    cases = [c for c in cases if c.get("query") and (c.get("doc_id") or not require_doc)]
    if not cases:
        raise ValueError(f"No usable cases in {path}")
    return cases


def select_queries(cases: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.categories:
        cases = [c for c in cases if c.get("category") in set(args.categories)]
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        raise ValueError("No golden queries matched the filters")
    return list(islice(cycle(cases), args.requests or len(cases) * args.repeat))


def parse_server_timing(header: str | None) -> dict[str, float]:
    timings = {}
    for part in (header or "").split(","):
        match = re.match(r"\s*(\w+);dur=([\d.]+)", part)
        if match:
            timings[match.group(1)] = float(match.group(2)) / 1000
    return timings


def run_direct(queries: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    from generation import AnswerGenerator, GenerationConfig

    overrides = {k: v for k, v in (("provider", args.provider), ("model", args.model)) if v}
    config = GenerationConfig(**overrides)
    generator = AnswerGenerator(config=config, root=ROOT)
    tracker = QueryTracker(writer=None, cold_requests=args.warmup)
    records = []
    for index, case in enumerate(queries, start=1):
        try:
            _, record = tracker.run(generator, case["query"], doc_id=case.get("doc_id"))
        except Exception as error:
            record = getattr(error, "query_record", None) or QueryRecord(
                timestamp=utc_now_iso(),
                request_id=None,
                doc_id=case.get("doc_id"),
                status="timeout" if is_timeout_error(error) else "error",
                error_type=type(error).__name__,
            )
        row = record.to_dict()
        row["case_id"] = case.get("id")
        row["client_e2e_seconds"] = record.total_seconds
        records.append(row)
        report(index, len(queries), row)
    return records


def run_api(queries: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    import httpx

    from api.config import api_settings

    api_key = args.api_key or api_settings.API_ACCESS_KEY
    headers = {"X-API-Key": api_key} if api_key else {}
    if args.bearer_token:
        headers["Authorization"] = f"Bearer {args.bearer_token}"
    records = []
    with httpx.Client(base_url=args.api_url, headers=headers, timeout=args.timeout) as client:
        for index, case in enumerate(queries, start=1):
            row: dict[str, Any] = {
                "timestamp": utc_now_iso(),
                "request_id": None,
                "doc_id": case["doc_id"],
                "case_id": case.get("id"),
                "status": "success",
            }
            start = time.perf_counter()
            try:
                response = client.post(
                    "/questions", json={"question": case["query"], "doc_id": case["doc_id"]}
                )
                row["client_e2e_seconds"] = time.perf_counter() - start
                if response.status_code == 200:
                    body = response.json()
                    timings = parse_server_timing(response.headers.get("Server-Timing"))
                    attempts = int(response.headers.get("X-LLM-Attempts", 1))
                    cost = compute_cost(
                        provider=settings.effective_llm_provider,
                        model=body["model"],
                        prompt_tokens=body["prompt_tokens"],
                        completion_tokens=body["completion_tokens"],
                        local_compute_seconds=timings.get("total", 0.0)
                        if settings.effective_llm_provider == "ollama"
                        else 0.0,
                        input_price_per_1m=settings.LLM_INPUT_PRICE_PER_1M,
                        output_price_per_1m=settings.LLM_OUTPUT_PRICE_PER_1M,
                        compute_cost_per_hour=settings.LOCAL_COMPUTE_COST_PER_HOUR,
                    )
                    row.update(
                        retrieval_seconds=timings.get("retrieval", 0.0),
                        generation_seconds=timings.get("generation", 0.0),
                        llm_seconds=body["latency_seconds"],
                        total_seconds=timings.get("total", row["client_e2e_seconds"]),
                        is_cold=response.headers.get("X-Cold-Start") == "true",
                        llm_attempts=attempts,
                        retries=max(attempts - 1, 0),
                        model=body["model"],
                        prompt_tokens=body["prompt_tokens"],
                        completion_tokens=body["completion_tokens"],
                        input_cost_usd=cost.input_cost_usd,
                        output_cost_usd=cost.output_cost_usd,
                        infra_cost_usd=cost.infra_cost_usd,
                        total_cost_usd=cost.total_cost_usd,
                        validation_passed=body["answer"].get("is_valid"),
                        insufficient_context=body["answer"].get("insufficient_context"),
                    )
                else:
                    row.update(
                        status="error", error_type=f"HTTP{response.status_code}", error_stage="api"
                    )
            except httpx.TimeoutException as error:
                row.update(
                    status="timeout",
                    error_type=type(error).__name__,
                    client_e2e_seconds=time.perf_counter() - start,
                )
            except httpx.HTTPError as error:
                row.update(
                    status="error",
                    error_type=type(error).__name__,
                    client_e2e_seconds=time.perf_counter() - start,
                )
            records.append(row)
            report(index, len(queries), row)
            if args.delay:
                time.sleep(args.delay)
    return records


def report(index: int, total: int, row: dict[str, Any]) -> None:
    print(
        f"[{index}/{total}] {row.get('case_id')} {row['status']} "
        f"e2e={row.get('client_e2e_seconds', 0):.2f}s "
        f"retr={row.get('retrieval_seconds', 0):.2f}s gen={row.get('generation_seconds', 0):.2f}s"
        + (" cold" if row.get("is_cold") else "")
        + (f" {row['error_type']}" if row.get("error_type") else "")
    )


def save(
    records: list[dict[str, Any]], summary: dict[str, Any], manifest: dict[str, Any], out_dir: Path
) -> Path:
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = out_dir / f"{stamp}_{manifest['mode']}"
    run_dir.mkdir(parents=True, exist_ok=False)
    with (run_dir / "queries.jsonl").open("w", encoding="utf-8") as handle:
        handle.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
    summary = {"run_id": run_dir.name, **summary}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return run_dir


def print_summary(summary: dict[str, Any]) -> None:
    def fmt(value: Any) -> str:
        return "-" if value is None else f"{value:.3f}"

    for kind in ("warm", "cold"):
        section = summary["latency"][kind]
        print(f"\nLatency ({kind}, {section['requests']} requests, seconds)")
        print(f"  {'component':<20}{'p50':>9}{'p95':>9}{'p99':>9}{'mean':>9}{'max':>9}")
        for field, dist in section.items():
            if isinstance(dist, dict) and "p50" in dist and dist["count"]:
                print(
                    f"  {field:<20}{fmt(dist['p50']):>9}{fmt(dist['p95']):>9}"
                    f"{fmt(dist['p99']):>9}{fmt(dist['mean']):>9}{fmt(dist['max']):>9}"
                )
    cost, rel = summary["cost"], summary["reliability"]
    print(
        f"\nCost per query: ${cost['cost_per_query_usd'] or 0:.6f}  total ${cost['total_cost_usd']:.6f}"
    )
    print(
        f"Reliability: success={rel['success_rate']} error={rel['error_rate']} "
        f"timeout={rel['timeout_rate']} retry={rel['retry_rate']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("mode", choices=["direct", "api", "analyze"])
    parser.add_argument("--golden-path", type=Path, default=DEFAULT_GOLDEN_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--categories", nargs="+")
    parser.add_argument("--limit", type=int, help="Use only the first N golden queries")
    parser.add_argument("--repeat", type=int, default=3, help="Passes over the selected queries")
    parser.add_argument("--requests", type=int, help="Total requests (overrides --repeat)")
    parser.add_argument("--warmup", type=int, default=2, help="First N requests reported as cold")
    parser.add_argument("--provider", help="direct mode: generator provider")
    parser.add_argument("--model", help="direct mode: generator model")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--api-key", help="X-API-Key for api mode (default: API_ACCESS_KEY from .env)"
    )
    parser.add_argument("--bearer-token", help="api mode: Supabase JWT, if the backend enforces it")
    parser.add_argument("--timeout", type=float, default=120.0, help="api mode: client timeout (s)")
    parser.add_argument("--delay", type=float, default=0.0, help="api mode: pause between requests")
    parser.add_argument(
        "--logs", type=Path, default=ROOT / settings.METRICS_DIR, help="analyze mode: file or dir"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.mode == "analyze":
        try:
            records = load_records(args.logs)
        except FileNotFoundError as error:
            raise SystemExit(str(error))
        summary = summarize(records, warmup=0)
        print_summary(summary)
        print(json.dumps(summary["reliability"], indent=2))
        return

    queries = select_queries(load_cases(args.golden_path, args.mode == "api"), args)
    manifest = {
        "mode": args.mode,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "golden_path": str(args.golden_path),
        "requests": len(queries),
        "warmup": args.warmup,
        "execution_modes": {
            "embedding": settings.effective_embedding_mode,
            "reranker": settings.effective_reranker_mode,
            "generation": settings.effective_generation_mode,
        },
        "llm_provider": args.provider or settings.effective_llm_provider,
        "vector_store": settings.VECTOR_STORE,
        "llm_max_retries": settings.LLM_MAX_RETRIES,
        "local_compute_cost_per_hour": settings.LOCAL_COMPUTE_COST_PER_HOUR,
        "api_url": args.api_url if args.mode == "api" else None,
        "timeout_seconds": args.timeout if args.mode == "api" else None,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "command": sys.argv,
    }
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return

    records = (run_direct if args.mode == "direct" else run_api)(queries, args)
    summary = summarize(records, warmup=args.warmup)
    run_dir = save(records, summary, manifest, args.output_dir)
    print_summary(summary)
    print(f"\nResults saved to: {run_dir}")


if __name__ == "__main__":
    main()
