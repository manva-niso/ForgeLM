"""Concurrent load test against a running ForgeLM API.

Usage:
    uv run python scripts/load_test.py --url http://127.0.0.1:8000 --workers 20 --requests 10
"""

from __future__ import annotations

import argparse
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

PROMPT = "### Instruction: Write a short story about a little cat.\n### Response:"


def one_request(client: httpx.Client, url: str) -> tuple[float, int]:
    start = time.monotonic()
    r = client.post(f"{url}/v1/completions", json={"prompt": PROMPT, "max_tokens": 32})
    return time.monotonic() - start, r.status_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="load-test")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--requests", type=int, default=10)
    args = parser.parse_args(argv)

    with httpx.Client(timeout=60) as client:
        r = client.get(f"{args.url}/healthz")
        if r.status_code != 200:
            print(f"server not reachable at {args.url}: {r.status_code}")
            return 1
        total = args.workers * args.requests
        start = time.monotonic()
        latencies: list[float] = []
        statuses: dict[int, int] = {}
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(one_request, client, args.url) for _ in range(total)]
            for f in futures:
                latency, status = f.result()
                latencies.append(latency)
                statuses[status] = statuses.get(status, 0) + 1
        elapsed = time.monotonic() - start

    latencies.sort()
    p50 = statistics.median(latencies)
    p95 = latencies[int(len(latencies) * 0.95) - 1]
    p99 = latencies[int(len(latencies) * 0.99) - 1]
    rps = total / elapsed
    md = [
        "# Benchmark: API Load Test - Day 11 (2026-08-31)",
        "",
        "## Setup",
        f"- URL: {args.url}",
        f"- Workers: {args.workers}, requests/worker: {args.requests} (total {total})",
        "- Prompt: story instruction, max_tokens 32",
        f"- Wall time: {elapsed:.1f}s",
        "",
        "## Results",
        "| metric | value |",
        "|---|---|",
        f"| requests/sec | {rps:.1f} |",
        f"| p50 latency | {p50 * 1000:.0f} ms |",
        f"| p95 latency | {p95 * 1000:.0f} ms |",
        f"| p99 latency | {p99 * 1000:.0f} ms |",
        f"| status codes | {statuses} |",
        "",
        "## Notes",
        "- Local uvicorn (single worker, CPU); rate limit 10/min would block this",
        "  many requests - load test ran with FORGE_LM_RATE_LIMIT raised.",
        "- Latency dominated by generation (32 tokens) + engine path.",
        "",
    ]
    Path("benchmarks/api_loadtest.md").write_text("\n".join(md), encoding="utf-8")
    print(f"rps={rps:.1f} p50={p50*1000:.0f}ms p95={p95*1000:.0f}ms p99={p99*1000:.0f}ms statuses={statuses}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())