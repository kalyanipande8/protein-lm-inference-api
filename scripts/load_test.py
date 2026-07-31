"""Fires concurrent /predict requests against a running instance of the API
and reports p50/p95/p99 latency, so the batching behavior can be verified
under load rather than just tested for correctness.

Usage:
    python scripts/load_test.py --url http://localhost:8000 --requests 500 --concurrency 50
"""

import argparse
import asyncio
import random
import statistics
import time

import httpx

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def random_sequence(min_len: int = 50, max_len: int = 300) -> str:
    length = random.randint(min_len, max_len)
    return "".join(random.choice(AMINO_ACIDS) for _ in range(length))


async def send_one(client: httpx.AsyncClient, url: str) -> float:
    start = time.perf_counter()
    resp = await client.post(f"{url}/predict", json={"sequence": random_sequence()})
    resp.raise_for_status()
    return (time.perf_counter() - start) * 1000  # ms


async def run(url: str, requests: int, concurrency: int) -> list[float]:
    latencies: list[float] = []
    semaphore = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(timeout=30) as client:
        async def bound_send():
            async with semaphore:
                latencies.append(await send_one(client, url))

        await asyncio.gather(*(bound_send() for _ in range(requests)))

    return latencies


def percentile(data: list[float], pct: float) -> float:
    data = sorted(data)
    idx = int(len(data) * pct)
    return data[min(idx, len(data) - 1)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--requests", type=int, default=500)
    parser.add_argument("--concurrency", type=int, default=50)
    args = parser.parse_args()

    latencies = asyncio.run(run(args.url, args.requests, args.concurrency))

    print(f"requests={len(latencies)} concurrency={args.concurrency}")
    print(f"p50={percentile(latencies, 0.50):.1f}ms")
    print(f"p95={percentile(latencies, 0.95):.1f}ms")
    print(f"p99={percentile(latencies, 0.99):.1f}ms")
    print(f"mean={statistics.mean(latencies):.1f}ms  max={max(latencies):.1f}ms")


if __name__ == "__main__":
    main()
