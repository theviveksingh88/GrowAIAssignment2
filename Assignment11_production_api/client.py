"""
Assignment 11 - API Client
==========================

Demonstrates the deployed API: health, streaming output, and the latency
difference between a cache miss and a cache hit.

Run (with the stack up):
    python client.py
"""

import json
import time

import httpx

API_URL = "http://localhost:8080"

QUESTION = "What is deep learning and where is it used?"


def show_health():

    print("=" * 70)
    print("HEALTH")
    print("=" * 70)

    response = httpx.get(f"{API_URL}/health", timeout=30)

    print(json.dumps(response.json(), indent=2))


def stream_answer(question, use_cache=True):
    """Consume the SSE stream, printing tokens as they arrive."""

    start = time.perf_counter()
    first_token_at = None
    collected = []

    with httpx.stream(
        "POST",
        f"{API_URL}/chat",
        json={"query": question, "stream": True, "use_cache": use_cache},
        timeout=600,
    ) as response:

        response.raise_for_status()

        for line in response.iter_lines():

            if not line.startswith("data: "):
                continue

            payload = json.loads(line[len("data: "):])

            if payload.get("done"):
                break

            if "error" in payload:
                print("\n[error]", payload["error"])
                break

            token = payload.get("token", "")

            if token and first_token_at is None:
                first_token_at = time.perf_counter() - start

            collected.append(token)
            print(token, end="", flush=True)

    total = time.perf_counter() - start

    print()

    return {
        "answer": "".join(collected),
        "time_to_first_token_s": round(first_token_at or 0, 3),
        "total_s": round(total, 3),
    }


def main():

    show_health()

    print("\n" + "=" * 70)
    print("STREAMING (cache miss)")
    print("=" * 70)

    first = stream_answer(QUESTION)

    print("\n" + "=" * 70)
    print("STREAMING (cache hit - same question)")
    print("=" * 70)

    second = stream_answer(QUESTION)

    print("\n" + "=" * 70)
    print("CACHE EFFECT")
    print("=" * 70)
    print(f"  miss: {first['total_s']:>7.3f}s "
          f"(first token {first['time_to_first_token_s']}s)")
    print(f"  hit : {second['total_s']:>7.3f}s")

    if second["total_s"] > 0:
        print(f"  speedup: {first['total_s'] / max(second['total_s'], 1e-6):.0f}x")

    print("\n" + "=" * 70)
    print("METRICS")
    print("=" * 70)

    print(json.dumps(httpx.get(f"{API_URL}/metrics", timeout=30).json(), indent=2))


if __name__ == "__main__":
    main()
