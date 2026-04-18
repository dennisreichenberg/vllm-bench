"""Async benchmark engine for vLLM OpenAI-compatible endpoints."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable

import httpx

from .metrics import RequestResult

_DEFAULT_PROMPT = (
    "Explain the concept of transformer attention mechanisms in detail, "
    "covering self-attention, multi-head attention, and their computational complexity."
)


async def _run_streaming_request(
    client: httpx.AsyncClient,
    url: str,
    payload: dict,
) -> RequestResult:
    start = time.perf_counter()
    first_token_time: float | None = None
    tokens_generated = 0
    inter_token_latencies: list[float] = []
    last_token_time = start

    try:
        async with client.stream("POST", url, json=payload, timeout=120.0) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                now = time.perf_counter()
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content", "")

                if content:
                    if first_token_time is None:
                        first_token_time = now
                        inter_token_latencies.append(now - start)
                    else:
                        inter_token_latencies.append(now - last_token_time)
                    last_token_time = now

                # Count tokens from usage if provided at end
                usage = chunk.get("usage")
                if usage:
                    tokens_generated = usage.get("completion_tokens", tokens_generated)
                elif content:
                    tokens_generated += 1  # rough estimate: 1 chunk ≈ 1 token

        total_s = time.perf_counter() - start
        return RequestResult(
            success=True,
            ttft_s=first_token_time - start if first_token_time else None,
            total_s=total_s,
            tokens_generated=tokens_generated,
            inter_token_latencies=inter_token_latencies,
        )
    except Exception as exc:
        return RequestResult(success=False, error=str(exc))


async def _run_non_streaming_request(
    client: httpx.AsyncClient,
    url: str,
    payload: dict,
) -> RequestResult:
    start = time.perf_counter()
    try:
        response = await client.post(url, json=payload, timeout=120.0)
        response.raise_for_status()
        total_s = time.perf_counter() - start
        data = response.json()
        usage = data.get("usage", {})
        tokens_generated = usage.get("completion_tokens", 0)
        return RequestResult(
            success=True,
            ttft_s=total_s,  # no streaming, so TTFT = total time
            total_s=total_s,
            tokens_generated=tokens_generated,
            inter_token_latencies=[],
        )
    except Exception as exc:
        return RequestResult(success=False, error=str(exc))


async def run_benchmark(
    base_url: str,
    model: str,
    num_requests: int,
    concurrency: int,
    prompt: str,
    max_tokens: int,
    streaming: bool,
    api_key: str | None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[list[RequestResult], float]:
    endpoint = base_url.rstrip("/") + "/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": streaming,
    }
    if streaming:
        payload["stream_options"] = {"include_usage": True}

    semaphore = asyncio.Semaphore(concurrency)
    completed = 0
    results: list[RequestResult] = []
    lock = asyncio.Lock()

    async def run_one(client: httpx.AsyncClient) -> RequestResult:
        nonlocal completed
        async with semaphore:
            if streaming:
                result = await _run_streaming_request(client, endpoint, payload)
            else:
                result = await _run_non_streaming_request(client, endpoint, payload)
            async with lock:
                completed += 1
                if progress_callback:
                    progress_callback(completed, num_requests)
            return result

    wall_start = time.perf_counter()
    async with httpx.AsyncClient(headers=headers) as client:
        tasks = [run_one(client) for _ in range(num_requests)]
        results = list(await asyncio.gather(*tasks))
    wall_total = time.perf_counter() - wall_start

    return results, wall_total
