"""Unit tests for metrics aggregation."""

from __future__ import annotations

from vllm_bench.metrics import RequestResult, aggregate


def test_aggregate_basic():
    results = [
        RequestResult(success=True, ttft_s=0.1, total_s=1.0, tokens_generated=50, inter_token_latencies=[0.02] * 10),
        RequestResult(success=True, ttft_s=0.2, total_s=2.0, tokens_generated=100, inter_token_latencies=[0.03] * 10),
        RequestResult(success=False, error="connection refused"),
    ]
    metrics = aggregate(results, total_wall_s=2.5)

    assert metrics.total_requests == 3
    assert metrics.successful == 2
    assert metrics.failed == 1
    assert abs(metrics.success_rate - 2 / 3) < 0.001
    assert metrics.ttft_mean == pytest.approx(0.15, abs=0.001)
    assert metrics.request_throughput == pytest.approx(3 / 2.5, abs=0.001)


def test_aggregate_all_failed():
    results = [RequestResult(success=False, error="err") for _ in range(5)]
    metrics = aggregate(results, total_wall_s=1.0)
    assert metrics.successful == 0
    assert metrics.success_rate == 0.0
    assert metrics.ttft_mean == 0.0


def test_tokens_per_second():
    r = RequestResult(success=True, ttft_s=0.1, total_s=2.0, tokens_generated=100)
    assert r.tokens_per_second == pytest.approx(50.0)


import pytest
