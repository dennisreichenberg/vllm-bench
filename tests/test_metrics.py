"""Unit tests for metrics aggregation."""

from __future__ import annotations

import pytest

from vllm_bench.metrics import AggregatedMetrics, RequestResult, aggregate


def test_aggregate_basic():
    results = [
        RequestResult(
            success=True, ttft_s=0.1, total_s=1.0,
            tokens_generated=50, inter_token_latencies=[0.02] * 10,
        ),
        RequestResult(
            success=True, ttft_s=0.2, total_s=2.0,
            tokens_generated=100, inter_token_latencies=[0.03] * 10,
        ),
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


def test_tokens_per_second_none_when_no_tokens():
    r = RequestResult(success=True, ttft_s=0.1, total_s=2.0, tokens_generated=None)
    assert r.tokens_per_second is None


def test_tokens_per_second_none_when_zero_total_s():
    r = RequestResult(success=True, ttft_s=0.1, total_s=0.0, tokens_generated=100)
    assert r.tokens_per_second is None


def test_tokens_per_second_none_when_zero_tokens():
    r = RequestResult(success=True, ttft_s=0.1, total_s=2.0, tokens_generated=0)
    assert r.tokens_per_second is None


def test_aggregate_empty_results():
    metrics = aggregate([], total_wall_s=1.0)
    assert metrics.total_requests == 0
    assert metrics.successful == 0
    assert metrics.failed == 0
    assert metrics.success_rate == 0.0
    assert metrics.ttft_mean == 0.0
    assert metrics.request_throughput == 0.0


def test_aggregate_zero_wall_time():
    results = [RequestResult(success=True, ttft_s=0.1, total_s=1.0, tokens_generated=10)]
    metrics = aggregate(results, total_wall_s=0.0)
    assert metrics.request_throughput == 0.0


def test_aggregate_all_successful():
    results = [
        RequestResult(success=True, ttft_s=0.1, total_s=1.0, tokens_generated=50),
        RequestResult(success=True, ttft_s=0.2, total_s=2.0, tokens_generated=100),
    ]
    metrics = aggregate(results, total_wall_s=2.0)
    assert metrics.successful == 2
    assert metrics.failed == 0
    assert metrics.success_rate == 1.0


def test_aggregate_ttft_percentiles_ordered():
    results = [
        RequestResult(success=True, ttft_s=v, total_s=v + 0.5, tokens_generated=10)
        for v in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ]
    metrics = aggregate(results, total_wall_s=5.0)
    assert metrics.ttft_p50 <= metrics.ttft_p90
    assert metrics.ttft_p90 <= metrics.ttft_p99


def test_aggregate_ttft_p50_value():
    results = [
        RequestResult(success=True, ttft_s=v, total_s=v + 0.5, tokens_generated=10)
        for v in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ]
    metrics = aggregate(results, total_wall_s=5.0)
    # p50 index = int(10 * 50 / 100) = 5, sorted[5] = 0.6
    assert metrics.ttft_p50 == pytest.approx(0.6, abs=0.01)


def test_aggregate_with_itl():
    results = [
        RequestResult(
            success=True,
            ttft_s=0.1,
            total_s=1.0,
            tokens_generated=5,
            inter_token_latencies=[0.01, 0.02, 0.03, 0.04, 0.05],
        )
    ]
    metrics = aggregate(results, total_wall_s=1.0)
    # Mean of [10, 20, 30, 40, 50] ms = 30 ms
    assert metrics.itl_mean_ms == pytest.approx(30.0, abs=0.1)


def test_aggregate_no_ttft_on_successful():
    # Successful results without TTFT field set
    results = [RequestResult(success=True, ttft_s=None, total_s=1.0, tokens_generated=50)]
    metrics = aggregate(results, total_wall_s=1.0)
    assert metrics.ttft_mean == 0.0
    assert metrics.ttft_p50 == 0.0


def test_aggregate_no_tokens_per_second():
    # Successful but no tokens/total_s to compute tps
    results = [RequestResult(success=True, ttft_s=0.1, total_s=None, tokens_generated=None)]
    metrics = aggregate(results, total_wall_s=1.0)
    assert metrics.tokens_per_second_mean == 0.0
    assert metrics.tokens_per_second_p50 == 0.0


def test_aggregate_itl_empty_when_no_streaming():
    results = [RequestResult(success=True, ttft_s=0.1, total_s=1.0, tokens_generated=50)]
    metrics = aggregate(results, total_wall_s=1.0)
    assert metrics.itl_mean_ms == 0.0
    assert metrics.itl_p90_ms == 0.0


def test_success_rate_zero_requests():
    metrics = AggregatedMetrics(
        total_requests=0,
        successful=0,
        failed=0,
        ttft_mean=0.0,
        ttft_p50=0.0,
        ttft_p90=0.0,
        ttft_p99=0.0,
        tokens_per_second_mean=0.0,
        tokens_per_second_p50=0.0,
        itl_mean_ms=0.0,
        itl_p90_ms=0.0,
        total_wall_s=1.0,
        request_throughput=0.0,
    )
    assert metrics.success_rate == 0.0


def test_success_rate_full():
    metrics = AggregatedMetrics(
        total_requests=5,
        successful=5,
        failed=0,
        ttft_mean=0.1,
        ttft_p50=0.1,
        ttft_p90=0.1,
        ttft_p99=0.1,
        tokens_per_second_mean=50.0,
        tokens_per_second_p50=50.0,
        itl_mean_ms=20.0,
        itl_p90_ms=30.0,
        total_wall_s=2.0,
        request_throughput=2.5,
    )
    assert metrics.success_rate == 1.0


def test_request_result_defaults():
    r = RequestResult(success=True)
    assert r.ttft_s is None
    assert r.total_s is None
    assert r.tokens_generated is None
    assert r.inter_token_latencies == []
    assert r.error is None


def test_aggregate_single_result():
    results = [RequestResult(success=True, ttft_s=0.5, total_s=1.0, tokens_generated=20)]
    metrics = aggregate(results, total_wall_s=1.0)
    assert metrics.total_requests == 1
    assert metrics.successful == 1
    assert metrics.ttft_mean == pytest.approx(0.5)
    assert metrics.ttft_p50 == pytest.approx(0.5)
    assert metrics.ttft_p90 == pytest.approx(0.5)
    assert metrics.ttft_p99 == pytest.approx(0.5)
