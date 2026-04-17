"""Result dataclasses and aggregation helpers."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field


@dataclass
class RequestResult:
    success: bool
    ttft_s: float | None = None          # time to first token
    total_s: float | None = None         # wall-clock time for full response
    tokens_generated: int | None = None
    inter_token_latencies: list[float] = field(default_factory=list)
    error: str | None = None

    @property
    def tokens_per_second(self) -> float | None:
        if self.tokens_generated and self.total_s and self.total_s > 0:
            return self.tokens_generated / self.total_s
        return None


@dataclass
class AggregatedMetrics:
    total_requests: int
    successful: int
    failed: int

    # TTFT percentiles (seconds)
    ttft_mean: float
    ttft_p50: float
    ttft_p90: float
    ttft_p99: float

    # Throughput
    tokens_per_second_mean: float
    tokens_per_second_p50: float

    # Inter-token latency (ms)
    itl_mean_ms: float
    itl_p90_ms: float

    # Overall
    total_wall_s: float
    request_throughput: float  # requests/sec

    @property
    def success_rate(self) -> float:
        return self.successful / self.total_requests if self.total_requests else 0.0


def aggregate(results: list[RequestResult], total_wall_s: float) -> AggregatedMetrics:
    successful = [r for r in results if r.success]
    failed = len(results) - len(successful)

    ttfts = [r.ttft_s for r in successful if r.ttft_s is not None]
    tps_values = [r.tokens_per_second for r in successful if r.tokens_per_second is not None]
    itls_ms = [
        latency * 1000
        for r in successful
        for latency in r.inter_token_latencies
    ]

    def _pct(data: list[float], p: float) -> float:
        if not data:
            return 0.0
        sorted_data = sorted(data)
        idx = int(len(sorted_data) * p / 100)
        return sorted_data[min(idx, len(sorted_data) - 1)]

    return AggregatedMetrics(
        total_requests=len(results),
        successful=len(successful),
        failed=failed,
        ttft_mean=statistics.mean(ttfts) if ttfts else 0.0,
        ttft_p50=_pct(ttfts, 50),
        ttft_p90=_pct(ttfts, 90),
        ttft_p99=_pct(ttfts, 99),
        tokens_per_second_mean=statistics.mean(tps_values) if tps_values else 0.0,
        tokens_per_second_p50=_pct(tps_values, 50),
        itl_mean_ms=statistics.mean(itls_ms) if itls_ms else 0.0,
        itl_p90_ms=_pct(itls_ms, 90),
        total_wall_s=total_wall_s,
        request_throughput=len(results) / total_wall_s if total_wall_s > 0 else 0.0,
    )
