"""Unit tests for the async benchmark engine."""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vllm_bench.benchmark import (
    _DEFAULT_PROMPT,
    _run_non_streaming_request,
    _run_streaming_request,
    run_benchmark,
)
from vllm_bench.metrics import RequestResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_streaming_client(lines: list[str]) -> MagicMock:
    """Build a mock httpx client whose .stream() yields the given SSE lines."""

    async def aiter_lines():
        for line in lines:
            yield line

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.aiter_lines = aiter_lines

    @asynccontextmanager
    async def fake_stream(*args, **kwargs):
        yield mock_response

    mock_client = MagicMock()
    mock_client.stream = fake_stream
    return mock_client


def make_non_streaming_client(json_body: dict, raise_exc: Exception | None = None) -> AsyncMock:
    """Build a mock httpx client whose .post() returns a JSON response."""
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = json_body

    mock_client = AsyncMock()
    if raise_exc:
        mock_client.post = AsyncMock(side_effect=raise_exc)
    else:
        mock_client.post = AsyncMock(return_value=mock_response)
    return mock_client


# ---------------------------------------------------------------------------
# _run_streaming_request
# ---------------------------------------------------------------------------


async def test_streaming_success_two_content_tokens():
    lines = [
        'data: {"choices": [{"delta": {"content": "Hello"}}]}',
        'data: {"choices": [{"delta": {"content": " World"}}]}',
        'data: {"choices": [{"delta": {}}], "usage": {"completion_tokens": 2}}',
        "data: [DONE]",
    ]
    client = make_streaming_client(lines)
    result = await _run_streaming_request(client, "http://localhost/v1/chat/completions", {})

    assert result.success is True
    assert result.tokens_generated == 2
    assert result.ttft_s is not None
    assert result.total_s is not None
    assert len(result.inter_token_latencies) == 2


async def test_streaming_skips_non_data_lines():
    lines = [
        ": ping",
        "",
        'data: {"choices": [{"delta": {"content": "Hi"}}]}',
        "data: [DONE]",
    ]
    client = make_streaming_client(lines)
    result = await _run_streaming_request(client, "http://localhost/v1/chat/completions", {})

    assert result.success is True
    assert result.tokens_generated == 1


async def test_streaming_skips_malformed_json():
    lines = [
        "data: {not valid json}",
        'data: {"choices": [{"delta": {"content": "Hi"}}]}',
        "data: [DONE]",
    ]
    client = make_streaming_client(lines)
    result = await _run_streaming_request(client, "http://localhost/v1/chat/completions", {})

    assert result.success is True
    assert result.tokens_generated == 1


async def test_streaming_no_content_tokens():
    lines = [
        'data: {"choices": [{"delta": {}}]}',
        "data: [DONE]",
    ]
    client = make_streaming_client(lines)
    result = await _run_streaming_request(client, "http://localhost/v1/chat/completions", {})

    assert result.success is True
    assert result.ttft_s is None
    assert result.tokens_generated == 0
    assert result.inter_token_latencies == []


async def test_streaming_usage_overrides_count():
    # 3 content chunks but usage says 10
    lines = [
        'data: {"choices": [{"delta": {"content": "a"}}]}',
        'data: {"choices": [{"delta": {"content": "b"}}]}',
        'data: {"choices": [{"delta": {"content": "c"}}]}',
        'data: {"choices": [{"delta": {}}], "usage": {"completion_tokens": 10}}',
        "data: [DONE]",
    ]
    client = make_streaming_client(lines)
    result = await _run_streaming_request(client, "http://localhost/v1/chat/completions", {})

    assert result.tokens_generated == 10


async def test_streaming_http_error_returns_failure():
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = Exception("HTTP 500")

    @asynccontextmanager
    async def fake_stream(*args, **kwargs):
        yield mock_response

    mock_client = MagicMock()
    mock_client.stream = fake_stream

    result = await _run_streaming_request(mock_client, "http://localhost/v1/chat/completions", {})

    assert result.success is False
    assert "HTTP 500" in result.error


async def test_streaming_connection_error_returns_failure():
    @asynccontextmanager
    async def fake_stream(*args, **kwargs):
        raise ConnectionError("refused")
        yield  # make it a generator

    mock_client = MagicMock()
    mock_client.stream = fake_stream

    result = await _run_streaming_request(mock_client, "http://localhost/v1/chat/completions", {})

    assert result.success is False
    assert result.error is not None


async def test_streaming_inter_token_latencies_length():
    # N content tokens should produce N inter-token latencies
    n = 4
    lines = [
        f'data: {{"choices": [{{"delta": {{"content": "t{i}"}}}}]}}'
        for i in range(n)
    ]
    lines.append("data: [DONE]")
    client = make_streaming_client(lines)
    result = await _run_streaming_request(client, "http://localhost/v1/chat/completions", {})

    assert result.success is True
    assert len(result.inter_token_latencies) == n


# ---------------------------------------------------------------------------
# _run_non_streaming_request
# ---------------------------------------------------------------------------


async def test_non_streaming_success_extracts_tokens():
    client = make_non_streaming_client({"usage": {"completion_tokens": 42}})
    result = await _run_non_streaming_request(client, "http://localhost/v1/chat/completions", {})

    assert result.success is True
    assert result.tokens_generated == 42
    assert result.ttft_s == result.total_s  # TTFT == total for non-streaming
    assert result.inter_token_latencies == []


async def test_non_streaming_missing_usage_defaults_to_zero():
    client = make_non_streaming_client({})
    result = await _run_non_streaming_request(client, "http://localhost/v1/chat/completions", {})

    assert result.success is True
    assert result.tokens_generated == 0


async def test_non_streaming_partial_usage():
    client = make_non_streaming_client({"usage": {}})
    result = await _run_non_streaming_request(client, "http://localhost/v1/chat/completions", {})

    assert result.success is True
    assert result.tokens_generated == 0


async def test_non_streaming_exception_returns_failure():
    client = make_non_streaming_client({}, raise_exc=Exception("timeout"))
    result = await _run_non_streaming_request(client, "http://localhost/v1/chat/completions", {})

    assert result.success is False
    assert "timeout" in result.error


async def test_non_streaming_http_error_returns_failure():
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = Exception("404 Not Found")
    mock_response.json.return_value = {}

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    result = await _run_non_streaming_request(mock_client, "http://localhost/v1/chat/completions", {})

    assert result.success is False


async def test_non_streaming_timing_positive():
    client = make_non_streaming_client({"usage": {"completion_tokens": 5}})
    result = await _run_non_streaming_request(client, "http://localhost/v1/chat/completions", {})

    assert result.total_s is not None
    assert result.total_s >= 0.0


# ---------------------------------------------------------------------------
# run_benchmark
# ---------------------------------------------------------------------------


async def test_run_benchmark_streaming_calls_streaming_fn():
    successful = RequestResult(success=True, ttft_s=0.1, total_s=1.0, tokens_generated=10)

    with patch("vllm_bench.benchmark._run_streaming_request", return_value=successful) as mock_fn:
        results, wall = await run_benchmark(
            base_url="http://localhost:8000",
            model="m",
            num_requests=3,
            concurrency=3,
            prompt="hi",
            max_tokens=64,
            streaming=True,
            api_key=None,
        )

    assert mock_fn.call_count == 3
    assert len(results) == 3
    assert wall >= 0.0


async def test_run_benchmark_non_streaming_calls_non_streaming_fn():
    successful = RequestResult(success=True, ttft_s=0.1, total_s=1.0, tokens_generated=10)

    with patch("vllm_bench.benchmark._run_non_streaming_request", return_value=successful) as mock_fn:
        results, wall = await run_benchmark(
            base_url="http://localhost:8000",
            model="m",
            num_requests=4,
            concurrency=4,
            prompt="hi",
            max_tokens=64,
            streaming=False,
            api_key=None,
        )

    assert mock_fn.call_count == 4
    assert len(results) == 4


async def test_run_benchmark_endpoint_url_construction():
    """run_benchmark must append /v1/chat/completions to base_url."""
    successful = RequestResult(success=True, ttft_s=0.1, total_s=1.0, tokens_generated=5)
    captured_urls: list[str] = []

    async def capture(client, url, payload):
        captured_urls.append(url)
        return successful

    with patch("vllm_bench.benchmark._run_non_streaming_request", side_effect=capture):
        await run_benchmark(
            base_url="http://localhost:8000/",
            model="m",
            num_requests=1,
            concurrency=1,
            prompt="hi",
            max_tokens=64,
            streaming=False,
            api_key=None,
        )

    assert captured_urls[0] == "http://localhost:8000/v1/chat/completions"


async def test_run_benchmark_auth_header_set():
    """When api_key is provided the client should carry an Authorization header."""
    successful = RequestResult(success=True, ttft_s=0.1, total_s=1.0, tokens_generated=5)
    captured_clients: list = []

    async def capture(client, url, payload):
        captured_clients.append(client)
        return successful

    with patch("vllm_bench.benchmark._run_non_streaming_request", side_effect=capture):
        await run_benchmark(
            base_url="http://localhost:8000",
            model="m",
            num_requests=1,
            concurrency=1,
            prompt="hi",
            max_tokens=64,
            streaming=False,
            api_key="secret-key",
        )

    assert len(captured_clients) == 1
    assert captured_clients[0].headers.get("authorization") == "Bearer secret-key"


async def test_run_benchmark_no_auth_header_without_api_key():
    successful = RequestResult(success=True, ttft_s=0.1, total_s=1.0, tokens_generated=5)
    captured_clients: list = []

    async def capture(client, url, payload):
        captured_clients.append(client)
        return successful

    with patch("vllm_bench.benchmark._run_non_streaming_request", side_effect=capture):
        await run_benchmark(
            base_url="http://localhost:8000",
            model="m",
            num_requests=1,
            concurrency=1,
            prompt="hi",
            max_tokens=64,
            streaming=False,
            api_key=None,
        )

    assert "authorization" not in captured_clients[0].headers


async def test_run_benchmark_progress_callback_called():
    successful = RequestResult(success=True, ttft_s=0.1, total_s=1.0, tokens_generated=5)
    progress: list[tuple[int, int]] = []

    def on_progress(done, total):
        progress.append((done, total))

    with patch("vllm_bench.benchmark._run_non_streaming_request", return_value=successful):
        await run_benchmark(
            base_url="http://localhost:8000",
            model="m",
            num_requests=5,
            concurrency=5,
            prompt="hi",
            max_tokens=64,
            streaming=False,
            api_key=None,
            progress_callback=on_progress,
        )

    assert len(progress) == 5
    # Final call should report all done
    assert progress[-1][0] == 5
    assert progress[-1][1] == 5


async def test_run_benchmark_no_progress_callback_ok():
    successful = RequestResult(success=True, ttft_s=0.1, total_s=1.0, tokens_generated=5)

    with patch("vllm_bench.benchmark._run_non_streaming_request", return_value=successful):
        results, _ = await run_benchmark(
            base_url="http://localhost:8000",
            model="m",
            num_requests=2,
            concurrency=2,
            prompt="hi",
            max_tokens=64,
            streaming=False,
            api_key=None,
            progress_callback=None,
        )

    assert len(results) == 2


async def test_run_benchmark_streaming_payload_includes_stream_options():
    """Streaming payloads must include stream_options.include_usage."""
    successful = RequestResult(success=True, ttft_s=0.1, total_s=1.0, tokens_generated=5)
    captured_payloads: list[dict] = []

    async def capture(client, url, payload):
        captured_payloads.append(payload)
        return successful

    with patch("vllm_bench.benchmark._run_streaming_request", side_effect=capture):
        await run_benchmark(
            base_url="http://localhost:8000",
            model="my-model",
            num_requests=1,
            concurrency=1,
            prompt="test",
            max_tokens=128,
            streaming=True,
            api_key=None,
        )

    payload = captured_payloads[0]
    assert payload["stream"] is True
    assert payload.get("stream_options", {}).get("include_usage") is True
    assert payload["model"] == "my-model"
    assert payload["max_tokens"] == 128


async def test_run_benchmark_non_streaming_payload_no_stream_options():
    successful = RequestResult(success=True, ttft_s=0.1, total_s=1.0, tokens_generated=5)
    captured_payloads: list[dict] = []

    async def capture(client, url, payload):
        captured_payloads.append(payload)
        return successful

    with patch("vllm_bench.benchmark._run_non_streaming_request", side_effect=capture):
        await run_benchmark(
            base_url="http://localhost:8000",
            model="m",
            num_requests=1,
            concurrency=1,
            prompt="test",
            max_tokens=64,
            streaming=False,
            api_key=None,
        )

    payload = captured_payloads[0]
    assert payload["stream"] is False
    assert "stream_options" not in payload


async def test_run_benchmark_returns_wall_time():
    successful = RequestResult(success=True, ttft_s=0.1, total_s=1.0, tokens_generated=5)

    with patch("vllm_bench.benchmark._run_non_streaming_request", return_value=successful):
        _, wall = await run_benchmark(
            base_url="http://localhost:8000",
            model="m",
            num_requests=1,
            concurrency=1,
            prompt="hi",
            max_tokens=64,
            streaming=False,
            api_key=None,
        )

    assert isinstance(wall, float)
    assert wall >= 0.0


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


def test_default_prompt_not_empty():
    assert len(_DEFAULT_PROMPT) > 0


def test_default_prompt_mentions_attention():
    assert "attention" in _DEFAULT_PROMPT.lower()
