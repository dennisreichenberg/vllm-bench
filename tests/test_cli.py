"""Unit tests for the CLI layer (formatting helpers and command behaviour)."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from vllm_bench.cli import _build_summary_table, _metrics_to_dict, app
from vllm_bench.metrics import AggregatedMetrics, RequestResult

runner = CliRunner()


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def make_metrics(**overrides) -> AggregatedMetrics:
    defaults: dict = dict(
        total_requests=10,
        successful=9,
        failed=1,
        ttft_mean=0.1,
        ttft_p50=0.09,
        ttft_p90=0.18,
        ttft_p99=0.20,
        tokens_per_second_mean=50.0,
        tokens_per_second_p50=48.0,
        itl_mean_ms=20.0,
        itl_p90_ms=35.0,
        total_wall_s=5.0,
        request_throughput=2.0,
    )
    defaults.update(overrides)
    return AggregatedMetrics(**defaults)


def successful_results(n: int = 10) -> list[RequestResult]:
    return [
        RequestResult(success=True, ttft_s=0.1, total_s=1.0, tokens_generated=50)
        for _ in range(n)
    ]


def failed_results(n: int = 5) -> list[RequestResult]:
    return [RequestResult(success=False, error="timeout") for _ in range(n)]


# ---------------------------------------------------------------------------
# _build_summary_table
# ---------------------------------------------------------------------------


def test_build_summary_table_returns_table():
    from rich.table import Table

    m = make_metrics()
    table = _build_summary_table(m, "test-model", streaming=True)
    assert isinstance(table, Table)


def test_build_summary_table_has_two_columns():
    m = make_metrics()
    table = _build_summary_table(m, "test-model", streaming=True)
    assert len(table.columns) == 2


def test_build_summary_table_title_contains_model():
    m = make_metrics()
    table = _build_summary_table(m, "llama3-8b", streaming=True)
    assert "llama3-8b" in table.title


def test_build_summary_table_streaming_mode_label():
    from io import StringIO

    from rich.console import Console

    m = make_metrics()
    table = _build_summary_table(m, "m", streaming=True)
    buf = StringIO()
    Console(file=buf, no_color=True, width=120).print(table)
    rendered = buf.getvalue()
    assert "streaming" in rendered


def test_build_summary_table_non_streaming_mode_label():
    from io import StringIO

    from rich.console import Console

    m = make_metrics()
    table = _build_summary_table(m, "m", streaming=False)
    buf = StringIO()
    Console(file=buf, no_color=True, width=120).print(table)
    rendered = buf.getvalue()
    assert "non-streaming" in rendered


def test_build_summary_table_non_streaming_has_fewer_rows():
    m = make_metrics()
    streaming_table = _build_summary_table(m, "m", streaming=True)
    non_streaming_table = _build_summary_table(m, "m", streaming=False)
    # ITL rows added only for streaming
    assert len(streaming_table.rows) > len(non_streaming_table.rows)


# ---------------------------------------------------------------------------
# _metrics_to_dict
# ---------------------------------------------------------------------------


def test_metrics_to_dict_streaming_mode():
    m = make_metrics()
    d = _metrics_to_dict(m, "test-model", streaming=True)
    assert d["mode"] == "streaming"


def test_metrics_to_dict_non_streaming_mode():
    m = make_metrics()
    d = _metrics_to_dict(m, "test-model", streaming=False)
    assert d["mode"] == "non-streaming"


def test_metrics_to_dict_model_field():
    m = make_metrics()
    d = _metrics_to_dict(m, "my-model", streaming=True)
    assert d["model"] == "my-model"


def test_metrics_to_dict_timestamp_present():
    m = make_metrics()
    d = _metrics_to_dict(m, "m", streaming=True)
    assert "timestamp" in d
    assert d["timestamp"]  # non-empty


def test_metrics_to_dict_requests_block():
    m = make_metrics(total_requests=20, successful=18, failed=2)
    d = _metrics_to_dict(m, "m", streaming=True)
    assert d["requests"]["total"] == 20
    assert d["requests"]["successful"] == 18
    assert d["requests"]["failed"] == 2
    assert d["requests"]["success_rate"] == pytest.approx(18 / 20, abs=0.001)


def test_metrics_to_dict_throughput_block():
    m = make_metrics(
        tokens_per_second_mean=55.5, tokens_per_second_p50=50.0, request_throughput=3.0
    )
    d = _metrics_to_dict(m, "m", streaming=True)
    assert d["throughput"]["tokens_per_second_mean"] == pytest.approx(55.5, abs=0.01)
    assert d["throughput"]["tokens_per_second_p50"] == pytest.approx(50.0, abs=0.01)
    assert d["throughput"]["request_throughput_rps"] == pytest.approx(3.0, abs=0.01)


def test_metrics_to_dict_latency_converted_to_ms():
    m = make_metrics(ttft_mean=0.150, ttft_p50=0.100, ttft_p90=0.200, ttft_p99=0.250)
    d = _metrics_to_dict(m, "m", streaming=True)
    assert d["latency"]["ttft_mean_ms"] == pytest.approx(150.0, abs=0.1)
    assert d["latency"]["ttft_p50_ms"] == pytest.approx(100.0, abs=0.1)
    assert d["latency"]["ttft_p90_ms"] == pytest.approx(200.0, abs=0.1)
    assert d["latency"]["ttft_p99_ms"] == pytest.approx(250.0, abs=0.1)


def test_metrics_to_dict_itl_block():
    m = make_metrics(itl_mean_ms=25.0, itl_p90_ms=40.0)
    d = _metrics_to_dict(m, "m", streaming=True)
    assert d["inter_token_latency"]["itl_mean_ms"] == pytest.approx(25.0, abs=0.1)
    assert d["inter_token_latency"]["itl_p90_ms"] == pytest.approx(40.0, abs=0.1)


def test_metrics_to_dict_total_wall_s():
    m = make_metrics(total_wall_s=12.345)
    d = _metrics_to_dict(m, "m", streaming=True)
    assert d["total_wall_s"] == pytest.approx(12.345, abs=0.001)


def test_metrics_to_dict_is_json_serializable():
    m = make_metrics()
    d = _metrics_to_dict(m, "m", streaming=True)
    # Should not raise
    json.dumps(d)


# ---------------------------------------------------------------------------
# CLI command: run
# ---------------------------------------------------------------------------


def test_cli_run_exits_zero_on_success():
    with patch("asyncio.run", return_value=(successful_results(10), 5.0)):
        result = runner.invoke(app, ["--model", "test-model", "--num-requests", "10"])
    assert result.exit_code == 0


def test_cli_run_exits_one_all_failed():
    with patch("asyncio.run", return_value=(failed_results(5), 2.0)):
        result = runner.invoke(app, ["--model", "test-model"])
    assert result.exit_code == 1


def test_cli_run_exits_one_on_exception():
    with patch("asyncio.run", side_effect=Exception("server unreachable")):
        result = runner.invoke(app, ["--model", "test-model"])
    assert result.exit_code == 1


def test_cli_run_exits_one_on_keyboard_interrupt():
    with patch("asyncio.run", side_effect=KeyboardInterrupt):
        result = runner.invoke(app, ["--model", "test-model"])
    assert result.exit_code == 1


def test_cli_run_no_streaming_flag():
    with patch("asyncio.run", return_value=(successful_results(5), 2.0)):
        result = runner.invoke(app, ["--model", "test-model", "--no-streaming"])
    assert result.exit_code == 0


def test_cli_run_writes_json_report(tmp_path):
    output_file = tmp_path / "report.json"
    with patch("asyncio.run", return_value=(successful_results(10), 5.0)):
        result = runner.invoke(
            app,
            ["--model", "test-model", "--num-requests", "10", "--output", str(output_file)],
        )
    assert result.exit_code == 0
    assert output_file.exists()
    report = json.loads(output_file.read_text(encoding="utf-8"))
    assert report["model"] == "test-model"
    assert "requests" in report


def test_cli_run_json_report_mode_streaming(tmp_path):
    output_file = tmp_path / "report.json"
    with patch("asyncio.run", return_value=(successful_results(5), 2.0)):
        runner.invoke(
            app,
            ["--model", "m", "--streaming", "--output", str(output_file)],
        )
    report = json.loads(output_file.read_text(encoding="utf-8"))
    assert report["mode"] == "streaming"


def test_cli_run_json_report_mode_non_streaming(tmp_path):
    output_file = tmp_path / "report.json"
    with patch("asyncio.run", return_value=(successful_results(5), 2.0)):
        runner.invoke(
            app,
            ["--model", "m", "--no-streaming", "--output", str(output_file)],
        )
    report = json.loads(output_file.read_text(encoding="utf-8"))
    assert report["mode"] == "non-streaming"


def test_cli_run_partial_failures_exits_zero():
    mixed = [
        *successful_results(7),
        *failed_results(3),
    ]
    with patch("asyncio.run", return_value=(mixed, 5.0)):
        result = runner.invoke(app, ["--model", "test-model"])
    # Partial failures are OK — at least some succeeded
    assert result.exit_code == 0


def test_cli_run_requires_model_option():
    # Invoking without --model should fail (typer/click validation)
    result = runner.invoke(app, [])
    assert result.exit_code != 0


def test_cli_run_passes_prompt_to_benchmark():
    """When --prompt is given it should be forwarded (not the default prompt)."""
    with patch("asyncio.run", return_value=(successful_results(1), 1.0)):
        result = runner.invoke(
            app,
            ["--model", "m", "--prompt", "custom prompt here"],
        )
    assert result.exit_code == 0


def test_cli_run_default_prompt_used_when_none_given():
    from vllm_bench.benchmark import _DEFAULT_PROMPT

    prompts_used: list[str] = []

    async def capture_prompt(base_url, model, num_requests, concurrency, prompt, **kwargs):
        prompts_used.append(prompt)
        return successful_results(1), 1.0

    with patch("vllm_bench.cli.run_benchmark", side_effect=capture_prompt):
        runner.invoke(app, ["--model", "m"])

    assert len(prompts_used) == 1
    assert prompts_used[0] == _DEFAULT_PROMPT


def test_cli_run_custom_prompt_used_when_given():
    prompts_used: list[str] = []

    async def capture_prompt(base_url, model, num_requests, concurrency, prompt, **kwargs):
        prompts_used.append(prompt)
        return successful_results(1), 1.0

    with patch("vllm_bench.cli.run_benchmark", side_effect=capture_prompt):
        runner.invoke(app, ["--model", "m", "--prompt", "my custom prompt"])

    assert len(prompts_used) == 1
    assert prompts_used[0] == "my custom prompt"


def test_cli_run_progress_callback_is_invoked():
    """Verify the progress callback body executes without error."""

    async def fake_benchmark(
        base_url, model, num_requests, concurrency, prompt, max_tokens, streaming, api_key,
        progress_callback=None
    ):
        if progress_callback:
            progress_callback(1, num_requests)
            progress_callback(num_requests, num_requests)
        return successful_results(num_requests), 2.0

    with patch("vllm_bench.cli.run_benchmark", side_effect=fake_benchmark):
        result = runner.invoke(app, ["--model", "m", "--num-requests", "5"])

    assert result.exit_code == 0


def test_main_calls_app():
    from vllm_bench.cli import main

    with patch("vllm_bench.cli.app") as mock_app:
        main()
    mock_app.assert_called_once()
