"""CLI entry point for vllm-bench."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .benchmark import _DEFAULT_PROMPT, run_benchmark
from .metrics import AggregatedMetrics, aggregate

app = typer.Typer(
    name="vllm-bench",
    help="Benchmark a vLLM server — measure TTFT, throughput, and latency under concurrent load.",
    add_completion=False,
)
console = Console()
err = Console(stderr=True)


def _build_summary_table(m: AggregatedMetrics, model: str, streaming: bool) -> Table:
    table = Table(title=f"vllm-bench Results — {model}", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="bold white")
    table.add_column("Value", justify="right", style="green")

    mode = "streaming" if streaming else "non-streaming"
    table.add_row("Mode", mode)
    table.add_row("Total Requests", str(m.total_requests))
    table.add_row("Successful", str(m.successful))
    table.add_row("Failed", str(m.failed))
    table.add_row("Success Rate", f"{m.success_rate * 100:.1f}%")
    table.add_row("", "")
    table.add_row("Request Throughput", f"{m.request_throughput:.2f} req/s")
    table.add_row("Total Wall Time", f"{m.total_wall_s:.2f} s")
    table.add_row("", "")
    table.add_row("TTFT Mean", f"{m.ttft_mean * 1000:.1f} ms")
    table.add_row("TTFT p50", f"{m.ttft_p50 * 1000:.1f} ms")
    table.add_row("TTFT p90", f"{m.ttft_p90 * 1000:.1f} ms")
    table.add_row("TTFT p99", f"{m.ttft_p99 * 1000:.1f} ms")
    table.add_row("", "")
    table.add_row("Tokens/s Mean", f"{m.tokens_per_second_mean:.1f}")
    table.add_row("Tokens/s p50", f"{m.tokens_per_second_p50:.1f}")
    if streaming:
        table.add_row("", "")
        table.add_row("ITL Mean", f"{m.itl_mean_ms:.1f} ms")
        table.add_row("ITL p90", f"{m.itl_p90_ms:.1f} ms")

    return table


def _metrics_to_dict(m: AggregatedMetrics, model: str, streaming: bool) -> dict:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "mode": "streaming" if streaming else "non-streaming",
        "requests": {
            "total": m.total_requests,
            "successful": m.successful,
            "failed": m.failed,
            "success_rate": round(m.success_rate, 4),
        },
        "throughput": {
            "request_throughput_rps": round(m.request_throughput, 4),
            "tokens_per_second_mean": round(m.tokens_per_second_mean, 2),
            "tokens_per_second_p50": round(m.tokens_per_second_p50, 2),
        },
        "latency": {
            "ttft_mean_ms": round(m.ttft_mean * 1000, 2),
            "ttft_p50_ms": round(m.ttft_p50 * 1000, 2),
            "ttft_p90_ms": round(m.ttft_p90 * 1000, 2),
            "ttft_p99_ms": round(m.ttft_p99 * 1000, 2),
        },
        "inter_token_latency": {
            "itl_mean_ms": round(m.itl_mean_ms, 2),
            "itl_p90_ms": round(m.itl_p90_ms, 2),
        },
        "total_wall_s": round(m.total_wall_s, 4),
    }


@app.command()
def run(
    url: str = typer.Option(
        "http://localhost:8000", "--url", "-u", help="Base URL of the vLLM server."
    ),
    model: str = typer.Option(
        ..., "--model", "-m", help="Model name to benchmark (e.g. meta-llama/Llama-3-8B-Instruct)."
    ),
    num_requests: int = typer.Option(
        100, "--num-requests", "-n", help="Total number of requests to send.", min=1
    ),
    concurrency: int = typer.Option(
        10, "--concurrency", "-c", help="Number of concurrent requests.", min=1
    ),
    prompt: str | None = typer.Option(
        None, "--prompt", "-p", help="Prompt text to use. Defaults to a built-in prompt."
    ),
    max_tokens: int = typer.Option(
        256, "--max-tokens", help="Maximum tokens to generate per request.", min=1
    ),
    streaming: bool = typer.Option(
        True, "--streaming/--no-streaming", help="Use streaming mode (default: on)."
    ),
    api_key: str | None = typer.Option(
        None, "--api-key", envvar="VLLM_API_KEY", help="API key for the vLLM server."
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Write JSON report to this file."
    ),
) -> None:
    """Run a load benchmark against a vLLM OpenAI-compatible endpoint.

    \b
    Examples:
      vllm-bench run --model meta-llama/Llama-3-8B-Instruct
      vllm-bench run --model mistralai/Mistral-7B -n 200 -c 20
      vllm-bench run --model llama3 --no-streaming -n 50 -c 5 -o report.json
    """
    actual_prompt = prompt or _DEFAULT_PROMPT

    console.print(f"\n[bold]vllm-bench[/bold] — [cyan]{model}[/cyan]")
    console.print(f"[dim]Server:[/dim] {url}")
    console.print(
        f"[dim]Config:[/dim] {num_requests} requests · "
        f"{concurrency} concurrent · "
        f"{max_tokens} max tokens · "
        f"{'streaming' if streaming else 'non-streaming'}\n"
    )

    completed_count = 0

    def on_progress(done: int, total: int) -> None:
        nonlocal completed_count
        completed_count = done
        bar_width = 30
        filled = int(bar_width * done / total)
        bar = "█" * filled + "░" * (bar_width - filled)
        console.print(f"  [{bar}] {done}/{total}", end="\r")

    try:
        results, wall_time = asyncio.run(
            run_benchmark(
                base_url=url,
                model=model,
                num_requests=num_requests,
                concurrency=concurrency,
                prompt=actual_prompt,
                max_tokens=max_tokens,
                streaming=streaming,
                api_key=api_key,
                progress_callback=on_progress,
            )
        )
    except KeyboardInterrupt:
        err.print("\n[yellow]Interrupted.[/yellow]")
        raise typer.Exit(1)
    except Exception as exc:
        err.print(f"\n[red]Fatal error:[/red] {exc}")
        raise typer.Exit(1)

    console.print()  # clear progress line

    failed_results = [r for r in results if not r.success]
    if failed_results:
        sample_errors = [r.error for r in failed_results[:3] if r.error]
        if sample_errors:
            err.print(f"[yellow]Sample errors:[/yellow] {'; '.join(sample_errors)}")

    if not any(r.success for r in results):
        err.print(
            "\n[red]All requests failed.[/red] Check that the server is running"
            " and the model name is correct."
        )
        raise typer.Exit(1)

    metrics = aggregate(results, wall_time)
    console.print(_build_summary_table(metrics, model, streaming))

    if output:
        report = _metrics_to_dict(metrics, model, streaming)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        console.print(f"\n[dim]JSON report saved → {output}[/dim]")
    elif metrics.failed > 0:
        console.print(
            f"\n[yellow]{metrics.failed} request(s) failed.[/yellow] "
            "Re-run with --output report.json for full details."
        )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
