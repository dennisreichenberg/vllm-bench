# vllm-bench

A CLI tool to benchmark [vLLM](https://github.com/vllm-project/vllm) server performance under concurrent load.

Measures:
- **TTFT** — Time To First Token (p50 / p90 / p99)
- **Throughput** — tokens/second and requests/second
- **Inter-Token Latency (ITL)** — in streaming mode
- **Success rate** — across all concurrent requests

---

## Installation

```bash
pip install vllm-bench
```

Or from source:

```bash
git clone https://github.com/dennisreichenberg/vllm-bench
cd vllm-bench
pip install -e .
```

---

## Quick Start

Make sure vLLM is running:

```bash
vllm serve meta-llama/Llama-3-8B-Instruct
```

Then run:

```bash
vllm-bench run --model meta-llama/Llama-3-8B-Instruct
```

---

## Usage

```
vllm-bench run [OPTIONS]

Options:
  -u, --url TEXT           Base URL of the vLLM server  [default: http://localhost:8000]
  -m, --model TEXT         Model name to benchmark  [required]
  -n, --num-requests INT   Total number of requests to send  [default: 100]
  -c, --concurrency INT    Number of concurrent requests  [default: 10]
  -p, --prompt TEXT        Prompt text to use (defaults to a built-in prompt)
      --max-tokens INT     Maximum tokens to generate per request  [default: 256]
      --streaming          Use streaming mode (default)
      --no-streaming       Use non-streaming mode
      --api-key TEXT       API key ($VLLM_API_KEY env var also works)
  -o, --output PATH        Write JSON report to this file
```

---

## Examples

**Basic benchmark with defaults (100 requests, 10 concurrent, streaming):**

```bash
vllm-bench run --model meta-llama/Llama-3-8B-Instruct
```

**High-concurrency load test:**

```bash
vllm-bench run --model mistralai/Mistral-7B-Instruct-v0.2 -n 500 -c 50
```

**Non-streaming mode, save JSON report:**

```bash
vllm-bench run --model llama3 --no-streaming -n 50 -c 5 -o report.json
```

**Custom prompt and max tokens:**

```bash
vllm-bench run \
  --model meta-llama/Llama-3-8B-Instruct \
  --prompt "Write a haiku about distributed systems." \
  --max-tokens 64 \
  -n 200 -c 20
```

**Remote server with API key:**

```bash
VLLM_API_KEY=my-secret vllm-bench run \
  --url https://my-vllm.example.com \
  --model Qwen/Qwen2.5-7B-Instruct \
  -n 100 -c 10
```

---

## Sample Output

```
vllm-bench — meta-llama/Llama-3-8B-Instruct
Server: http://localhost:8000
Config: 100 requests · 10 concurrent · 256 max tokens · streaming

  [██████████████████████████████] 100/100

         vllm-bench Results — meta-llama/Llama-3-8B-Instruct
┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Metric                ┃         Value ┃
┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ Mode                  │     streaming │
│ Total Requests        │           100 │
│ Successful            │           100 │
│ Failed                │             0 │
│ Success Rate          │        100.0% │
│                       │               │
│ Request Throughput    │    8.42 req/s │
│ Total Wall Time       │      11.88 s  │
│                       │               │
│ TTFT Mean             │      52.3 ms  │
│ TTFT p50              │      48.1 ms  │
│ TTFT p90              │      89.4 ms  │
│ TTFT p99              │     124.7 ms  │
│                       │               │
│ Tokens/s Mean         │       98.4    │
│ Tokens/s p50          │       97.1    │
│                       │               │
│ ITL Mean              │      10.2 ms  │
│ ITL p90               │      18.7 ms  │
└───────────────────────┴───────────────┘
```

---

## JSON Report Format

When using `--output report.json`:

```json
{
  "timestamp": "2025-04-18T12:00:00+00:00",
  "model": "meta-llama/Llama-3-8B-Instruct",
  "mode": "streaming",
  "requests": {
    "total": 100,
    "successful": 100,
    "failed": 0,
    "success_rate": 1.0
  },
  "throughput": {
    "request_throughput_rps": 8.42,
    "tokens_per_second_mean": 98.4,
    "tokens_per_second_p50": 97.1
  },
  "latency": {
    "ttft_mean_ms": 52.3,
    "ttft_p50_ms": 48.1,
    "ttft_p90_ms": 89.4,
    "ttft_p99_ms": 124.7
  },
  "inter_token_latency": {
    "itl_mean_ms": 10.2,
    "itl_p90_ms": 18.7
  },
  "total_wall_s": 11.88
}
```

---

## License

MIT © Dennis Reichenberg
