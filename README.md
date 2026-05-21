# Efficient Inference Playground

Benchmark playground for Project D: compare efficient inference techniques for a small-to-medium LLM across speed, memory, and output quality.

## Project Scope

Primary backend:

- Hugging Face Transformers for baseline and KV-cache rows.
- Ollama GGUF fallback for the quantized row on this Apple Silicon machine when HF Metal INT4 is unstable.

Techniques covered:

- Baseline generation with default cache.
- 4-bit quantization via Ollama GGUF fallback (`Q4_K_M`) for the final quantized row on this machine.
- KV-cache policy experiments through a Gemma4-compatible hybrid recency-window cache.

Reviewed but intentionally not adopted as the primary KV row:

- Silvi's `Ollama + num_ctx=512` approximation, because it caps usable context instead of implementing sink-token retention plus a recent-window cache.

## Repository Layout

```text
configs/                 Experiment configs
prompts/                 Evaluation prompts at short, medium, and long lengths
scripts/                 CLI entrypoints for prompt prep, benchmarks, evaluation, summaries, and plots
src/eip/                 Reusable Python package code
results/raw/             Raw benchmark CSV files
results/processed/       Quality metrics and joined analysis CSV files
results/plots/           Generated figures
report/                  Technical report outline
presentation/            Presentation outline
team_members.txt         Team roster
```

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Generate the standardized prompt sets:

```bash
python scripts/prepare_benchmark_prompts.py \
  --mode needle \
  --count 10 \
  --out prompts/needle_eval_prompts.jsonl

python scripts/prepare_benchmark_prompts.py \
  --mode full \
  --count 10 \
  --out prompts/eval_prompts.jsonl
```

Check progress and get the next exact command:

```bash
.venv/bin/python scripts/check_benchmark_progress.py --markdown
```

Run the standardized Needle benchmark rows:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 caffeinate -dims .venv/bin/python scripts/run_hf_experiments.py \
  --config configs/hf_baseline_gemma4_e4b.json \
  --prompts prompts/needle_short_prompts.jsonl \
  --out results/raw/hf_baseline_short.csv

.venv/bin/python scripts/run_ollama_benchmark.py \
  --config configs/ollama_quantized_gemma4_e4b.json \
  --prompts prompts/needle_short_prompts.jsonl \
  --out results/raw/ollama_quantized_short.csv

PYTORCH_ENABLE_MPS_FALLBACK=1 caffeinate -dims .venv/bin/python scripts/run_hf_experiments.py \
  --config configs/hf_kv_window_gemma4_e4b.json \
  --prompts prompts/needle_short_prompts.jsonl \
  --out results/raw/hf_kv_short.csv
```

## Experiment Matrix

Current standardized matrix:

| Experiment | Backend | Technique | Notes |
| --- | --- | --- | --- |
| `hf_baseline_gemma4_e4b` | Transformers | Baseline | Default KV-cache; short/medium on MPS, long may use CPU fallback |
| `ollama_quantized_gemma4_e4b_q4km` | Ollama | Quantization | GGUF `Q4_K_M` fallback because HF Metal INT4 was unstable on this machine |
| `hf_kv_window_gemma4_e4b` | Transformers | KV-cache optimization | Hybrid recency-window: keep native sliding layers and apply a 512-token contiguous recency window to full-attention layers for short/medium; long final run uses 768 |

The full prompt generator standardizes all team buckets to 10 prompts each. The current Sargis execution scope is Needle short, medium, and long with 10 prompts per bucket.

## Metrics

Speed:

- end-to-end latency in seconds
- throughput in generated tokens per second
- per-token latency estimate

Memory:

- process RSS delta and peak RSS where observable
- system memory usage before and after each run
- these are reported as memory proxies, not true device VRAM measurements

Quality:

- exact match against prompt reference
- contains-match against prompt reference

Interpretation notes:

- latency is end-to-end wall-clock latency per prompt, not isolated decode-only latency
- per-token latency is a proxy computed as `latency_s / generated_tokens`
- tokens/sec is adequate for within-project comparison, but very small outputs can make it noisy
- baseline and quantized rows are already methodologically usable
- KV metrics are only final after the rerun under the new `recency_window` policy completes

Not collected in the current local setup:

- energy
- strict KL divergence to baseline
- ROUGE-L to baseline
- similarity-to-baseline metrics beyond exact/contains

Fallback candidate only:

- `Qwen/Qwen2.5-1.5B-Instruct` is kept as a documented contingency model for KV experiments if Gemma4 KV proves infeasible on local Apple Silicon. It is not the primary planned row.

## Team Reproduction

For teammates reproducing the current standardized setup:

1. Create and activate `.venv`, then install `requirements.txt`.
2. Use the committed prompt files in `prompts/` instead of regenerating new prompt sets.
3. Treat these files as canonical inputs:
   - `prompts/needle_short_prompts.jsonl`
   - `prompts/needle_medium_prompts.jsonl`
   - `prompts/needle_long_prompts.jsonl`
4. Treat these files as canonical baseline outputs:
   - `results/raw/hf_baseline_all_clean.csv`
   - `results/processed/hf_baseline_summary.csv`
5. For the quantized row on Apple Silicon, use the Ollama fallback config:
   - `configs/ollama_quantized_gemma4_e4b.json`
6. For the KV row, use only the current hybrid recency-window config:
   - `configs/hf_kv_window_gemma4_e4b.json`

Important:

- Do not use the older Gemma4 attention-sink KV results as final evidence.
- Do not claim VRAM or energy metrics; the repository reports memory proxies only.
- Compare long-latency rows only against other rows that used the same CPU fallback policy.


