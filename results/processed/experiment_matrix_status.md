# Experiment Matrix Status

Date: 2026-05-20

## Short Answer

Only one variant has completed numeric metrics on this local machine:

| Variant | Backend | Model | Status | Numeric metrics |
| --- | --- | --- | --- | --- |
| Installed local E4B Q4 run | Ollama | `gemma4:e4b` | Completed | Yes |
| E4B baseline + default KV-cache | Hugging Face Transformers | `google/gemma-4-E4B-it` | Blocked locally | No |
| E4B INT4/Q4 + default KV-cache | Hugging Face Transformers Metal | `google/gemma-4-E4B-it` | Blocked locally | No |
| E4B + KV optimization | Hugging Face Transformers | `google/gemma-4-E4B-it` | Blocked locally | No |
| E4B + speculative E2B draft | Hugging Face Transformers | target E4B + draft E2B | Skipped by scope | No |

## Why This Is Not Yet a Full Baseline / INT4 / KV Comparison

The local Ollama model is already quantized:

```text
ollama show gemma4:e4b
parameters: 8.0B
quantization: Q4_K_M
context length: 131072
```

That means the completed Ollama table is not a full-precision baseline. It is a local Q4 GGUF result.

For a fair comparison, the rows should come from one backend:

| Fair comparison row | Correct backend | Reason |
| --- | --- | --- |
| Full E4B baseline | Hugging Face Transformers | Loads the same HF model with default KV-cache |
| E4B INT4/Q4 | Hugging Face Transformers | Uses the same HF runtime plus quantization config |
| E4B + KV optimization | Hugging Face Transformers | Exposes cache controls such as sliding-window cache |

## Local HF Attempts

| Attempt | Command path | Result |
| --- | --- | --- |
| Full HF E4B baseline smoke run | `scripts/run_hf_experiments.py --config /tmp/hf_baseline_gemma4_e4b.json --prompts /tmp/needle_smoke.jsonl` | Downloaded HF weights but did not finish load/generate on this 19GB Apple Silicon machine in a practical time window. |
| HF INT4 Metal smoke run | `MetalConfig(bits=4)` through `scripts/run_hf_experiments.py` | Failed during model load with `RuntimeError: Invalid buffer size: 15.05 GiB`. |
| HF KV-cache optimization | `cache_implementation="sliding_window"` config exists in `configs/hf_gemma4_e4b_needle.json` | Not numerically runnable locally because it depends on the same HF E4B load path as baseline. |

## Completed Numeric Result: Local Ollama Q4

Source files:

| File | Purpose |
| --- | --- |
| `results/raw/ollama_needle_benchmark.csv` | Raw per-prompt run data |
| `results/processed/ollama_needle_quality.csv` | Exact/contains answer quality |
| `results/processed/ollama_needle_summary.csv` | Aggregated assignment metrics |
| `results/processed/ollama_needle_summary.md` | Markdown summary table |

Summary:

| Length | Prompts | Mean prompt tokens | Latency p50, s | Latency p99, s | Per-token latency p50, s | Per-token latency p99, s | Mean tokens/sec | Exact answer match | Contains answer match |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Short | 10 | 236.6 | 0.68 | 51.40 | 0.096 | 12.829 | 40.62 | 0.80 | 0.80 |
| Medium | 10 | 1812.6 | 2.70 | 4.39 | 0.284 | 1.062 | 38.69 | 0.50 | 0.90 |
| Long | 10 | 7155.6 | 13.88 | 18.84 | 0.588 | 1.964 | 33.30 | 0.10 | 0.60 |

## What Must Be Run Next for a Real Full Result

Run this on a machine that can load `google/gemma-4-E4B-it` in Hugging Face:

```bash
python scripts/run_hf_experiments.py \
  --config configs/hf_gemma4_e4b_needle.json \
  --prompts prompts/needle_eval_prompts.jsonl \
  --out results/raw/hf_needle_benchmark.csv

python scripts/evaluate_task_quality.py \
  --input results/raw/hf_needle_benchmark.csv \
  --prompts prompts/needle_eval_prompts.jsonl \
  --out results/processed/hf_needle_quality.csv

python scripts/summarize_results.py \
  --input results/raw/hf_needle_benchmark.csv \
  --quality results/processed/hf_needle_quality.csv \
  --out_csv results/processed/hf_needle_summary.csv \
  --out_md results/processed/hf_needle_summary.md
```

Expected HF rows from `configs/hf_gemma4_e4b_needle.json`:

| Experiment | Technique |
| --- | --- |
| `hf_baseline_gemma4_e4b` | baseline |
| `hf_quantized_gemma4_e4b_int4` | INT4 Metal quantization |
| `hf_kv_window_gemma4_e4b` | sliding-window KV-cache optimization |

