# Experiment Matrix Status

Date: 2026-05-21

This repository now supports a standardized Needle benchmark with a mixed-backend fallback for the quantized row.

## Active Matrix

| Experiment | Technique | Output row target |
| --- | --- | --- |
| `hf_baseline_gemma4_e4b` | baseline | `results/raw/hf_baseline_all_clean.csv` |
| `ollama_quantized_gemma4_e4b_q4km` | GGUF Q4_K_M quantization fallback | `results/raw/ollama_quantized_*.csv` |
| `hf_kv_window_gemma4_e4b` | hybrid recency-window KV-cache optimization | `results/raw/hf_kv_*.csv` |

## Standardized Run Commands

```bash
.venv/bin/python scripts/check_benchmark_progress.py --markdown

.venv/bin/python scripts/run_hf_experiments.py \
  --config configs/hf_baseline_gemma4_e4b.json \
  --prompts prompts/needle_short_prompts.jsonl \
  --out results/raw/hf_baseline_short.csv

.venv/bin/python scripts/run_ollama_benchmark.py \
  --config configs/ollama_quantized_gemma4_e4b.json \
  --prompts prompts/needle_short_prompts.jsonl \
  --out results/raw/ollama_quantized_short.csv

.venv/bin/python scripts/run_hf_experiments.py \
  --config configs/hf_kv_window_gemma4_e4b.json \
  --prompts prompts/needle_short_prompts.jsonl \
  --out results/raw/hf_kv_short.csv

.venv/bin/python scripts/evaluate_task_quality.py \
  --input results/raw/hf_baseline_all_clean.csv \
  --prompts prompts/eval_prompts.jsonl \
  --out results/processed/hf_baseline_quality.csv

.venv/bin/python scripts/summarize_results.py \
  --input results/raw/hf_baseline_all_clean.csv \
  --quality results/processed/hf_baseline_quality.csv \
  --out_csv results/processed/hf_baseline_summary.csv \
  --out_md results/processed/hf_baseline_summary.md
```

## Notes

- `results/raw/hf_baseline_short.csv`, `results/raw/hf_baseline_medium.csv`, and `results/raw/hf_baseline_long.csv` are the valid baseline bucket files.
- `results/raw/hf_baseline_all_clean.csv` is the canonical baseline source built from those three bucket files.
- Speculative decoding is out of scope until a smaller draft model is available.
- The active KV-cache row uses a Gemma4-compatible hybrid recency-window policy: native sliding layers are preserved, and full-attention layers keep a contiguous 512-token recency window for short/medium; the final long run uses 768.
- Older Gemma4 attention-sink KV rows should not be treated as canonical evidence.
- HF Metal INT4 failed on this machine because the external Apple Metal quantization kernel stack was unstable, so the quantized row falls back to Ollama GGUF `Q4_K_M`.
- Long baseline and long KV must be compared only against rows that used the same CPU fallback policy.
