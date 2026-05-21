# Sargis Needle Results

Date: 2026-05-21

## Scope

Sargis side is standardized to 30 Needle-in-a-Haystack prompts:

| Task | Short | Medium | Long |
| --- | ---: | ---: | ---: |
| Needle retrieval | 10 | 10 | 10 |

Prompt file: `prompts/needle_eval_prompts.jsonl`

## Active Experiment Matrix

This file tracks the final mixed-backend comparison that is currently feasible on this Apple Silicon machine:

| Experiment | Backend | Technique |
| --- | --- | --- |
| `hf_baseline_gemma4_e4b` | Transformers | baseline |
| `ollama_quantized_gemma4_e4b_q4km` | Ollama | GGUF Q4_K_M quantized fallback |
| `hf_kv_window_gemma4_e4b` | Transformers | hybrid recency-window KV-cache optimization |

## Result Files

| File | Purpose |
| --- | --- |
| `results/raw/hf_baseline_all_clean.csv` | Canonical baseline raw rows |
| `results/raw/hf_baseline_long.csv` | Canonical long baseline raw rows |
| `results/raw/ollama_quantized_*.csv` | Canonical quantized raw rows on this Apple Silicon machine |
| `results/raw/hf_kv_*.csv` | KV-cache raw rows |
| `results/processed/hf_baseline_summary.csv` | Baseline aggregated metrics table |
| `results/processed/summary.csv` | Compatibility summary output |

## Shareable Table Template

Current baseline rows are filled from `results/raw/hf_baseline_all_clean.csv` and now cover all `10 short + 10 medium + 10 long` prompts.

| Experiment | Length | Latency p50, s | Latency p99, s | Per-token latency p50, s | Mean tokens/sec | Peak process RSS, MB | Exact answer match | Contains answer match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `hf_baseline_gemma4_e4b` | Short | 293.70 | 569.19 | 52.7775 | 0.0183 | 66.88 | 0.80 | 0.90 |
| `hf_baseline_gemma4_e4b` | Medium | 278.28 | 1042.56 | 55.0690 | 0.0178 | 73.12 | 1.00 | 1.00 |
| `hf_baseline_gemma4_e4b` | Long | 1109.35 | 1145.92 | 202.4802 | 0.0050 | 9468.39 | 0.90 | 1.00 |
| `ollama_quantized_gemma4_e4b_q4km` | Short | see final summary | see final summary | see final summary | see final summary | see final summary | see final summary | see final summary |
| `ollama_quantized_gemma4_e4b_q4km` | Medium | see final summary | see final summary | see final summary | see final summary | see final summary | see final summary | see final summary |
| `ollama_quantized_gemma4_e4b_q4km` | Long | see final summary | see final summary | see final summary | see final summary | see final summary | see final summary | see final summary |
| `hf_kv_window_gemma4_e4b` | Short | pending | pending | pending | pending | pending | pending | pending |
| `hf_kv_window_gemma4_e4b` | Medium | pending | pending | pending | pending | pending | pending | pending |
| `hf_kv_window_gemma4_e4b` | Long | pending | pending | pending | pending | pending | pending | pending |

Note: baseline long used CPU fallback because full Gemma4 E4B long-context generation did not fit on MPS. This is acceptable for collecting quality and a runnable long baseline, but latency/memory should be compared only against variants run with the same device policy.

Note: HF Metal INT4 on this machine failed due to external kernel/runtime issues in the Apple Metal quantization stack, so the quantized row is collected with Ollama GGUF instead.

Note: the first Gemma4 KV attempt used an attention-sink-style cache and was replaced. The active KV rerun uses a hybrid recency-window cache that is compatible with Gemma4's sliding/full attention structure. Only the rerun should be used in final comparisons.

Note: reported memory is a process/system memory proxy, not true VRAM. Energy, KL divergence, ROUGE-L-to-baseline, and strict similarity-to-baseline are not collected in the current local setup.

## References

- Hugging Face Transformers KV cache docs, current docs: https://huggingface.co/docs/transformers/v4.50.0/kv_cache
- Hugging Face Transformers Metal quantization docs, current docs: https://huggingface.co/docs/transformers/quantization/metal
- Xiao et al., 2023, "Efficient Streaming Language Models with Attention Sinks": https://arxiv.org/abs/2309.17453
