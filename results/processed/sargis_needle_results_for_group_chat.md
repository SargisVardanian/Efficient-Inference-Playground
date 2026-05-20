# Sargis Needle Results

Date: 2026-05-20

## Scope

Sargis side is standardized to 30 Needle-in-a-Haystack prompts:

| Task | Short | Medium | Long |
| --- | ---: | ---: | ---: |
| Needle retrieval | 10 | 10 | 10 |

Prompt file: `prompts/needle_eval_prompts.jsonl`

## Completed Local Run

Available local model:

| Backend | Model | Reported quantization | Context |
| --- | --- | --- | --- |
| Ollama | `gemma4:e4b` | `Q4_K_M` | 8192 |

Summary with assignment metrics:

| Length | Prompts | Mean prompt tokens | Latency p50, s | Latency p99, s | Per-token latency p50, s | Per-token latency p99, s | Mean tokens/sec | Peak process RSS, MB | Peak system used, MB | Exact answer match | Contains answer match |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Short | 10 | 236.6 | 0.68 | 51.40 | 0.096 | 12.829 | 40.62 | 25.48 | 13871.84 | 0.80 | 0.80 |
| Medium | 10 | 1812.6 | 2.70 | 4.39 | 0.284 | 1.062 | 38.69 | 26.89 | 13926.42 | 0.50 | 0.90 |
| Long | 10 | 7155.6 | 13.88 | 18.84 | 0.588 | 1.964 | 33.30 | 27.44 | 13941.22 | 0.10 | 0.60 |

Notes:

- `tokens/sec` comes from Ollama `eval_count / eval_duration`.
- `per-token latency` is end-to-end latency divided by generated token count.
- Peak VRAM is not directly exposed by Ollama on Apple Silicon; the table reports peak process RSS and peak system used memory as the available local memory proxy.
- Energy was not measured because no reliable non-privileged energy counter was available in this run.

Files:

| File | Purpose |
| --- | --- |
| `results/raw/ollama_needle_benchmark.csv` | Raw per-prompt latency, token, memory, and output rows |
| `results/processed/ollama_needle_quality.csv` | Reference exact/contains quality metrics |
| `results/processed/ollama_needle_summary.csv` | Aggregated metrics table |
| `results/processed/ollama_needle_summary.md` | Markdown metrics table |

## Experiment Feasibility Status

| Experiment | Intended backend | Status | Notes |
| --- | --- | --- | --- |
| E4B baseline + default KV-cache | Transformers | Blocked locally | Full HF `google/gemma-4-E4B-it` weights downloaded, but full-precision MPS load/generate did not finish on this 19GB Apple Silicon machine. |
| E4B INT4 / Q4 weights + default KV-cache | Transformers Metal | Blocked locally | `MetalConfig(bits=4)` starts, but fails with `RuntimeError: Invalid buffer size: 15.05 GiB`. |
| E4B + KV optimization | Transformers | Blocked locally | Requires the same HF model path as baseline. Local `transformers` exposes sliding-window cache, but not the older top-level `SinkCache` API. |
| Installed local E4B Q4 run | Ollama | Completed | `ollama show gemma4:e4b` reports `Q4_K_M`; this is the runnable local result above. |

## Backend Decision

For a fair comparison of baseline vs INT4 vs KV-cache, use one backend for all compared rows. Mixing Ollama and Hugging Face is not a clean apples-to-apples comparison because the installed Ollama model is already Q4 GGUF, while Hugging Face loads safetensors with a different runtime and memory behavior.

Practical current path:

1. Use Ollama result as the local available Q4/GGUF benchmark.
2. Run true HF baseline / INT4 / KV on a larger machine or Colab-tier GPU/MPS environment.
3. Keep speculative decoding out of the current table until an E2B draft model is available.

## References

- Hugging Face Transformers KV cache docs, current docs: https://huggingface.co/docs/transformers/v4.50.0/kv_cache
- Hugging Face Transformers Metal quantization docs, current docs: https://huggingface.co/docs/transformers/quantization/metal
- Xiao et al., 2023, "Efficient Streaming Language Models with Attention Sinks": https://arxiv.org/abs/2309.17453
