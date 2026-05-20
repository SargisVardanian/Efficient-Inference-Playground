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

Summary:

| Length | Prompts | Mean prompt tokens | Mean latency, s | Mean tokens/sec | Exact answer match | Contains answer match |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Short | 10 | 236.6 | 6.38 | 40.62 | 0.80 | 0.80 |
| Medium | 10 | 1812.6 | 2.85 | 38.69 | 0.50 | 0.90 |
| Long | 10 | 7155.6 | 12.68 | 33.30 | 0.10 | 0.60 |

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
