# Needle Benchmark Plan

This project now standardizes the long-context retrieval slice around 30 synthetic Needle-in-a-Haystack prompts:

- `needle_short_*`: 10 prompts
- `needle_medium_*`: 10 prompts
- `needle_long_*`: 10 prompts

Current comparison matrix for local Apple Silicon work:

| Experiment | Backend | Technique | Status |
| --- | --- | --- | --- |
| `hf_baseline_gemma4_e4b` | Transformers | Baseline | Implemented |
| `hf_quantized_gemma4_e4b_int4` | Transformers | 4-bit Metal quantization | Implemented |
| `hf_kv_window_gemma4_e4b` | Transformers | Sliding-window KV cache | Implemented |

Notes:

- Speculative decoding is intentionally omitted from the current run matrix because no smaller draft model is available locally.
- The KV-cache path uses a runtime-compatible sliding-window cache on this machine. The config keeps a `num_sink_tokens` field for documentation parity with the original StreamingLLM-style intent, but the local `transformers` runtime does not expose the earlier `SinkCache` API directly.
- Ollama remains useful for the originally scaffolded baseline path, but the standardized E4B baseline/int4/KV comparison is now configured through Transformers so all three runs use the same backend and schema.

Reference links:

- Hugging Face Transformers Metal quantization docs: https://huggingface.co/docs/transformers/quantization/metal
- Hugging Face Transformers KV cache docs: https://huggingface.co/docs/transformers/kv_cache
- Xiao et al., 2023, "Efficient Streaming Language Models with Attention Sinks": https://arxiv.org/abs/2309.17453
