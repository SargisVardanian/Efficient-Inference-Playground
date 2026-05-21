# Needle Benchmark Plan

This project now standardizes the long-context retrieval slice around 30 synthetic Needle-in-a-Haystack prompts:

- `needle_short_*`: 10 prompts
- `needle_medium_*`: 10 prompts
- `needle_long_*`: 10 prompts

Current comparison matrix for local Apple Silicon work:

| Experiment | Backend | Technique | Status |
| --- | --- | --- | --- |
| `hf_baseline_gemma4_e4b` | Transformers | Baseline | Implemented |
| `ollama_quantized_gemma4_e4b_q4km` | Ollama | GGUF `Q4_K_M` quantization fallback | Implemented |
| `hf_kv_window_gemma4_e4b` | Transformers | Hybrid recency-window KV cache | Implemented and rerunning |

Notes:

- Speculative decoding is intentionally omitted from the current run matrix because no smaller draft model is available locally.
- The original local Attention Sink implementation was reviewed and found incompatible with Gemma4's hybrid sliding/full attention structure under the current HF cache-mask contract.
- The active KV-cache path now uses a Gemma4-compatible hybrid recency-window policy: native sliding layers are preserved, and full-attention layers keep a contiguous 512-token recency window for short/medium; the final long run uses 768.
- Earlier KV short rows collected under the old attention-sink implementation should not be treated as canonical evidence.
- Silvi's reviewed KV approximation (`Ollama + num_ctx=512`) is intentionally not adopted as the primary KV row because it truncates the effective context window instead of implementing sink-plus-recent-window retention.
- The repository is standardized around one common result schema, not one common backend. Quantized remains a documented mixed-backend fallback.

Reference links:

- Hugging Face Transformers Metal quantization docs: https://huggingface.co/docs/transformers/quantization/metal
- Hugging Face Transformers KV cache docs: https://huggingface.co/docs/transformers/kv_cache
- Hugging Face model card, Qwen2.5-1.5B-Instruct (fallback candidate only): https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
- Xiao et al., 2023, "Efficient Streaming Language Models with Attention Sinks": https://arxiv.org/abs/2309.17453
