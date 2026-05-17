# Technical Report Outline

Target length: 6-8 pages.

## 1. Abstract

Briefly state the target model, efficient inference techniques, prompt length sweep, and headline tradeoff.

## 2. Problem and Motivation

Explain why LLM inference cost matters. State that the project benchmarks inference only, without fine-tuning.

## 3. Methods

- Baseline generation with local Gemma 4 through Ollama.
- Quantized model variants through Ollama tags.
- Speculative decoding through Hugging Face Transformers target + assistant models.
- Optional KV-cache experiment if implemented.

## 4. Experimental Setup

- Hardware and OS.
- Model identifiers and quantization tags.
- Prompt buckets: short around 128 tokens, medium around 1K tokens, long around 8K tokens where feasible.
- Generation settings: greedy decoding, max new tokens, context window.

## 5. Metrics

- Latency, first-token latency, p50/p99 latency.
- Throughput in generated tokens/sec.
- Memory: RAM/VRAM where observable.
- Quality: exact match, ROUGE-L, and similarity to baseline output.

## 6. Results

Insert tables and plots from `results/processed/summary.csv` and `results/plots/`.

## 7. Discussion

Compare tradeoffs: speedup, memory reduction, quality degradation, hardware limitations, and reproducibility limits.

## 8. Deployment Recipe

Give a practical recommendation for Colab-tier or local laptop hardware.

## 9. Conclusion

State which technique or combination is best under the measured constraints.

## References

Use the primary-source list in `docs/references.md` for speculative decoding, quantization, and KV-cache eviction/compression.
