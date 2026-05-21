# Final Benchmark Results Summary

Generated: 2026-05-22

## Canonical Result Files

- `results/sargis_needle_benchmark_results.xlsx` - Sargis Needle benchmark workbook in the same multi-sheet format as Silvi's workbook.
- `results/team_final_benchmark_results.xlsx` - combined Sargis + Silvi workbook with `owner` and `hardware` columns.
- `datasets/sargis_needle/prompts.jsonl` - canonical 30-prompt Needle dataset used for Sargis's final results.
- `results/processed/sargis_needle_summary.md` - markdown summary for Sargis Needle rows.
- `results/processed/sargis_needle_final_summary.csv` - generated summary CSV; ignored by default but reproducible from the workbook script.
- `results/processed/team_final_summary.csv` - generated team summary CSV; ignored by default but reproducible from the workbook script.

## Hardware Split

| Owner | Hardware | Benchmark family |
| --- | --- | --- |
| Sargis | Apple M3 Pro | Needle-in-a-haystack retrieval |
| Silvi | Apple M4 Max | CNN/GovReport summarization + GSM8K reasoning |

## Sargis Needle Summary

| Experiment | Backend | Bucket | p50 latency (s) | p99 latency (s) | tok/s | exact | contains |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| HF baseline | Transformers | short | 293.70 | 569.19 | 0.0183 | 0.80 | 0.90 |
| HF baseline | Transformers | medium | 278.28 | 1042.56 | 0.0178 | 1.00 | 1.00 |
| HF baseline | Transformers | long | 1109.35 | 1145.92 | 0.0050 | 0.90 | 1.00 |
| Ollama GGUF Q4_K_M | Ollama | short | 0.68 | 5.67 | 40.6433 | 0.80 | 0.80 |
| Ollama GGUF Q4_K_M | Ollama | medium | 2.74 | 4.44 | 39.2641 | 0.50 | 0.90 |
| Ollama GGUF Q4_K_M | Ollama | long | 13.78 | 91.82 | 34.5823 | 0.10 | 0.60 |
| HF recency-window KV | Transformers | short | 314.27 | 522.13 | 0.0165 | 0.90 | 0.90 |
| HF recency-window KV | Transformers | medium | 154.64 | 378.18 | 0.0186 | 0.30 | 0.30 |
| HF recency-window KV | Transformers | long | 1185.56 | 1261.75 | 0.0021 | 0.10 | 0.10 |

## Interpretation

- Sargis baseline and KV rows use HF Transformers on M3 Pro; long rows used the same CPU fallback policy and are comparable to each other.
- Sargis quantized row is mixed-backend Ollama GGUF `Q4_K_M`, so it is a practical local fallback result, not an HF Metal INT4 result.
- KV recency-window is a workload-fit negative result for Needle: the task has long prefill and very short decode, so cache eviction does not create the speedup expected in long-generation workloads.
- Memory columns are RSS/system-memory proxies, not true VRAM.
- Silvi's workbook is sourced from `origin/experiments/baseline-kvcache:results/benchmark_results.xlsx` and is preserved in the combined workbook format.
