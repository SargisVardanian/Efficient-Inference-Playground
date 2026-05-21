# Efficient Inference Playground

**AUA NLP Course — Project D**
Comparing efficient-inference techniques on a small-to-medium LLM. We measure speed and memory savings — and the corresponding quality loss — across three techniques and six prompt length buckets.

---

## What This Project Does (Plain English)

Large language models are slow and memory-hungry. Engineers use tricks to make them faster: compressing the weights ("quantization"), letting a small model guess for a big model ("speculative decoding"), or throwing away old context the model no longer needs ("KV-cache eviction").

We take **one model**, run it on **30 / 60 prompts** spanning summarisation and reasoning at short / medium / long lengths, and compare three optimisation techniques head-to-head against the unoptimised baseline.

---

## Techniques Compared

| # | Technique | Backend | Implementation |
|---|---|---|---|
| 1 | **Baseline (Q4 quantised)** | Ollama | `gemma4:e4b` — Gemma 4 8B compressed to 4-bit GGUF (Q4_K_M). Full 8 192-token context. |
| 2 | **KV-Cache eviction** | Ollama | Same model, context window capped at 512 tokens (`num_ctx=512`). Simulates sliding-window cache. |
| 3 | **SinkCache** | HuggingFace | Qwen2.5-1.5B-Instruct with `SinkCache(window_length=256, num_sink_tokens=4)`. |

> **Note on Gemma 4 + HuggingFace.** Loading `google/gemma-4-E4B-it` through HuggingFace Transformers requires the dev branch (Python ≥ 3.10) and 19 GB+ free memory just to instantiate the processor. Teammate Sargis documented this as a hard blocker (`results/processed/experiment_matrix_status.md`). We therefore evaluate SinkCache on **Qwen2.5-1.5B-Instruct**, which is a fair comparison because SinkCache is a **technique-level intervention** — the relative speedup/memory pattern is what matters, not the underlying model identity. The Ollama experiments continue to use Gemma 4 as planned.

---

## Evaluation Set (30 prompts)

### Summarisation
| Bucket | Dataset | Prompt length |
|---|---|---|
| Short | CNN/DailyMail | ~150 tokens |
| Medium | CNN/DailyMail | ~800 tokens |
| Long | GovReport (`ccdv/govreport-summarization`) | ~8 000 tokens (truncated) |

### Reasoning
| Bucket | Dataset | Prompt length |
|---|---|---|
| Short | GSM8K | ~80 tokens |
| Medium | GSM8K | ~1 000 tokens (padded with filler) |
| Long | GSM8K | ~8 000 tokens (padded with filler) |

GSM8K medium/long prompts are intentionally padded — they stress-test both reasoning **and** long-context handling: can the model still find and solve the math problem when it is buried in irrelevant context?

> **5 prompts per bucket** by default. Set `N_PER_BUCKET = 10` at the top of `scripts/prepare_prompts.py` to double the eval set to 60 prompts.

---

## Metrics

**Speed:**
- End-to-end latency (s) — p50 and p99 across 3 runs
- First-token latency (s) — streaming Ollama only
- Throughput (tokens / second)

**Memory:**
- Process RSS delta (MB) before vs after generation
- System memory used (MB)

**Quality** (vs same-family baseline output):
- ROUGE-L
- Exact match
- Character-level similarity

---

## Repository Layout

```
configs/                  Experiment configs (Ollama JSON)
  ollama_gemma4.json        baseline experiment
  ollama_kvcache.json       num_ctx=512 KV-cache eviction experiment
prompts/
  eval_prompts.jsonl        the 30 / 60 evaluation prompts
scripts/
  prepare_prompts.py        builds prompts from CNN/DailyMail, GovReport, GSM8K
  run_ollama_benchmark.py   runs Ollama experiments → results/raw/*.csv
  run_hf_sinkcache.py       runs baseline + SinkCache via HF Transformers
  evaluate_quality.py       ROUGE-L / exact-match / char-sim vs baseline
  combine_and_analyze.py    merges all CSVs, computes summary, writes Excel
  make_plots.py             latency / throughput / ROUGE bar charts
src/eip/                  Reusable library (metrics, Ollama client, plotting)
results/
  raw/                      Per-run CSV output (git-ignored)
  processed/                Quality metrics + summary CSVs (git-ignored)
  plots/                    PNGs (git-ignored)
  benchmark_results.xlsx    Styled multi-sheet workbook (git-ignored)
report/                   Technical report outline
presentation/             Presentation outline
team_members.txt          Team roster
```

---

## Quick Start (Apple Silicon / Mac)

### 1. Set up environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt datasets openpyxl protobuf
```

### 2. Install Ollama and pull the model

```bash
brew install ollama
ollama serve &              # leave running in background
ollama pull gemma4:e4b      # ~9.6 GB download (Q4_K_M GGUF)
```

### 3. Generate the prompts (CNN, GovReport, GSM8K)

```bash
python scripts/prepare_prompts.py
```

### 4. Run all three experiments

```bash
# Baseline (Gemma 4 Q4_K_M, 8192 ctx)
python scripts/run_ollama_benchmark.py \
  --config configs/ollama_gemma4.json \
  --prompts prompts/eval_prompts.jsonl \
  --out results/raw/ollama_benchmark.csv

# KV-Cache eviction (Gemma 4 Q4_K_M, 512 ctx)
python scripts/run_ollama_benchmark.py \
  --config configs/ollama_kvcache.json \
  --prompts prompts/eval_prompts.jsonl \
  --out results/raw/kvcache_benchmark.csv

# SinkCache (HF Qwen2.5-1.5B-Instruct, both baseline + sinkcache)
python scripts/run_hf_sinkcache.py \
  --prompts prompts/eval_prompts.jsonl \
  --out results/raw/hf_sinkcache.csv
```

### 5. Combine, evaluate, and produce the Excel report

```bash
python scripts/combine_and_analyze.py
```

Outputs:
- `results/processed/all_runs.csv` — harmonised per-run table
- `results/processed/quality_metrics.csv` — ROUGE-L / exact-match / char-sim per run
- `results/processed/summary.csv` — aggregated per (experiment, length, task)
- `results/benchmark_results.xlsx` — styled workbook (5 sheets)

### 6. Optional plots

```bash
python scripts/make_plots.py \
  --input results/raw/ollama_benchmark.csv \
  --quality results/processed/quality_metrics.csv \
  --outdir results/plots
```

---

## Results (60 prompts × 3 runs, Apple M4 Max)

### Latency — seconds per generation (median, lower = better)

| Experiment | Short (~150 tok) | Medium (~800 tok) | Long (~8 000 tok) |
|---|---:|---:|---:|
| `baseline_gemma4_e4b` (8 192 ctx) | **1.62** | **1.64** | **1.77** ⭐ |
| `kvcache_limited_512` (512 ctx)   | 1.68 | 1.66 | 3.25 ❌ |

### Throughput — tokens / second (median, higher = better)

| Experiment | Short | Medium | Long |
|---|---:|---:|---:|
| `baseline_gemma4_e4b` | 90.2 | 89.8 | 85.8 |
| `kvcache_limited_512` | 89.0 | 89.4 | 87.2 |

### Key Finding

**KV-cache eviction is a memory trade-off, not a speed trade-off.**

On short and medium prompts the two configurations are statistically identical — the cache was never the bottleneck. On the long 8 000-token prompt, however, capping the context window to 512 tokens makes inference **~2× slower** (3.25 s vs 1.77 s).

This is initially counter-intuitive but mechanically clear:

1. Ollama still has to ingest the full 8 000-token input regardless of `num_ctx`.
2. Constraining the cache to 512 tokens forces continuous eviction and re-computation of attention during prompt processing.
3. The eviction overhead exceeds any memory savings on a fast unified-memory chip with plenty of RAM headroom (37 GB).

**Implication.** KV-cache eviction (SinkCache, H2O, StreamingLLM, etc.) pays off when:
- VRAM is the binding constraint (e.g., 8 GB laptop GPU running long-context inference);
- many concurrent requests share one model and per-request memory must shrink;
- prompts dwarf the cache budget by orders of magnitude.

On a single-user M4 Max running one prompt at a time, KV-cache eviction is a **cost**, not a win. The performance cliff appears only at the long-prompt extreme, exactly where attention-cache pressure becomes real.

### Reproducing These Numbers

```bash
python scripts/prepare_prompts.py
python scripts/run_ollama_benchmark.py --config configs/ollama_gemma4.json   --prompts prompts/eval_prompts.jsonl --out results/raw/ollama_benchmark.csv
python scripts/combine_and_analyze.py
open results/benchmark_results.xlsx
```

Full results land in `results/benchmark_results.xlsx` (5 styled sheets: Summary, Latency, Throughput, ROUGE-L, All Runs).

---

## Hardware

Tested on Apple M4 Max (37 GB unified memory, Metal GPU). All inference runs locally without CUDA.

---

## Known Limitations

1. **No HuggingFace Gemma 4 path.** Documented in detail above. Mitigation: SinkCache evaluated on Qwen2.5; baseline / KV-cache evaluated on Gemma 4 via Ollama (already Q4 quantised).
2. **`gemma4:e4b` is already quantised.** The "baseline" is therefore a Q4_K_M baseline, not full-precision. A true FP16 baseline would require the HF path, which is blocked (see #1).
3. **Speculative decoding not yet measured.** Scripted (`scripts/run_hf_speculative.py`) but not executed in this round; would require a small Qwen draft model. Listed as future work.
4. **Quality metrics are output-vs-baseline.** ROUGE-L is computed against the **baseline experiment's own output**, not against the dataset reference. This measures *how much the optimisation changed the model's behaviour*, which is the right signal for an inference-optimisation study.

---

## Future Work

- Run speculative decoding (Qwen2.5-3B target + Qwen2.5-0.5B draft) and report acceptance rate.
- Compare KV-cache eviction policies (sliding window vs SinkCache vs H2O).
- Pareto curves of quality × latency × memory.
- Energy measurements via `powermetrics` on Apple Silicon.
