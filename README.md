# Efficient Inference Playground

**AUA NLP Course — Project D**

Benchmark playground for comparing efficient inference techniques on a small-to-medium LLM. We measure how much speed and memory you gain — and how much output quality you lose — when applying inference optimisations.

---

## Techniques Compared

| Technique | Backend | Description |
|---|---|---|
| **Baseline** | Ollama | `gemma4:e4b` (4-bit quantised GGUF) at full context (8 192 tokens) |
| **KV-Cache Eviction** | Ollama | Same model, context window capped at 512 tokens — simulates SinkCache-style eviction |
| **Speculative Decoding** | HuggingFace | Large target model + small draft model via `assistant_model` API |

> `gemma4:e4b` is a 4-bit GGUF quantised model, so the baseline experiment simultaneously represents quantised inference on Apple Silicon.

---

## Evaluation Datasets

### Summarisation
| Split | Dataset | Prompt length |
|---|---|---|
| Short | CNN/DailyMail | ~150 tokens |
| Medium | CNN/DailyMail | ~800 tokens |
| Long | GovReport (`ccdv/govreport-summarization`) | ~8 000 tokens (truncated) |

### Reasoning
| Split | Dataset | Prompt length |
|---|---|---|
| Short | GSM8K | ~80 tokens (raw problem) |
| Medium | GSM8K | ~1 000 tokens (padded with filler context) |
| Long | GSM8K | ~8 000 tokens (padded with filler context) |

**5 prompts per split → 30 prompts total.**

GSM8K medium/long prompts are padded intentionally: they test whether the model can extract and solve a math problem when it is surrounded by irrelevant context — stress-testing both reasoning and long-context handling.

---

## Repository Layout

```
configs/            Experiment configs (Ollama JSON)
prompts/            30 evaluation prompts (JSONL)
scripts/            CLI scripts: prepare prompts, run benchmarks, evaluate, plot
src/eip/            Reusable Python package (metrics, Ollama client, plotting, system stats)
results/raw/        Raw benchmark CSVs  (git-ignored, generated locally)
results/processed/  Quality metrics CSVs (git-ignored, generated locally)
results/plots/      Output figures       (git-ignored, generated locally)
report/             Technical report outline
presentation/       Presentation outline
team_members.txt    Team roster
```

---

## Quick Start

### 1. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt datasets
```

### 2. Install and start Ollama, pull the model

```bash
brew install ollama
ollama serve &           # keep running in background
ollama pull gemma4:e4b   # ~9.6 GB download
```

### 3. Generate the 30 evaluation prompts

```bash
python scripts/prepare_prompts.py
```

Expected output:
```
CNN short: 5 | CNN medium: 5 | GovReport long: 5
GSM8K short: 5 | GSM8K medium: 5 | GSM8K long: 5
Total: 30 prompts
```

### 4. Run experiments

**Baseline (gemma4:e4b, full 8 192-token context):**
```bash
python scripts/run_ollama_benchmark.py \
  --config configs/ollama_gemma4.json \
  --prompts prompts/eval_prompts.jsonl \
  --out results/raw/ollama_benchmark.csv
```

**KV-Cache eviction (512-token context window):**
```bash
python scripts/run_ollama_benchmark.py \
  --config configs/ollama_kvcache.json \
  --prompts prompts/eval_prompts.jsonl \
  --out results/raw/kvcache_benchmark.csv
```

**Speculative decoding (HuggingFace Transformers):**
```bash
python scripts/run_hf_speculative.py \
  --target_model Qwen/Qwen2.5-3B-Instruct \
  --assistant_model Qwen/Qwen2.5-0.5B-Instruct \
  --prompts prompts/eval_prompts.jsonl \
  --out results/raw/hf_speculative.csv \
  --max_new_tokens 128
```

### 5. Evaluate output quality

```bash
python scripts/evaluate_quality.py \
  --input results/raw/ollama_benchmark.csv \
  --out results/processed/quality_metrics.csv
```

### 6. Generate plots

```bash
python scripts/make_plots.py \
  --input results/raw/ollama_benchmark.csv \
  --quality results/processed/quality_metrics.csv \
  --outdir results/plots
```

---

## Metrics Collected

**Speed:**
- End-to-end latency (seconds) — p50 and p99 across 3 runs
- First-token latency (seconds)
- Throughput (tokens / second)

**Memory:**
- Process RSS delta (MB)
- System memory used (MB)

**Quality:**
- ROUGE-L vs baseline output
- Exact match vs baseline output
- Embedding cosine similarity (sentence-transformers)

---

## Experiment Configuration

Each Ollama experiment is a JSON file under `configs/`. Key fields:

```json
{
  "runs_per_prompt": 3,
  "generation_options": {
    "temperature": 0,
    "num_ctx": 8192,
    "num_predict": 128
  },
  "experiments": [
    {
      "name": "baseline_gemma4_e4b",
      "model": "gemma4:e4b",
      "technique": "baseline"
    }
  ]
}
```

`runs_per_prompt: 3` provides median and p99 latency estimates.

---

## Hardware

Tested on Apple M4 Max (37 GB unified memory). Inference runs on Metal (Apple GPU). No CUDA required.
