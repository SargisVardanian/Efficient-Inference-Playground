# Efficient Inference Playground

Benchmark playground for Project D: compare efficient inference techniques for a small-to-medium LLM across speed, memory, and output quality.

## Project Scope

Primary backend:

- `Ollama` with local `gemma4:e4b` for baseline measurements.
- Additional Ollama model tags can be added for quantized variants when available locally.

Secondary backend:

- Hugging Face Transformers for standardized baseline, quantization, speculative decoding, and KV-cache policy experiments.

Techniques covered:

- Baseline generation.
- Weight quantization through Ollama model variants when available, or through Transformers quantization backends.
- Speculative decoding through Transformers target + assistant models.
- KV-cache policy experiments through Transformers cache controls.

Optional extension:

- KV-cache policy experiments or long-context cache settings, if time and hardware allow.

## Repository Layout

```text
configs/                 Experiment configs
prompts/                 Evaluation prompts at short, medium, and long lengths
scripts/                 CLI entrypoints for checks, benchmarks, evaluation, and plots
src/eip/                 Reusable Python package code
results/raw/             Raw benchmark CSV files
results/processed/       Quality metrics and joined analysis CSV files
results/plots/           Generated figures
report/                  Technical report outline
presentation/            Presentation outline
team_members.txt         Team roster
```

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Check local Ollama models:

```bash
ollama list
python scripts/check_ollama.py --model gemma4:e4b
```

Generate the standardized Needle-only prompt set:

```bash
python scripts/prepare_benchmark_prompts.py \
  --mode needle \
  --count 10 \
  --out prompts/needle_eval_prompts.jsonl
```

Run the Ollama benchmark:

```bash
python scripts/run_ollama_benchmark.py \
  --config configs/ollama_gemma4.json \
  --prompts prompts/eval_prompts.jsonl \
  --out results/raw/ollama_benchmark.csv
```

Compute quality metrics against the baseline output:

```bash
python scripts/evaluate_quality.py \
  --input results/raw/ollama_benchmark.csv \
  --out results/processed/quality_metrics.csv
```

Generate plots:

```bash
python scripts/make_plots.py \
  --input results/raw/ollama_benchmark.csv \
  --quality results/processed/quality_metrics.csv \
  --outdir results/plots
```

Run speculative decoding through Hugging Face:

```bash
python scripts/run_hf_speculative.py \
  --target_model Qwen/Qwen2.5-3B-Instruct \
  --assistant_model Qwen/Qwen2.5-0.5B-Instruct \
  --prompts prompts/eval_prompts.jsonl \
  --out results/raw/hf_speculative.csv \
  --max_new_tokens 128
```

Run the standardized Hugging Face baseline / int4 / KV benchmark:

```bash
python scripts/run_hf_experiments.py \
  --config configs/hf_gemma4_e4b_needle.json \
  --prompts prompts/needle_eval_prompts.jsonl \
  --out results/raw/hf_needle_benchmark.csv

python scripts/evaluate_task_quality.py \
  --input results/raw/hf_needle_benchmark.csv \
  --out results/processed/hf_needle_quality.csv

python scripts/summarize_results.py \
  --input results/raw/hf_needle_benchmark.csv \
  --quality results/processed/hf_needle_quality.csv \
  --out_csv results/processed/hf_needle_summary.csv \
  --out_md results/processed/hf_needle_summary.md
```

## Experiment Matrix

Start with this minimum table:

| Experiment | Backend | Technique | Notes |
| --- | --- | --- | --- |
| `baseline_gemma4_e4b` | Ollama | Baseline | Uses installed `gemma4:e4b` |
| `quantized_variant_1` | Ollama | Quantization | Replace with installed Q4/Q8 tag |
| `hf_baseline` | Transformers | Baseline | Same target model without assistant |
| `hf_speculative` | Transformers | Speculative decoding | Target + smaller assistant |

Measure each row on short, medium, and long prompts.

## Metrics

Speed:

- end-to-end latency in seconds
- first-token latency in seconds for streaming Ollama runs
- throughput in generated tokens per second
- per-token latency estimate

Memory:

- process RSS delta and peak RSS where observable
- optional `nvidia-smi` peak VRAM on CUDA machines
- optional `powermetrics` or external tooling for energy

Quality:

- exact match against baseline greedy output
- ROUGE-L against baseline output
- embedding similarity when `sentence-transformers` is available

## Deadline Note

The assignment deadline is Friday, 22 May 2026, 12:00 AM. Treat this as the beginning of Friday and plan to submit on Thursday evening, 21 May 2026.
