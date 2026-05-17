.PHONY: setup check benchmark quality plots all

setup:
	python3 -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt

check:
	python scripts/check_ollama.py --model gemma4:e4b

benchmark:
	python scripts/run_ollama_benchmark.py --config configs/ollama_gemma4.json --prompts prompts/eval_prompts.jsonl --out results/raw/ollama_benchmark.csv

quality:
	python scripts/evaluate_quality.py --input results/raw/ollama_benchmark.csv --out results/processed/quality_metrics.csv

plots:
	python scripts/make_plots.py --input results/raw/ollama_benchmark.csv --quality results/processed/quality_metrics.csv --outdir results/plots

all: check benchmark quality plots

