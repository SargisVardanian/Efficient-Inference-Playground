from __future__ import annotations

import argparse
import csv
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eip.io_utils import read_jsonl


EXPECTED_RUNS = [
    {
        "label": "baseline short",
        "experiment": "hf_baseline_gemma4_e4b",
        "csv": "results/raw/hf_baseline_short.csv",
        "prompts": "prompts/needle_short_prompts.jsonl",
        "command": 'PYTORCH_ENABLE_MPS_FALLBACK=1 caffeinate -dims .venv/bin/python scripts/run_hf_experiments.py --config configs/hf_baseline_gemma4_e4b.json --prompts prompts/needle_short_prompts.jsonl --out results/raw/hf_baseline_short.csv',
    },
    {
        "label": "baseline medium",
        "experiment": "hf_baseline_gemma4_e4b",
        "csv": "results/raw/hf_baseline_medium.csv",
        "prompts": "prompts/needle_medium_prompts.jsonl",
        "command": 'PYTORCH_ENABLE_MPS_FALLBACK=1 caffeinate -dims .venv/bin/python scripts/run_hf_experiments.py --config configs/hf_baseline_gemma4_e4b.json --prompts prompts/needle_medium_prompts.jsonl --out results/raw/hf_baseline_medium.csv',
    },
    {
        "label": "baseline long",
        "experiment": "hf_baseline_gemma4_e4b",
        "csv": "results/raw/hf_baseline_long.csv",
        "prompts": "prompts/needle_long_prompts.jsonl",
        "command": 'PYTORCH_ENABLE_MPS_FALLBACK=1 caffeinate -dims .venv/bin/python scripts/run_hf_experiments.py --config configs/hf_baseline_gemma4_e4b.json --prompts prompts/needle_long_prompts.jsonl --out results/raw/hf_baseline_long.csv',
    },
    {
        "label": "quantized short",
        "experiment": "ollama_quantized_gemma4_e4b_q4km",
        "csv": "results/raw/ollama_quantized_short.csv",
        "prompts": "prompts/needle_short_prompts.jsonl",
        "command": '.venv/bin/python scripts/run_ollama_benchmark.py --config configs/ollama_quantized_gemma4_e4b.json --prompts prompts/needle_short_prompts.jsonl --out results/raw/ollama_quantized_short.csv',
    },
    {
        "label": "quantized medium",
        "experiment": "ollama_quantized_gemma4_e4b_q4km",
        "csv": "results/raw/ollama_quantized_medium.csv",
        "prompts": "prompts/needle_medium_prompts.jsonl",
        "command": '.venv/bin/python scripts/run_ollama_benchmark.py --config configs/ollama_quantized_gemma4_e4b.json --prompts prompts/needle_medium_prompts.jsonl --out results/raw/ollama_quantized_medium.csv',
    },
    {
        "label": "quantized long",
        "experiment": "ollama_quantized_gemma4_e4b_q4km",
        "csv": "results/raw/ollama_quantized_long.csv",
        "prompts": "prompts/needle_long_prompts.jsonl",
        "command": '.venv/bin/python scripts/run_ollama_benchmark.py --config configs/ollama_quantized_gemma4_e4b.json --prompts prompts/needle_long_prompts.jsonl --out results/raw/ollama_quantized_long.csv',
    },
    {
        "label": "kv short",
        "experiment": "hf_kv_window_gemma4_e4b",
        "csv": "results/raw/hf_kv_short.csv",
        "prompts": "prompts/needle_short_prompts.jsonl",
        "command": 'PYTORCH_ENABLE_MPS_FALLBACK=1 caffeinate -dims .venv/bin/python scripts/run_hf_experiments.py --config configs/hf_kv_window_gemma4_e4b.json --prompts prompts/needle_short_prompts.jsonl --out results/raw/hf_kv_short.csv',
    },
    {
        "label": "kv medium",
        "experiment": "hf_kv_window_gemma4_e4b",
        "csv": "results/raw/hf_kv_medium.csv",
        "prompts": "prompts/needle_medium_prompts.jsonl",
        "command": 'PYTORCH_ENABLE_MPS_FALLBACK=1 caffeinate -dims .venv/bin/python scripts/run_hf_experiments.py --config configs/hf_kv_window_gemma4_e4b.json --prompts prompts/needle_medium_prompts.jsonl --out results/raw/hf_kv_medium.csv',
    },
    {
        "label": "kv long",
        "experiment": "hf_kv_window_gemma4_e4b",
        "csv": "results/raw/hf_kv_long.csv",
        "prompts": "prompts/needle_long_prompts.jsonl",
        "command": 'PYTORCH_ENABLE_MPS_FALLBACK=1 caffeinate -dims .venv/bin/python scripts/run_hf_experiments.py --config configs/hf_kv_window_gemma4_e4b_long_768.json --prompts prompts/needle_long_prompts.jsonl --out results/raw/hf_kv_long.csv',
    },
]


def completed_prompt_ids(csv_path: Path, experiment: str) -> set[str]:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return set()
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return {
            row["prompt_id"]
            for row in reader
            if row.get("experiment") == experiment and row.get("prompt_id")
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args()

    rows = []
    next_command = None
    for spec in EXPECTED_RUNS:
        expected = len(read_jsonl(spec["prompts"]))
        done = len(completed_prompt_ids(Path(spec["csv"]), spec["experiment"]))
        status = "done" if done >= expected else ("partial" if done > 0 else "pending")
        rows.append(
            {
                "label": spec["label"],
                "status": status,
                "completed": done,
                "expected": expected,
                "csv": spec["csv"],
                "command": spec["command"],
            }
        )
        if next_command is None and done < expected:
            next_command = spec["command"]

    if args.markdown:
        print("| Run | Status | Completed | File |")
        print("| --- | --- | ---: | --- |")
        for row in rows:
            print(f"| `{row['label']}` | {row['status']} | {row['completed']}/{row['expected']} | `{row['csv']}` |")
        print("")
        if next_command:
            print("Next command:")
            print("```bash")
            print(next_command)
            print("```")
        else:
            print("All planned runs are complete.")
    else:
        for row in rows:
            print(f"{row['label']}: {row['completed']}/{row['expected']} [{row['status']}] -> {row['csv']}")
        if next_command:
            print("\nNEXT COMMAND:")
            print(next_command)
        else:
            print("\nAll planned runs are complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
