from __future__ import annotations

import argparse
import csv
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eip.io_utils import ensure_parent, read_json, read_jsonl
from eip.metrics import safe_tokens_per_second
from eip.ollama_client import OllamaClient
from eip.system_stats import current_memory


def read_completed_prompt_ids(path: Path, experiment_name: str) -> set[str]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return {
            row["prompt_id"]
            for row in reader
            if row.get("experiment") == experiment_name and row.get("prompt_id")
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = read_json(args.config)
    prompts = read_jsonl(args.prompts)
    client = OllamaClient(config.get("ollama_url", "http://localhost:11434"))
    runs_per_prompt = int(config.get("runs_per_prompt", 1))
    base_options = config.get("generation_options", {})
    experiments = [exp for exp in config["experiments"] if exp.get("enabled", True)]

    ensure_parent(args.out)
    fields = [
        "experiment",
        "technique",
        "backend",
        "model",
        "quantization_method",
        "quantization_bits",
        "cache_policy",
        "cache_window_length",
        "device_runtime_note",
        "prompt_id",
        "length_bucket",
        "task",
        "dataset",
        "reference",
        "run",
        "latency_s",
        "generated_tokens",
        "prompt_tokens",
        "tokens_per_second",
        "rss_before_mb",
        "rss_after_mb",
        "system_used_before_mb",
        "system_used_after_mb",
        "output",
    ]

    out_path = Path(args.out)
    auto_resume = not args.overwrite and out_path.exists() and out_path.stat().st_size > 0
    resume_mode = args.resume or auto_resume
    completed_by_experiment = {
        exp["name"]: read_completed_prompt_ids(out_path, exp["name"]) if resume_mode else set()
        for exp in experiments
    }
    open_mode = "a" if resume_mode and out_path.exists() else "w"
    write_header = not (resume_mode and out_path.exists() and out_path.stat().st_size > 0)

    with out_path.open(open_mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if write_header:
            writer.writeheader()
            handle.flush()

        for exp in experiments:
            options = dict(base_options)
            options.update(exp.get("options", {}))
            completed_prompt_ids = completed_by_experiment.get(exp["name"], set())
            for prompt in prompts:
                if prompt["id"] in completed_prompt_ids:
                    print(f"Skipping completed prompt {prompt['id']} for {exp['name']}.", flush=True)
                    continue
                for run_idx in range(runs_per_prompt):
                    before = current_memory()
                    result = client.generate(exp["model"], prompt["prompt"], options)
                    after = current_memory()
                    quantization = exp.get("quantization") or {}
                    cache_policy = exp.get("cache_policy") or {}
                    writer.writerow(
                        {
                            "experiment": exp["name"],
                            "technique": exp["technique"],
                            "backend": "ollama",
                            "model": exp["model"],
                            "quantization_method": quantization.get("method", "gguf"),
                            "quantization_bits": quantization.get("bits", 4),
                            "cache_policy": cache_policy.get("type", "default"),
                            "cache_window_length": cache_policy.get("window_length"),
                            "device_runtime_note": exp.get(
                                "device_runtime_note",
                                "Ollama GGUF fallback because HF Metal INT4 kernel loading failed on this Apple Silicon machine.",
                            ),
                            "prompt_id": prompt["id"],
                            "length_bucket": prompt["length_bucket"],
                            "task": prompt.get("task", ""),
                            "dataset": prompt.get("dataset", ""),
                            "reference": prompt.get("reference", ""),
                            "run": run_idx + 1,
                            "latency_s": result.latency_s,
                            "generated_tokens": result.generated_tokens,
                            "prompt_tokens": result.prompt_tokens,
                            "tokens_per_second": safe_tokens_per_second(
                                result.generated_tokens,
                                result.eval_duration_s or result.latency_s,
                            ),
                            "rss_before_mb": before.rss_mb,
                            "rss_after_mb": after.rss_mb,
                            "system_used_before_mb": before.system_used_mb,
                            "system_used_after_mb": after.system_used_mb,
                            "output": result.output,
                        }
                    )
                    handle.flush()

    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
