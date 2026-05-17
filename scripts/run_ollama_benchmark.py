from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eip.io_utils import ensure_parent, read_json, read_jsonl
from eip.metrics import safe_tokens_per_second
from eip.ollama_client import OllamaClient
from eip.system_stats import current_memory


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--out", required=True)
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
        "model",
        "prompt_id",
        "length_bucket",
        "task",
        "run",
        "latency_s",
        "first_token_latency_s",
        "generated_tokens",
        "prompt_tokens",
        "tokens_per_second",
        "eval_duration_s",
        "prompt_eval_duration_s",
        "rss_before_mb",
        "rss_after_mb",
        "system_used_before_mb",
        "system_used_after_mb",
        "output",
    ]

    with Path(args.out).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        total = len(experiments) * len(prompts) * runs_per_prompt
        for exp in tqdm(experiments, total=len(experiments), desc="experiments"):
            options = dict(base_options)
            options.update(exp.get("options", {}))
            for prompt in prompts:
                for run_idx in range(runs_per_prompt):
                    before = current_memory()
                    result = client.generate_streaming(exp["model"], prompt["prompt"], options)
                    after = current_memory()
                    writer.writerow(
                        {
                            "experiment": exp["name"],
                            "technique": exp.get("technique", "unknown"),
                            "model": exp["model"],
                            "prompt_id": prompt["id"],
                            "length_bucket": prompt["length_bucket"],
                            "task": prompt.get("task", ""),
                            "run": run_idx + 1,
                            "latency_s": result.latency_s,
                            "first_token_latency_s": result.first_token_latency_s,
                            "generated_tokens": result.generated_tokens,
                            "prompt_tokens": result.prompt_tokens,
                            "tokens_per_second": safe_tokens_per_second(result.generated_tokens, result.eval_duration_s or result.latency_s),
                            "eval_duration_s": result.eval_duration_s,
                            "prompt_eval_duration_s": result.prompt_eval_duration_s,
                            "rss_before_mb": before.rss_mb,
                            "rss_after_mb": after.rss_mb,
                            "system_used_before_mb": before.system_used_mb,
                            "system_used_after_mb": after.system_used_mb,
                            "output": result.output,
                        }
                    )
                    handle.flush()

    print(f"Wrote {args.out} ({total} planned runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

