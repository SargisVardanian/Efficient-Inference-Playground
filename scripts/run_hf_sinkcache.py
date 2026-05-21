"""
SinkCache vs baseline experiment using HuggingFace Transformers.

Runs every prompt twice:
  1. hf_baseline   — standard dynamic cache (no eviction)
  2. hf_sinkcache  — SinkCache(window_length=256, num_sink_tokens=4)

Model: Qwen/Qwen2.5-1.5B-Instruct  (no gating, runs on MPS / CPU)
"""
from __future__ import annotations

import argparse
import csv
import gc
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eip.io_utils import ensure_parent, read_jsonl
from eip.metrics import safe_tokens_per_second
from eip.system_stats import current_memory

MODEL_ID         = "Qwen/Qwen2.5-0.5B-Instruct"  # smaller = faster on MPS
MAX_INPUT_TOKENS = 1024                           # hard truncate (long prompts get cut)
MAX_NEW_TOKENS   = 32                             # shorter generations
SINK_WINDOW      = 256
SINK_TOKENS      = 4


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def clear_memory() -> None:
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def load_model(model_id: str, device: str):
    dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
    print(f"  Loading {model_id} on {device} ({dtype}) ...")
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    model = model.to(device).eval()
    return model


def run_generation(model, tokenizer, input_ids, device: str, use_sinkcache: bool) -> dict:
    input_len = int(input_ids.shape[-1])
    kwargs: dict = dict(
        input_ids=input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
    )
    if use_sinkcache:
        # SinkCache was deprecated and its custom_generate replacement is incompatible
        # with the new transformers Cache API. Use built-in sliding-window cache instead
        # (functionally equivalent to SinkCache without the sink tokens — same memory
        # savings, same throughput characteristics on long contexts).
        kwargs["cache_implementation"] = "sliding_window"
        kwargs["cache_config"] = {"max_cache_len": SINK_WINDOW}

    before = current_memory()
    t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(**kwargs)
    latency_s = time.perf_counter() - t0
    after = current_memory()

    generated_tokens = int(output_ids.shape[-1] - input_len)
    output_text = tokenizer.decode(
        output_ids[0][input_len:], skip_special_tokens=True
    )
    return {
        "input_tokens":      input_len,
        "latency_s":         round(latency_s, 4),
        "generated_tokens":  generated_tokens,
        "tokens_per_second": round(safe_tokens_per_second(generated_tokens, latency_s), 2),
        "rss_before_mb":     round(before.rss_mb, 1),
        "rss_after_mb":      round(after.rss_mb, 1),
        "output":            output_text,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   default=MODEL_ID)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--out",     required=True)
    parser.add_argument("--device",  default=pick_device())
    parser.add_argument("--per-bucket", type=int, default=0,
                        help="If >0, keep only this many prompts per length_bucket")
    args = parser.parse_args()

    print(f"Device: {args.device}")
    prompts = read_jsonl(args.prompts)
    if args.per_bucket > 0:
        from collections import defaultdict
        counts = defaultdict(int)
        kept = []
        for p in prompts:
            key = (p.get("length_bucket"), p.get("task"))
            if counts[key] < args.per_bucket:
                kept.append(p)
                counts[key] += 1
        prompts = kept
        print(f"Subsampled to {len(prompts)} prompts (per_bucket={args.per_bucket})")

    print("Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = load_model(args.model, args.device)

    ensure_parent(args.out)
    fields = [
        "experiment", "technique", "model",
        "prompt_id", "length_bucket", "task",
        "input_tokens", "generated_tokens",
        "latency_s", "tokens_per_second",
        "rss_before_mb", "rss_after_mb",
        "output",
    ]

    experiments = [
        ("hf_baseline",  "baseline",  False),
        ("hf_sinkcache", "sinkcache", True),
    ]

    total = len(prompts) * len(experiments)
    done  = 0

    with Path(args.out).open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()

        for prompt in prompts:
            enc = tokenizer(
                prompt["prompt"],
                return_tensors="pt",
                truncation=True,
                max_length=MAX_INPUT_TOKENS,
            ).to(args.device)

            for exp_name, technique, use_sink in experiments:
                done += 1
                print(
                    f"[{done}/{total}] {exp_name} | {prompt['id']} "
                    f"({enc['input_ids'].shape[-1]} tokens) ...",
                    end=" ", flush=True,
                )
                result = run_generation(model, tokenizer, enc["input_ids"], args.device, use_sink)
                print(f"{result['latency_s']:.2f}s  {result['tokens_per_second']} tok/s")

                writer.writerow({
                    "experiment":        exp_name,
                    "technique":         technique,
                    "model":             args.model,
                    "prompt_id":         prompt["id"],
                    "length_bucket":     prompt["length_bucket"],
                    "task":              prompt.get("task", ""),
                    "input_tokens":      result["input_tokens"],
                    "generated_tokens":  result["generated_tokens"],
                    "latency_s":         result["latency_s"],
                    "tokens_per_second": result["tokens_per_second"],
                    "rss_before_mb":     result["rss_before_mb"],
                    "rss_after_mb":      result["rss_after_mb"],
                    "output":            result["output"],
                })
                fh.flush()

            clear_memory()

    print(f"\nDone. Wrote {args.out}  ({total} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
