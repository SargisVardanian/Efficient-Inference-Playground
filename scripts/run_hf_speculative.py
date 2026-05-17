from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eip.io_utils import ensure_parent, read_jsonl
from eip.metrics import safe_tokens_per_second
from eip.system_stats import current_memory


def load_model(model_id: str, device: str):
    dtype = torch.float16 if device == "cuda" else torch.float32
    return AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(device).eval()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_model", required=True)
    parser.add_argument("--assistant_model", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    prompts = read_jsonl(args.prompts)
    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    target = load_model(args.target_model, args.device)
    assistant = load_model(args.assistant_model, args.device)

    ensure_parent(args.out)
    fields = [
        "experiment",
        "technique",
        "target_model",
        "assistant_model",
        "prompt_id",
        "length_bucket",
        "latency_s",
        "generated_tokens",
        "tokens_per_second",
        "rss_before_mb",
        "rss_after_mb",
        "output",
    ]
    with Path(args.out).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for prompt in prompts:
            encoded = tokenizer(prompt["prompt"], return_tensors="pt").to(args.device)
            before = current_memory()
            started = time.perf_counter()
            with torch.no_grad():
                output_ids = target.generate(
                    **encoded,
                    assistant_model=assistant,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )
            latency_s = time.perf_counter() - started
            after = current_memory()
            generated_tokens = int(output_ids.shape[-1] - encoded["input_ids"].shape[-1])
            output = tokenizer.decode(output_ids[0][encoded["input_ids"].shape[-1] :], skip_special_tokens=True)
            writer.writerow(
                {
                    "experiment": "hf_speculative",
                    "technique": "speculative_decoding",
                    "target_model": args.target_model,
                    "assistant_model": args.assistant_model,
                    "prompt_id": prompt["id"],
                    "length_bucket": prompt["length_bucket"],
                    "latency_s": latency_s,
                    "generated_tokens": generated_tokens,
                    "tokens_per_second": safe_tokens_per_second(generated_tokens, latency_s),
                    "rss_before_mb": before.rss_mb,
                    "rss_after_mb": after.rss_mb,
                    "output": output,
                }
            )
            handle.flush()

    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

