from __future__ import annotations

import argparse
import csv
import gc
import time
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, MetalConfig

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eip.io_utils import ensure_parent, read_json, read_jsonl
from eip.metrics import safe_tokens_per_second
from eip.system_stats import current_memory


def clear_memory() -> None:
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def resolve_device(explicit_device: str | None) -> str:
    if explicit_device:
        return explicit_device
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(model_id: str, device: str, quantization: dict | None):
    kwargs = {"trust_remote_code": True}
    if quantization and quantization.get("method") == "metal":
        kwargs["quantization_config"] = MetalConfig(bits=int(quantization.get("bits", 4)))
        kwargs["device_map"] = device
    else:
        kwargs["dtype"] = torch.bfloat16 if device in {"cuda", "mps"} else torch.float32
    model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
    if not quantization or quantization.get("method") != "metal":
        model = model.to(device)
    return model.eval()


def generation_kwargs(exp: dict) -> dict:
    kwargs = {
        "do_sample": False,
        "max_new_tokens": int(exp.get("max_new_tokens", 128)),
        "use_cache": True,
    }
    cache_policy = exp.get("cache_policy") or {}
    if cache_policy.get("type") == "sliding_window":
        kwargs["cache_implementation"] = "sliding_window"
        kwargs["cache_config"] = {"max_cache_len": int(cache_policy["window_length"])}
    return kwargs


def encode_prompt(processor, prompt_text: str, device: str) -> dict:
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]
    encoded = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    )
    return {key: value.to(device) for key, value in encoded.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device")
    args = parser.parse_args()

    config = read_json(args.config)
    prompts = read_jsonl(args.prompts)
    device = resolve_device(args.device or config.get("device"))
    processor = AutoProcessor.from_pretrained(config["model_id"], trust_remote_code=True)
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

    with Path(args.out).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()

        for exp in config["experiments"]:
            if not exp.get("enabled", True):
                continue
            clear_memory()
            model = load_model(config["model_id"], device, exp.get("quantization"))
            for prompt in prompts:
                encoded = encode_prompt(processor, prompt["prompt"], device)
                for run_idx in range(int(config.get("runs_per_prompt", 1))):
                    before = current_memory()
                    started = time.perf_counter()
                    with torch.no_grad():
                        output_ids = model.generate(**encoded, **generation_kwargs(exp))
                    latency_s = time.perf_counter() - started
                    after = current_memory()
                    prompt_tokens = int(encoded["input_ids"].shape[-1])
                    generated_tokens = int(output_ids.shape[-1] - prompt_tokens)
                    output = processor.decode(output_ids[0][prompt_tokens:], skip_special_tokens=True)
                    cache_policy = exp.get("cache_policy") or {}
                    quantization = exp.get("quantization") or {}
                    writer.writerow(
                        {
                            "experiment": exp["name"],
                            "technique": exp["technique"],
                            "backend": "transformers",
                            "model": config["model_id"],
                            "quantization_method": quantization.get("method", "none"),
                            "quantization_bits": quantization.get("bits"),
                            "cache_policy": cache_policy.get("type", "default"),
                            "cache_window_length": cache_policy.get("window_length"),
                            "prompt_id": prompt["id"],
                            "length_bucket": prompt["length_bucket"],
                            "task": prompt["task"],
                            "dataset": prompt.get("dataset", ""),
                            "reference": prompt.get("reference", ""),
                            "run": run_idx + 1,
                            "latency_s": latency_s,
                            "generated_tokens": generated_tokens,
                            "prompt_tokens": prompt_tokens,
                            "tokens_per_second": safe_tokens_per_second(generated_tokens, latency_s),
                            "rss_before_mb": before.rss_mb,
                            "rss_after_mb": after.rss_mb,
                            "system_used_before_mb": before.system_used_mb,
                            "system_used_after_mb": after.system_used_mb,
                            "output": output,
                        }
                    )
                    handle.flush()
            del model
            clear_memory()

    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
