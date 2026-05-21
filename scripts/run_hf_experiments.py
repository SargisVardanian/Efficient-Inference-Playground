from __future__ import annotations

import argparse
import csv
import gc
import os
import re
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

import torch
import transformers.modeling_utils
transformers.modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None

from transformers import AutoModelForImageTextToText, AutoProcessor, MetalConfig

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eip.io_utils import ensure_parent, read_json, read_jsonl
from eip.metrics import safe_tokens_per_second
from eip.system_stats import current_memory
from eip.cache_policies import AttentionSinkCache, HybridRecencyWindowCache


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


def load_model(model_id: str, device: str, quantization: dict | None, local_files_only: bool):
    kwargs = {"trust_remote_code": False, "local_files_only": local_files_only}
    if quantization and quantization.get("method") == "metal":
        kwargs["quantization_config"] = MetalConfig(bits=int(quantization.get("bits", 4)))
        kwargs["device_map"] = device
    else:
        kwargs["dtype"] = torch.bfloat16
    model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
    if not quantization or quantization.get("method") != "metal":
        model = model.to(device)
    return model.eval()


def validate_quantization_runtime(quantization: dict | None) -> None:
    if not quantization or quantization.get("method") != "metal":
        return
    match = re.match(r"^(\d+)\.(\d+)", torch.__version__)
    torch_tag = f"torch{match.group(1)}{match.group(2)}" if match else torch.__version__
    supported_tags = {"torch28", "torch29", "torch210"}
    if torch_tag not in supported_tags:
        supported = ", ".join(sorted(supported_tags))
        raise SystemExit(
            "Metal INT4 quantization requires a Hugging Face Metal kernel build matching the installed torch version. "
            f"Current torch is {torch.__version__} ({torch_tag}), but available kernel builds are {supported}. "
            "Use a separate INT4 environment with torch==2.10.0 or torch==2.9.x."
        )


def should_force_cpu_for_prompt(base_device: str, quantization: dict | None, prompt_tokens: int, threshold: int) -> bool:
    return base_device == "mps" and not quantization and prompt_tokens >= threshold


def should_load_experiment_on_cpu(
    base_device: str,
    quantization: dict | None,
    prompts: list[dict],
    completed_prompt_ids: set[str],
) -> bool:
    if base_device != "mps" or quantization:
        return False
    pending_prompts = [prompt for prompt in prompts if prompt["id"] not in completed_prompt_ids]
    return bool(pending_prompts) and all(prompt.get("length_bucket") == "long" for prompt in pending_prompts)


def is_mps_oom_error(exc: RuntimeError) -> bool:
    text = str(exc)
    return "MPS backend out of memory" in text or "kIOGPUCommandBufferCallbackErrorOutOfMemory" in text


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


def build_cache_for_prompt(model, exp: dict):
    cache_policy = exp.get("cache_policy") or {}
    cache_type = cache_policy.get("type")
    if cache_type == "attention_sink":
        return AttentionSinkCache(
            config=model.config,
            window_length=int(cache_policy["window_length"]),
            num_sink_tokens=int(cache_policy.get("num_sink_tokens", 4)),
        )
    if cache_type == "recency_window":
        return HybridRecencyWindowCache(
            config=model.config,
            window_length=int(cache_policy["window_length"]),
        )
    return None


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
    parser.add_argument("--device")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-hub", action="store_true")
    args = parser.parse_args()

    config = read_json(args.config)
    prompts = read_jsonl(args.prompts)
    device = resolve_device(args.device or config.get("device"))
    local_files_only = not args.allow_hub
    cpu_fallback_prompt_tokens = int(config.get("cpu_fallback_prompt_tokens", 4096))
    processor = AutoProcessor.from_pretrained(
        config["model_id"],
        trust_remote_code=False,
        local_files_only=local_files_only,
    )
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
    if auto_resume and not args.resume:
        print(f"Detected existing output at {out_path}; auto-resuming completed prompts.", flush=True)
    completed_by_experiment = {
        exp["name"]: read_completed_prompt_ids(out_path, exp["name"]) if resume_mode else set()
        for exp in config["experiments"]
    }
    open_mode = "a" if resume_mode and out_path.exists() else "w"
    write_header = not (resume_mode and out_path.exists() and out_path.stat().st_size > 0)

    with out_path.open(open_mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if write_header:
            writer.writeheader()
            handle.flush()

        for exp in config["experiments"]:
            if not exp.get("enabled", True):
                continue
            clear_memory()
            quantization = exp.get("quantization")
            validate_quantization_runtime(quantization)
            completed_prompt_ids = completed_by_experiment.get(exp["name"], set())
            experiment_device = "cpu" if should_load_experiment_on_cpu(device, quantization, prompts, completed_prompt_ids) else device
            if experiment_device == "cpu" and device == "mps":
                print(
                    f"All pending prompts for {exp['name']} are long; loading one CPU bfloat16 model instead of MPS+CPU fallback.",
                    flush=True,
                )
            model = load_model(config["model_id"], experiment_device, quantization, local_files_only)
            cpu_model = None
            for prompt in prompts:
                if prompt["id"] in completed_prompt_ids:
                    print(f"Skipping completed prompt {prompt['id']} for {exp['name']}.", flush=True)
                    continue
                encoded = encode_prompt(processor, prompt["prompt"], experiment_device)
                prompt_tokens = int(encoded["input_ids"].shape[-1])
                current_device = (
                    "cpu"
                    if should_force_cpu_for_prompt(experiment_device, quantization, prompt_tokens, cpu_fallback_prompt_tokens)
                    else experiment_device
                )
                if current_device == "cpu":
                    del encoded
                    clear_memory()
                    if experiment_device == "cpu":
                        active_model = model
                    elif cpu_model is None:
                        print(
                            f"Prompt {prompt['id']} has {prompt_tokens} tokens; loading CPU fallback model for long-context execution.",
                            flush=True,
                        )
                        del model
                        model = None
                        clear_memory()
                        cpu_model = load_model(config["model_id"], "cpu", None, local_files_only)
                        active_model = cpu_model
                    else:
                        active_model = cpu_model
                    encoded = encode_prompt(processor, prompt["prompt"], "cpu")
                else:
                    active_model = model
                for run_idx in range(int(config.get("runs_per_prompt", 1))):
                    before = current_memory()
                    started = time.perf_counter()
                    prompt_cache = build_cache_for_prompt(active_model, exp)
                    with torch.no_grad():
                        try:
                            output_ids = active_model.generate(
                                **encoded,
                                **generation_kwargs(exp),
                                **({"past_key_values": prompt_cache} if prompt_cache is not None else {}),
                            )
                        except RuntimeError as exc:
                            if current_device == "mps" and not quantization and is_mps_oom_error(exc):
                                print(
                                    f"MPS OOM on prompt {prompt['id']} ({prompt_tokens} tokens); retrying on CPU.",
                                    flush=True,
                                )
                                clear_memory()
                                if cpu_model is None:
                                    cpu_model = load_model(config["model_id"], "cpu", None, local_files_only)
                                del encoded
                                encoded = encode_prompt(processor, prompt["prompt"], "cpu")
                                active_model = cpu_model
                                current_device = "cpu"
                                prompt_cache = build_cache_for_prompt(active_model, exp)
                                output_ids = active_model.generate(
                                    **encoded,
                                    **generation_kwargs(exp),
                                    **({"past_key_values": prompt_cache} if prompt_cache is not None else {}),
                                )
                            else:
                                raise
                    latency_s = time.perf_counter() - started
                    after = current_memory()
                    prompt_tokens = int(encoded["input_ids"].shape[-1])
                    generated_tokens = int(output_ids.shape[-1] - prompt_tokens)
                    output = processor.decode(output_ids[0][prompt_tokens:], skip_special_tokens=True)
                    cache_policy = exp.get("cache_policy") or {}
                    quantization_row = exp.get("quantization") or {}
                    writer.writerow(
                        {
                            "experiment": exp["name"],
                            "technique": exp["technique"],
                            "backend": "transformers",
                            "model": config["model_id"],
                            "quantization_method": quantization_row.get("method", "none"),
                            "quantization_bits": quantization_row.get("bits"),
                            "cache_policy": cache_policy.get("type", "default"),
                            "cache_window_length": cache_policy.get("window_length"),
                            "device_runtime_note": (
                                "CPU fallback for long prompts on Apple Silicon; short/medium used Transformers on MPS."
                                if current_device == "cpu" and device == "mps" and not quantization
                                else (
                                    "Transformers attention-sink cache on Apple Silicon MPS."
                                    if cache_policy.get("type") == "attention_sink"
                                    else (
                                        "Transformers hybrid recency-window KV cache on Apple Silicon MPS."
                                        if cache_policy.get("type") == "recency_window"
                                        else f"Transformers default cache on {current_device}."
                                    )
                                )
                            ),
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
                    del output_ids
                    clear_memory()
                del encoded
                clear_memory()
            del model
            if cpu_model is not None:
                del cpu_model
            clear_memory()

    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
