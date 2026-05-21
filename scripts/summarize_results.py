from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eip.io_utils import ensure_parent


def infer_device_runtime_note(row: pd.Series) -> str:
    existing = str(row.get("device_runtime_note", "") or "").strip()
    cache_policy = str(row.get("cache_policy", "") or "")
    if cache_policy == "recency_window" and (
        existing == ""
        or existing.startswith("Transformers default cache on ")
        or existing == "Transformers default cache on Apple Silicon MPS."
    ):
        existing = ""
    if existing:
        return existing

    backend = str(row.get("backend", "") or "")
    technique = str(row.get("technique", "") or "")
    quant_method = str(row.get("quantization_method", "") or "")
    length_bucket = str(row.get("length_bucket", "") or "")

    if backend == "ollama":
        return "Ollama GGUF quantized fallback on Apple Silicon; report separately from HF device-path rows."
    if backend == "transformers" and quant_method == "none" and cache_policy == "default":
        if length_bucket == "long":
            return "CPU fallback for long prompts on Apple Silicon; compare only with rows that used the same device policy."
        return "Transformers default cache on Apple Silicon MPS."
    if backend == "transformers" and cache_policy == "attention_sink":
        if length_bucket == "long":
            return "Transformers attention-sink cache with CPU fallback for long prompts if MPS cannot fit them."
        return "Transformers attention-sink cache on Apple Silicon MPS."
    if backend == "transformers" and cache_policy == "recency_window":
        if length_bucket == "long":
            return "Transformers hybrid recency-window KV cache with CPU fallback for long prompts if MPS cannot fit them."
        return "Transformers hybrid recency-window KV cache on Apple Silicon MPS."
    if backend == "transformers" and technique == "quantization":
        return "Transformers quantization row; runtime note should be verified per backend."
    return ""


def build_compat_summary(summary: pd.DataFrame) -> pd.DataFrame:
    compat = summary.copy()
    compat["latency_s"] = compat["latency_s_p50"]
    compat["first_token_latency_s"] = None
    compat["per_token_latency_s"] = compat["per_token_latency_s_p50"]
    compat["tokens_per_second"] = compat["tokens_per_second_mean"]
    compat["prompt_tokens"] = compat["prompt_tokens_mean"]
    compat["generated_tokens"] = compat["generated_tokens_mean"]
    compat["reference_exact_match"] = compat["reference_exact_match_mean"]
    compat["reference_contains_match"] = compat["reference_contains_match_mean"]
    compat["rougeL_to_baseline"] = None
    compat["char_similarity_to_baseline"] = None
    compat["exact_match_to_baseline"] = None
    ordered = [
        "experiment",
        "technique",
        "backend",
        "quantization_method",
        "cache_policy",
        "length_bucket",
        "task",
        "dataset",
        "prompts",
        "runs",
        "latency_s",
        "latency_s_p99",
        "first_token_latency_s",
        "per_token_latency_s",
        "per_token_latency_s_p99",
        "tokens_per_second",
        "prompt_tokens",
        "generated_tokens",
        "rss_after_mb_peak",
        "system_used_after_mb_peak",
        "reference_exact_match",
        "reference_contains_match",
        "rougeL_to_baseline",
        "char_similarity_to_baseline",
        "exact_match_to_baseline",
    ]
    return compat[ordered]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--quality")
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--out_md", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    for column in ["task", "dataset"]:
        if column not in df.columns:
            df[column] = ""
    if args.quality:
        quality = pd.read_csv(args.quality)
        df = df.merge(
            quality[
                [
                    "experiment",
                    "prompt_id",
                    "run",
                    "task",
                    "dataset",
                    "reference_exact_match",
                    "reference_contains_match",
                ]
            ],
            on=["experiment", "prompt_id", "run"],
            how="left",
            suffixes=("", "_quality"),
        )
        for column in ["task", "dataset"]:
            quality_column = f"{column}_quality"
            if quality_column in df.columns:
                df[column] = df[column].where(df[column].fillna("").astype(str) != "", df[quality_column])
    else:
        df["reference_exact_match"] = None
        df["reference_contains_match"] = None

    df["per_token_latency_s"] = df["latency_s"] / df["generated_tokens"].where(df["generated_tokens"] > 0)
    df["system_used_delta_mb"] = df["system_used_after_mb"] - df["system_used_before_mb"]
    df["rss_delta_mb"] = df["rss_after_mb"] - df["rss_before_mb"]
    for column, default in [
        ("backend", "unknown"),
        ("quantization_method", "none"),
        ("cache_policy", "default"),
        ("device_runtime_note", ""),
    ]:
        if column not in df.columns:
            df[column] = default
    df["device_runtime_note"] = df.apply(infer_device_runtime_note, axis=1)

    group_cols = [
        "experiment",
        "technique",
        "backend",
        "quantization_method",
        "cache_policy",
        "length_bucket",
        "task",
        "dataset",
    ]
    summary = (
        df.groupby(group_cols, dropna=False)
        .agg(
            prompts=("prompt_id", "nunique"),
            runs=("run", "count"),
            latency_s_mean=("latency_s", "mean"),
            latency_s_p50=("latency_s", "median"),
            latency_s_p99=("latency_s", lambda values: values.quantile(0.99)),
            per_token_latency_s_p50=("per_token_latency_s", "median"),
            per_token_latency_s_p99=("per_token_latency_s", lambda values: values.quantile(0.99)),
            tokens_per_second_mean=("tokens_per_second", "mean"),
            prompt_tokens_mean=("prompt_tokens", "mean"),
            generated_tokens_mean=("generated_tokens", "mean"),
            rss_after_mb_mean=("rss_after_mb", "mean"),
            rss_after_mb_peak=("rss_after_mb", "max"),
            rss_delta_mb_peak=("rss_delta_mb", "max"),
            system_used_after_mb_mean=("system_used_after_mb", "mean"),
            system_used_after_mb_peak=("system_used_after_mb", "max"),
            system_used_delta_mb_peak=("system_used_delta_mb", "max"),
            reference_exact_match_mean=("reference_exact_match", "mean"),
            reference_contains_match_mean=("reference_contains_match", "mean"),
            device_runtime_note=("device_runtime_note", "first"),
        )
        .reset_index()
        .sort_values(group_cols)
    )

    ensure_parent(args.out_csv)
    summary.to_csv(args.out_csv, index=False)
    compat_summary = build_compat_summary(summary)
    compat_out = Path(args.out_csv).with_name("summary.csv")
    compat_summary.to_csv(compat_out, index=False)

    lines = [
        "# Benchmark Summary",
        "",
        f"Source CSV: `{args.input}`",
        "",
        summary.to_markdown(index=False),
        "",
    ]
    ensure_parent(args.out_md)
    Path(args.out_md).write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.out_csv}, {compat_out}, and {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
