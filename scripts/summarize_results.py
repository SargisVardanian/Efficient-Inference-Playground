from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eip.io_utils import ensure_parent


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

    group_cols = ["experiment", "technique", "length_bucket", "task", "dataset"]
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
        )
        .reset_index()
        .sort_values(group_cols)
    )

    ensure_parent(args.out_csv)
    summary.to_csv(args.out_csv, index=False)

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
    print(f"Wrote {args.out_csv} and {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
