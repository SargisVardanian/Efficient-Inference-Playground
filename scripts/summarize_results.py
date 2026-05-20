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

    group_cols = ["experiment", "technique", "length_bucket", "task", "dataset"]
    summary = (
        df.groupby(group_cols, dropna=False)
        .agg(
            prompts=("prompt_id", "nunique"),
            runs=("run", "count"),
            latency_s_mean=("latency_s", "mean"),
            latency_s_median=("latency_s", "median"),
            tokens_per_second_mean=("tokens_per_second", "mean"),
            prompt_tokens_mean=("prompt_tokens", "mean"),
            generated_tokens_mean=("generated_tokens", "mean"),
            rss_after_mb_mean=("rss_after_mb", "mean"),
            system_used_after_mb_mean=("system_used_after_mb", "mean"),
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
