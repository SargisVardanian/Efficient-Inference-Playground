from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eip.plotting import save_barplot


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--quality", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    perf = pd.read_csv(args.input)
    quality = pd.read_csv(args.quality)

    perf_summary = (
        perf.groupby(["experiment", "technique", "length_bucket"], as_index=False)
        .agg(
            latency_s=("latency_s", "median"),
            first_token_latency_s=("first_token_latency_s", "median"),
            tokens_per_second=("tokens_per_second", "median"),
            prompt_tokens=("prompt_tokens", "median"),
            generated_tokens=("generated_tokens", "median"),
        )
    )
    quality_summary = (
        quality.groupby(["experiment", "technique", "length_bucket"], as_index=False)
        .agg(
            rougeL_to_baseline=("rougeL_to_baseline", "mean"),
            char_similarity_to_baseline=("char_similarity_to_baseline", "mean"),
            exact_match_to_baseline=("exact_match_to_baseline", "mean"),
        )
    )
    joined = perf_summary.merge(quality_summary, on=["experiment", "technique", "length_bucket"], how="left")
    joined.to_csv(outdir.parent / "processed" / "summary.csv", index=False)

    save_barplot(joined, "length_bucket", "latency_s", "experiment", "Median Latency by Prompt Length", outdir / "latency_by_length.png")
    save_barplot(joined, "length_bucket", "tokens_per_second", "experiment", "Median Throughput by Prompt Length", outdir / "throughput_by_length.png")
    save_barplot(joined, "length_bucket", "rougeL_to_baseline", "experiment", "ROUGE-L Similarity to Baseline", outdir / "rouge_by_length.png")
    print(f"Wrote plots to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

