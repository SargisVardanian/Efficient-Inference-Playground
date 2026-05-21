from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eip.io_utils import ensure_parent


DEFAULT_INPUTS = [
    "results/raw/hf_baseline_all_clean.csv",
    "results/raw/ollama_quantized_short.csv",
    "results/raw/ollama_quantized_medium.csv",
    "results/raw/ollama_quantized_long.csv",
    "results/raw/hf_kv_short.csv",
    "results/raw/hf_kv_medium.csv",
    "results/raw/hf_kv_long.csv",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--inputs", nargs="*", default=DEFAULT_INPUTS)
    args = parser.parse_args()

    frames = []
    for item in args.inputs:
        path = Path(item)
        if not path.exists() or path.stat().st_size == 0:
            continue
        df = pd.read_csv(path)
        if len(df) == 0:
            continue
        frames.append(df)

    if not frames:
        raise SystemExit("No input result files were found.")

    merged = pd.concat(frames, ignore_index=True)
    merged = merged[merged["experiment"].notna() & merged["prompt_id"].notna()].copy()
    merged = merged.drop_duplicates(["experiment", "prompt_id", "run"], keep="last")
    merged = merged.sort_values(["experiment", "length_bucket", "prompt_id", "run"]).reset_index(drop=True)

    ensure_parent(args.out)
    merged.to_csv(args.out, index=False)
    print(f"Wrote {args.out} ({len(merged)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
