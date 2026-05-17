from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from rouge_score import rouge_scorer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eip.io_utils import ensure_parent
from eip.metrics import char_similarity, exact_match


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--baseline", default="baseline_gemma4_e4b")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df["output"] = df["output"].fillna("").astype(str)
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    baseline = (
        df[df["experiment"] == args.baseline]
        .sort_values(["prompt_id", "run"])
        .groupby("prompt_id")["output"]
        .first()
        .to_dict()
    )

    rows = []
    for row in df.to_dict("records"):
        base_output = baseline.get(row["prompt_id"], "")
        rouge_l = scorer.score(base_output, row["output"])["rougeL"].fmeasure if base_output else None
        rows.append(
            {
                "experiment": row["experiment"],
                "technique": row["technique"],
                "model": row["model"],
                "prompt_id": row["prompt_id"],
                "length_bucket": row["length_bucket"],
                "run": row["run"],
                "exact_match_to_baseline": exact_match(base_output, row["output"]) if base_output else None,
                "char_similarity_to_baseline": char_similarity(base_output, row["output"]) if base_output else None,
                "rougeL_to_baseline": rouge_l,
            }
        )

    out = pd.DataFrame(rows)
    ensure_parent(args.out)
    out.to_csv(args.out, index=False)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
