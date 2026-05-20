from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eip.io_utils import ensure_parent


def normalize(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--prompts")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if args.prompts:
        import json

        prompt_rows = []
        with Path(args.prompts).open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    item = json.loads(line)
                    prompt_rows.append(
                        {
                            "prompt_id": item["id"],
                            "prompt_reference": item.get("reference", ""),
                            "prompt_dataset": item.get("dataset", ""),
                            "prompt_task": item.get("task", ""),
                        }
                    )
        prompts = pd.DataFrame(prompt_rows)
        df = df.merge(prompts, on="prompt_id", how="left")
    rows = []
    for row in df.to_dict("records"):
        reference = normalize(row.get("reference", "") or row.get("prompt_reference", ""))
        output = normalize(row.get("output", ""))
        has_reference = bool(reference)
        rows.append(
            {
                "experiment": row["experiment"],
                "technique": row["technique"],
                "prompt_id": row["prompt_id"],
                "length_bucket": row["length_bucket"],
                "task": row.get("task", "") or row.get("prompt_task", ""),
                "dataset": row.get("dataset", "") or row.get("prompt_dataset", ""),
                "run": row["run"],
                "reference_exact_match": float(output == reference) if has_reference else None,
                "reference_contains_match": float(reference in output) if has_reference else None,
            }
        )

    out = pd.DataFrame(rows)
    ensure_parent(args.out)
    out.to_csv(args.out, index=False)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
