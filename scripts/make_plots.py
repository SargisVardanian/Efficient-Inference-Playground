from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

def save_barplot(df: pd.DataFrame, metric: str, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot = df.pivot(index="length_bucket", columns="experiment", values=metric)
    pivot.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Length bucket")
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(args.summary)

    save_barplot(summary, "latency_s_p50", "Latency p50 by Prompt Length", outdir / "latency_by_length.png")
    save_barplot(summary, "tokens_per_second_mean", "Throughput by Prompt Length", outdir / "throughput_by_length.png")
    save_barplot(summary, "reference_exact_match_mean", "Exact Match by Prompt Length", outdir / "exact_match_by_length.png")
    print(f"Wrote plots to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
