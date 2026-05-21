from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_INPUT_WORKBOOK = "/Users/sargisvardanyan/Downloads/benchmark_results.xlsx"
DEFAULT_OUTPUT_WORKBOOK = "results/benchmark_results.xlsx"

RAW_INPUTS = [
    "results/raw/hf_baseline_all_clean.csv",
    "results/raw/ollama_quantized_short.csv",
    "results/raw/ollama_quantized_medium.csv",
    "results/raw/ollama_quantized_long.csv",
    "results/raw/hf_kv_short.csv",
    "results/raw/hf_kv_medium.csv",
    "results/raw/hf_kv_long.csv",
]


EXPERIMENT_RENAMES = {
    "hf_baseline_gemma4_e4b": "needle_hf_baseline_gemma4_e4b",
    "ollama_quantized_gemma4_e4b_q4km": "needle_ollama_quantized_gemma4_e4b_q4km",
    "hf_kv_window_gemma4_e4b": "needle_hf_kv_window_gemma4_e4b",
}


def normalize(text: object) -> str:
    return " ".join(str(text or "").strip().lower().split())


def load_needle_runs() -> pd.DataFrame:
    frames = []
    for item in RAW_INPUTS:
        path = Path(item)
        if path.exists() and path.stat().st_size > 0:
            frames.append(pd.read_csv(path))
    if not frames:
        raise SystemExit("No Needle raw CSV files found.")

    df = pd.concat(frames, ignore_index=True)
    df = df[df["experiment"].notna() & df["prompt_id"].notna()].copy()
    df = df.drop_duplicates(["experiment", "prompt_id", "run"], keep="last")
    df["experiment"] = df["experiment"].replace(EXPERIMENT_RENAMES)
    df["hardware"] = "Apple M3 Pro"
    df["benchmark_suite"] = "needle_in_a_haystack"
    df["backend"] = df["backend"].replace({"transformers": "huggingface"})
    df = df.rename(columns={"prompt_tokens": "input_tokens"})
    for col in [
        "model",
        "task",
        "input_tokens",
        "generated_tokens",
        "latency_s",
        "tokens_per_second",
        "rss_before_mb",
        "rss_after_mb",
        "reference",
        "output",
    ]:
        if col not in df.columns:
            df[col] = None
    return df


def build_needle_quality(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in runs.to_dict("records"):
        reference = normalize(row.get("reference"))
        output = normalize(row.get("output"))
        has_reference = bool(reference)
        rows.append(
            {
                "experiment": row["experiment"],
                "technique": row["technique"],
                "backend": row["backend"],
                "model": row.get("model"),
                "prompt_id": row["prompt_id"],
                "length_bucket": row["length_bucket"],
                "task": row.get("task"),
                "run": row["run"],
                "exact_match_to_baseline": None,
                "char_similarity_to_baseline": None,
                "rougeL_to_baseline": None,
                "reference_exact_match": float(output == reference) if has_reference else None,
                "reference_contains_match": float(reference in output) if has_reference else None,
                "hardware": "Apple M3 Pro",
                "benchmark_suite": "needle_in_a_haystack",
            }
        )
    return pd.DataFrame(rows)


def build_needle_summary(runs: pd.DataFrame, quality: pd.DataFrame) -> pd.DataFrame:
    df = runs.copy()
    for col in ["latency_s", "tokens_per_second", "input_tokens", "generated_tokens"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    q = quality.copy()
    for col in ["reference_exact_match", "reference_contains_match"]:
        q[col] = pd.to_numeric(q[col], errors="coerce")

    group_cols = ["experiment", "technique", "backend", "length_bucket", "task"]
    perf = (
        df.groupby(group_cols, dropna=False)
        .agg(
            latency_p50=("latency_s", "median"),
            latency_p99=("latency_s", lambda values: values.quantile(0.99)),
            tokens_per_second=("tokens_per_second", "mean"),
            generated_tokens=("generated_tokens", "mean"),
            input_tokens=("input_tokens", "mean"),
            n_runs=("latency_s", "count"),
        )
        .reset_index()
    )
    qual = (
        q.groupby(group_cols, dropna=False)
        .agg(
            reference_exact_match=("reference_exact_match", "mean"),
            reference_contains_match=("reference_contains_match", "mean"),
        )
        .reset_index()
    )
    out = perf.merge(qual, on=group_cols, how="left")
    out["rougeL_to_baseline"] = None
    out["char_similarity_to_baseline"] = None
    out["exact_match_to_baseline"] = None
    out["hardware"] = "Apple M3 Pro"
    out["benchmark_suite"] = "needle_in_a_haystack"
    return out


def load_needle_prompts() -> pd.DataFrame:
    return pd.read_json("prompts/needle_eval_prompts.jsonl", lines=True)


def add_hardware_columns(df: pd.DataFrame, hardware: str, suite: str) -> pd.DataFrame:
    df = df.copy()
    if "hardware" not in df.columns:
        df["hardware"] = hardware
    else:
        df["hardware"] = df["hardware"].fillna(hardware)
    if "benchmark_suite" not in df.columns:
        df["benchmark_suite"] = suite
    else:
        df["benchmark_suite"] = df["benchmark_suite"].fillna(suite)
    return df


def drop_existing_needle_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mask = pd.Series(False, index=df.index)
    if "benchmark_suite" in df.columns:
        mask |= df["benchmark_suite"].eq("needle_in_a_haystack")
    if "experiment" in df.columns:
        mask |= df["experiment"].astype(str).str.startswith("needle_")
    return df.loc[~mask].copy()


def union_columns(left: pd.DataFrame, right: pd.DataFrame) -> list[str]:
    cols = list(left.columns)
    for col in right.columns:
        if col not in cols:
            cols.append(col)
    return cols


def write_workbook(path: Path, sheets: dict[str, pd.DataFrame]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

        from openpyxl.styles import Alignment, Font, PatternFill
        from openpyxl.utils import get_column_letter

        for ws in writer.book.worksheets:
            for cell in ws[1]:
                cell.font = Font(bold=True, color="FFFFFF")
                cell.fill = PatternFill("solid", fgColor="1F4E79")
                cell.alignment = Alignment(horizontal="center", wrap_text=True)
            for col in ws.columns:
                width = max((len(str(cell.value)) if cell.value is not None else 0) for cell in col)
                ws.column_dimensions[get_column_letter(col[0].column)].width = min(max(width + 3, 12), 48)
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT_WORKBOOK)
    parser.add_argument("--out", default=DEFAULT_OUTPUT_WORKBOOK)
    parser.add_argument("--downloads-out", default=DEFAULT_INPUT_WORKBOOK)
    args = parser.parse_args()

    workbook = pd.read_excel(args.input, sheet_name=None)
    for sheet in ["Summary", "Quality (per run)", "All Runs"]:
        workbook[sheet] = add_hardware_columns(
            drop_existing_needle_rows(workbook[sheet]),
            hardware="Apple M4 Max",
            suite="summarization_reasoning",
        )

    needle_runs = load_needle_runs()
    needle_quality = build_needle_quality(needle_runs)
    needle_summary = build_needle_summary(needle_runs, needle_quality)

    summary_cols = union_columns(workbook["Summary"], needle_summary)
    quality_cols = union_columns(workbook["Quality (per run)"], needle_quality)
    all_runs_cols = union_columns(workbook["All Runs"], needle_runs)

    workbook["Summary"] = pd.concat(
        [workbook["Summary"], needle_summary.reindex(columns=summary_cols)],
        ignore_index=True,
    )
    workbook["Quality (per run)"] = pd.concat(
        [workbook["Quality (per run)"], needle_quality.reindex(columns=quality_cols)],
        ignore_index=True,
    )
    workbook["All Runs"] = pd.concat(
        [
            workbook["All Runs"],
            needle_runs.reindex(columns=all_runs_cols),
        ],
        ignore_index=True,
    )

    combined_summary = workbook["Summary"]
    workbook["Latency p50 (s)"] = (
        combined_summary.pivot_table(index="experiment", columns="length_bucket", values="latency_p50", aggfunc="median")
        .reset_index()
    )
    workbook["Throughput (tok-s)"] = (
        combined_summary.pivot_table(index="experiment", columns="length_bucket", values="tokens_per_second", aggfunc="median")
        .reset_index()
    )
    rouge = workbook.get("ROUGE-L vs baseline", pd.DataFrame()).copy()
    if len(rouge):
        rouge = drop_existing_needle_rows(rouge)
    needle_rouge = pd.DataFrame(
        {
            "experiment": sorted(needle_summary["experiment"].unique()),
            "long": None,
            "medium": None,
            "short": None,
        }
    )
    workbook["ROUGE-L vs baseline"] = pd.concat([rouge, needle_rouge], ignore_index=True)
    workbook["Needle Prompts"] = load_needle_prompts()
    workbook["Hardware"] = pd.DataFrame(
        [
            {
                "hardware": "Apple M4 Max",
                "benchmark_suite": "summarization_reasoning",
                "applies_to": "original workbook rows",
            },
            {
                "hardware": "Apple M3 Pro",
                "benchmark_suite": "needle_in_a_haystack",
                "applies_to": "Needle all-variants rows and Needle Prompts sheet",
            },
        ]
    )

    ordered = [
        "Summary",
        "Latency p50 (s)",
        "Throughput (tok-s)",
        "ROUGE-L vs baseline",
        "Quality (per run)",
        "All Runs",
        "Needle Prompts",
        "Hardware",
    ]
    sheets = {name: workbook[name] for name in ordered}
    write_workbook(Path(args.out), sheets)
    if args.downloads_out:
        write_workbook(Path(args.downloads_out), sheets)
    print(f"Wrote {args.out}")
    print(f"Wrote {args.downloads_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
