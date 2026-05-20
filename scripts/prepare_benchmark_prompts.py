from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path


SEED = 42
DEFAULT_COUNT = 10


def approximate_tokens(text: str) -> int:
    return max(1, int(len(text.split()) / 0.75))


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    words = text.split()
    max_words = max(1, int(max_tokens * 0.75))
    return " ".join(words[:max_words])


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


@dataclass(frozen=True)
class NeedleSpec:
    prefix: str
    question: str
    answer: str


NEEDLE_SPECS = [
    NeedleSpec("The secret launch code is ALPHA-7.", "What is the secret launch code?", "ALPHA-7"),
    NeedleSpec("The archive room access number is 4821.", "What is the archive room access number?", "4821"),
    NeedleSpec("The meeting is scheduled for March 15th at 3pm.", "When is the meeting scheduled?", "March 15th at 3pm"),
    NeedleSpec("The winner of the robotics competition was Sarah Johnson.", "Who won the robotics competition?", "Sarah Johnson"),
    NeedleSpec("The backup encryption passphrase is cobalt river.", "What is the backup encryption passphrase?", "cobalt river"),
    NeedleSpec("The sample labeled Delta-19 is the contaminated one.", "Which sample is contaminated?", "Delta-19"),
    NeedleSpec("The package should be delivered to 14 Cedar Avenue.", "Where should the package be delivered?", "14 Cedar Avenue"),
    NeedleSpec("The telescope calibration offset is minus 2.4 degrees.", "What is the telescope calibration offset?", "minus 2.4 degrees"),
    NeedleSpec("The emergency contact is Dr. Elena Markarian.", "Who is the emergency contact?", "Dr. Elena Markarian"),
    NeedleSpec("The prototype battery lasted 17.5 hours on the final test.", "How long did the prototype battery last on the final test?", "17.5 hours"),
]

FILLER_SENTENCES = [
    "Large language model evaluation requires controlling both prompt length and answer format.",
    "Inference benchmarks often trade off latency, throughput, memory footprint, and answer quality.",
    "Local deployment setups can behave differently depending on cache policy, backend, and quantization method.",
    "A reproducible benchmark should log prompt metadata, decoding settings, and environment assumptions.",
    "Synthetic retrieval tasks are useful because the reference answer is exact and easy to score.",
    "Reasoning prompts can be padded with neutral context to isolate long-context effects from task difficulty.",
    "Summarization quality should be compared with a fixed reference output or a stable baseline model.",
    "When cache memory grows with context length, runtime policy changes can become as important as weight size.",
    "Structured CSV outputs make it easier to aggregate results across prompt families and length buckets.",
    "The benchmark should stay deterministic whenever the goal is fair comparison instead of best-case sampling.",
]

NEEDLE_BUCKET_TOKENS = {
    "short": 256,
    "medium": 2048,
    "long": 8192,
}


def build_filler(min_tokens: int, rng: random.Random) -> str:
    sentences: list[str] = []
    while approximate_tokens(" ".join(sentences)) < min_tokens:
        sentence = rng.choice(FILLER_SENTENCES)
        sentences.append(sentence)
    return " ".join(sentences)


def make_needle_prompts(count_per_bucket: int = DEFAULT_COUNT, seed: int = SEED) -> list[dict]:
    rng = random.Random(seed)
    prompts: list[dict] = []
    specs = NEEDLE_SPECS[:count_per_bucket]
    if len(specs) < count_per_bucket:
        raise ValueError(f"Requested {count_per_bucket} needle prompts but only {len(NEEDLE_SPECS)} are defined.")

    for bucket_name, target_tokens in NEEDLE_BUCKET_TOKENS.items():
        filler = build_filler(max(64, target_tokens - 64), rng)
        for index, spec in enumerate(specs, start=1):
            words = filler.split()
            insert_pos = (len(words) * (index + 1)) // (len(specs) + 2)
            words.insert(insert_pos, spec.prefix)
            context = truncate_to_tokens(" ".join(words), target_tokens)
            prompts.append(
                {
                    "id": f"needle_{bucket_name}_{index:02d}",
                    "length_bucket": bucket_name,
                    "task": "long_context_retrieval",
                    "dataset": "needle_in_a_haystack",
                    "reference": spec.answer,
                    "prompt": (
                        "Read the full context carefully. The answer appears exactly once in the text. "
                        "Reply with only the exact answer.\n\n"
                        f"{context}\n\nQuestion: {spec.question}"
                    ),
                }
            )
    return prompts


def load_dataset_rows(dataset_name: str, config_name: str | None, split: str):
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise SystemExit("The 'datasets' package is required for full prompt generation. Install requirements first.") from exc
    return load_dataset(dataset_name, config_name, split=split)


def select_by_length(rows, text_key: str, buckets: dict[str, tuple[int, int]], count: int) -> dict[str, list[dict]]:
    selected = {name: [] for name in buckets}
    for row in rows:
        text = row[text_key]
        tokens = approximate_tokens(text)
        for bucket_name, (lower, upper) in buckets.items():
            if lower <= tokens <= upper and len(selected[bucket_name]) < count:
                selected[bucket_name].append(row)
        if all(len(items) >= count for items in selected.values()):
            break
    return selected


def build_full_prompt_set(count_per_bucket: int = DEFAULT_COUNT) -> list[dict]:
    prompts: list[dict] = []

    cnn_rows = load_dataset_rows("abisee/cnn_dailymail", "3.0.0", "test")
    cnn_selected = select_by_length(
        cnn_rows,
        text_key="article",
        buckets={"short": (100, 220), "medium": (700, 950)},
        count=count_per_bucket,
    )
    for bucket_name, rows in cnn_selected.items():
        for index, row in enumerate(rows, start=1):
            prompts.append(
                {
                    "id": f"cnn_{bucket_name}_{index:02d}",
                    "length_bucket": bucket_name,
                    "task": "summarization",
                    "dataset": "cnn_dailymail",
                    "reference": row["highlights"],
                    "prompt": f"Summarize this news article in 2-3 sentences.\n\n{row['article']}",
                }
            )

    gov_rows = load_dataset_rows("ccdv/govreport-summarization", "document", "test")
    gov_selected = select_by_length(
        gov_rows,
        text_key="report",
        buckets={"long": (4000, 12000)},
        count=count_per_bucket,
    )
    for index, row in enumerate(gov_selected["long"], start=1):
        prompts.append(
            {
                "id": f"govreport_long_{index:02d}",
                "length_bucket": "long",
                "task": "summarization",
                "dataset": "govreport",
                "reference": row["summary"],
                "prompt": f"Summarize this government report in 4-6 sentences.\n\n{truncate_to_tokens(row['report'], 8192)}",
            }
        )

    gsm_rows = load_dataset_rows("openai/gsm8k", "main", "test")
    gsm_short = []
    for row in gsm_rows:
        if len(gsm_short) >= count_per_bucket:
            break
        gsm_short.append(row)

    for index, row in enumerate(gsm_short, start=1):
        answer = row["answer"].split("####")[-1].strip()
        base_prompt = f"Solve the math problem step by step and end with only the final numeric answer.\n\n{row['question']}"
        prompts.append(
            {
                "id": f"gsm8k_short_{index:02d}",
                "length_bucket": "short",
                "task": "reasoning",
                "dataset": "gsm8k",
                "reference": answer,
                "prompt": base_prompt,
            }
        )
        medium_padding = truncate_to_tokens(build_filler(1400, random.Random(SEED + index)), 1400)
        long_padding = truncate_to_tokens(build_filler(7000, random.Random(SEED + 100 + index)), 7000)
        prompts.append(
            {
                "id": f"gsm8k_medium_{index:02d}",
                "length_bucket": "medium",
                "task": "reasoning",
                "dataset": "gsm8k_padded",
                "reference": answer,
                "prompt": (
                    "The following background notes are unrelated and may be ignored unless needed.\n\n"
                    f"{medium_padding}\n\nNow solve the math problem step by step and end with only the final numeric answer.\n\n{row['question']}"
                ),
            }
        )
        prompts.append(
            {
                "id": f"gsm8k_long_{index:02d}",
                "length_bucket": "long",
                "task": "reasoning",
                "dataset": "gsm8k_padded",
                "reference": answer,
                "prompt": (
                    "The following background notes are unrelated and may be ignored unless needed.\n\n"
                    f"{long_padding}\n\nNow solve the math problem step by step and end with only the final numeric answer.\n\n{row['question']}"
                ),
            }
        )

    prompts.extend(make_needle_prompts(count_per_bucket=count_per_bucket, seed=SEED))
    return prompts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["full", "needle"], default="full")
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = (
        build_full_prompt_set(count_per_bucket=args.count)
        if args.mode == "full"
        else make_needle_prompts(count_per_bucket=args.count, seed=SEED)
    )
    write_jsonl(args.out, rows)
    print(f"Wrote {args.out} with {len(rows)} prompts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
