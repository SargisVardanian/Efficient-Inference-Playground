from __future__ import annotations
import json
import random
from pathlib import Path
from datasets import load_dataset

OUTPUT_FILE = "prompts/eval_prompts.jsonl"
N_PER_BUCKET = 10
SEED = 42
random.seed(SEED)

# Filler text used to pad GSM8K prompts to medium/long lengths
FILLER = (
    "The development of artificial intelligence has progressed rapidly over the past decade. "
    "Researchers have made significant advances in natural language processing, computer vision, "
    "and reinforcement learning. Large language models have demonstrated impressive capabilities "
    "across a wide range of tasks. The field continues to evolve with new architectures and "
    "training techniques being developed regularly. Scientists and engineers work together to "
    "push the boundaries of what is possible with modern AI systems. "
) * 300  # enough filler to pad up to ~8000 tokens


def count_tokens(text: str) -> int:
    return int(len(text.split()) / 0.75)


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    words = text.split()
    max_words = int(max_tokens * 0.75)
    return " ".join(words[:max_words])


def pad_to_tokens(text: str, target_tokens: int) -> str:
    """Pad text with filler so the total reaches approximately target_tokens."""
    current = count_tokens(text)
    if current >= target_tokens:
        return text
    needed = target_tokens - current
    filler_words = FILLER.split()[:int(needed * 0.75)]
    padding = " ".join(filler_words)
    return (
        f"[Background context — not needed to answer the question below]\n\n"
        f"{padding}\n\n"
        f"[End of background context]\n\n"
        f"{text}"
    )


def load_cnn_prompts() -> list[dict]:
    """Load short and medium summarization prompts from CNN/DailyMail."""
    print("Loading CNN/DailyMail (short + medium)...")
    ds = load_dataset("cnn_dailymail", "3.0.0", split="test")

    short_prompts: list[dict] = []
    medium_prompts: list[dict] = []

    for item in ds:
        article = item["article"]
        reference = item["highlights"]
        tokens = count_tokens(article)

        if 100 <= tokens <= 200 and len(short_prompts) < N_PER_BUCKET:
            short_prompts.append({
                "id": f"cnn_short_{len(short_prompts) + 1}",
                "length_bucket": "short",
                "task": "summarization",
                "reference": reference,
                "prompt": f"Summarize this news article in 2-3 sentences:\n\n{article}",
            })

        if 700 <= tokens <= 900 and len(medium_prompts) < N_PER_BUCKET:
            medium_prompts.append({
                "id": f"cnn_medium_{len(medium_prompts) + 1}",
                "length_bucket": "medium",
                "task": "summarization",
                "reference": reference,
                "prompt": f"Summarize this news article in 2-3 sentences:\n\n{article}",
            })

        if len(short_prompts) == N_PER_BUCKET and len(medium_prompts) == N_PER_BUCKET:
            break

    print(f"  CNN short:  {len(short_prompts)} prompts")
    print(f"  CNN medium: {len(medium_prompts)} prompts")
    return short_prompts + medium_prompts


def load_govreport_prompts() -> list[dict]:
    """Load long summarization prompts from GovReport (naturally 3k-10k token reports)."""
    print("Loading GovReport (long)...")
    ds = load_dataset("ccdv/govreport-summarization", split="test")

    long_prompts: list[dict] = []
    for item in ds:
        if len(long_prompts) >= N_PER_BUCKET:
            break
        report = item["report"]
        reference = item["summary"]
        tokens = count_tokens(report)
        if tokens < 3000:
            continue  # skip short reports
        # Truncate to 8000 tokens to keep inference feasible
        report = truncate_to_tokens(report, 8000)
        long_prompts.append({
            "id": f"govreport_long_{len(long_prompts) + 1}",
            "length_bucket": "long",
            "task": "summarization",
            "reference": reference,
            "prompt": f"Summarize this government report in 3-5 sentences:\n\n{report}",
        })

    print(f"  GovReport long: {len(long_prompts)} prompts")
    return long_prompts


def load_gsm8k_prompts() -> list[dict]:
    """Return short (raw), medium (~1 000 tokens padded), and long (~8 000 tokens padded) GSM8K prompts."""
    print("Loading GSM8K...")
    ds = load_dataset("gsm8k", "main", split="test")

    base_problems: list[dict] = []
    for i, item in enumerate(ds):
        if len(base_problems) >= N_PER_BUCKET:
            break
        question = item["question"]
        answer = item["answer"].split("####")[-1].strip()
        base_problems.append({"question": question, "answer": answer, "idx": i + 1})

    short_prompts, medium_prompts, long_prompts = [], [], []

    for p in base_problems:
        q = p["question"]
        a = p["answer"]
        idx = p["idx"]

        # Short — raw question, no padding
        short_prompts.append({
            "id": f"gsm8k_short_{idx}",
            "length_bucket": "short",
            "task": "reasoning",
            "reference": a,
            "prompt": f"Solve this math problem step by step, then give the final answer:\n\n{q}",
        })

        # Medium — pad raw question to ~1 000 tokens
        medium_text = pad_to_tokens(
            f"Solve this math problem step by step, then give the final answer:\n\n{q}",
            target_tokens=1000,
        )
        medium_prompts.append({
            "id": f"gsm8k_medium_{idx}",
            "length_bucket": "medium",
            "task": "reasoning",
            "reference": a,
            "prompt": medium_text,
        })

        # Long — pad raw question to ~8 000 tokens
        long_text = pad_to_tokens(
            f"Solve this math problem step by step, then give the final answer:\n\n{q}",
            target_tokens=8000,
        )
        long_prompts.append({
            "id": f"gsm8k_long_{idx}",
            "length_bucket": "long",
            "task": "reasoning",
            "reference": a,
            "prompt": long_text,
        })

    print(f"  GSM8K short:  {len(short_prompts)} prompts")
    print(f"  GSM8K medium: {len(medium_prompts)} prompts")
    print(f"  GSM8K long:   {len(long_prompts)} prompts")

    return short_prompts + medium_prompts + long_prompts


def main():
    cnn_prompts = load_cnn_prompts()
    govreport_prompts = load_govreport_prompts()
    gsm8k_prompts = load_gsm8k_prompts()

    all_prompts = cnn_prompts + govreport_prompts + gsm8k_prompts

    print(f"\nTotal prompts: {len(all_prompts)}")
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for prompt in all_prompts:
            f.write(json.dumps(prompt) + "\n")

    print(f"\nWrote {OUTPUT_FILE}")
    print("\nFinal summary:")
    counts: dict[str, int] = {}
    with open(OUTPUT_FILE) as f:
        for line in f:
            p = json.loads(line)
            key = f"{p['task']:20s}  {p['length_bucket']}"
            counts[key] = counts.get(key, 0) + 1
    for key, count in sorted(counts.items()):
        print(f"  {key}: {count}")
    print("\nDone!")


if __name__ == "__main__":
    main()
