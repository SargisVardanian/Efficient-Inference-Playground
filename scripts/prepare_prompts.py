from __future__ import annotations
import json
import random
from pathlib import Path
from datasets import load_dataset

OUTPUT_FILE = "prompts/eval_prompts.jsonl"
N_PER_BUCKET = 5
SEED = 42
random.seed(SEED)

def count_tokens(text: str) -> int:
    return int(len(text.split()) / 0.75)

def truncate_to_tokens(text: str, max_tokens: int) -> str:
    words = text.split()
    max_words = int(max_tokens * 0.75)
    return " ".join(words[:max_words])

def load_cnn_prompts() -> list[dict]:
    print("Loading CNN/DailyMail...")
    ds = load_dataset("cnn_dailymail", "3.0.0", split="test")

    buckets = {
        "short":  {"min": 100,  "max": 200,  "prompts": []},
        "medium": {"min": 700,  "max": 900,  "prompts": []},
        "long":   {"min": 5000, "max": 9000, "prompts": []},
    }

    for item in ds:
        article = item["article"]
        reference = item["highlights"]
        tokens = count_tokens(article)

        for bucket_name, bucket in buckets.items():
            if (bucket["min"] <= tokens <= bucket["max"] and
                    len(bucket["prompts"]) < N_PER_BUCKET):

                if bucket_name == "long":
                    article = truncate_to_tokens(article, 8000)

                bucket["prompts"].append({
                    "id": f"cnn_{bucket_name}_{len(bucket['prompts'])+1}",
                    "length_bucket": bucket_name,
                    "task": "summarization",
                    "reference": reference,
                    "prompt": f"Summarize this news article in 2-3 sentences:\n\n{article}"
                })

        if all(len(b["prompts"]) == N_PER_BUCKET for b in buckets.values()):
            break

    all_prompts = []
    for name, bucket in buckets.items():
        print(f"  CNN {name}: {len(bucket['prompts'])} prompts")
        all_prompts.extend(bucket["prompts"])

    return all_prompts


def load_gsm8k_prompts() -> list[dict]:
    print("Loading GSM8K...")
    ds = load_dataset("gsm8k", "main", split="test")

    prompts = []
    for i, item in enumerate(ds):
        if len(prompts) >= N_PER_BUCKET:
            break

        question = item["question"]
        answer = item["answer"].split("####")[-1].strip()

        prompts.append({
            "id": f"gsm8k_short_{i+1}",
            "length_bucket": "short",
            "task": "reasoning",
            "reference": answer,
            "prompt": f"Solve this math problem step by step, then give the final answer:\n\n{question}"
        })

    print(f"  GSM8K short: {len(prompts)} prompts")
    return prompts


def make_needle_prompts() -> list[dict]:
    print("Generating needle-in-a-haystack prompts...")

    needles = [
        ("The secret code is ALPHA-7.", "What is the secret code?", "ALPHA-7"),
        ("The meeting is scheduled for March 15th at 3pm.", "When is the meeting scheduled?", "March 15th at 3pm"),
        ("The password to the vault is 4821.", "What is the password to the vault?", "4821"),
        ("The treasure is buried under the old oak tree.", "Where is the treasure buried?", "under the old oak tree"),
        ("The winner of the competition was Sarah Johnson.", "Who won the competition?", "Sarah Johnson"),
    ]

    filler = """
    The development of artificial intelligence has progressed rapidly over the past decade.
    Researchers have made significant advances in natural language processing, computer vision,
    and reinforcement learning. Large language models have demonstrated impressive capabilities
    across a wide range of tasks. The field continues to evolve with new architectures and
    training techniques being developed regularly. Scientists and engineers work together to
    push the boundaries of what is possible with modern AI systems.
    """ * 150  

    prompts = []
    for i, (needle, question, answer) in enumerate(needles):
        words = filler.split()
        insert_pos = len(words) // 2
        words.insert(insert_pos, needle)
        long_text = " ".join(words)

        long_text = truncate_to_tokens(long_text, 7500)

        prompts.append({
            "id": f"needle_long_{i+1}",
            "length_bucket": "long",
            "task": "long_context_retrieval",
            "reference": answer,
            "prompt": f"Read this text carefully and answer the question at the end.\n\n{long_text}\n\nQuestion: {question}"
        })

    print(f"  Needle-in-haystack long: {len(prompts)} prompts")
    return prompts


def main():
    cnn_prompts = load_cnn_prompts()
    gsm8k_prompts = load_gsm8k_prompts()
    needle_prompts = make_needle_prompts()

    all_new_prompts = cnn_prompts + gsm8k_prompts + needle_prompts

    print(f"\nTotal new prompts to add: {len(all_new_prompts)}")

    print(f"Appending to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for prompt in all_new_prompts:
            f.write(json.dumps(prompt) + "\n")

    print("\nFinal prompts file summary:")
    counts = {}
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            key = f"{p['length_bucket']} / {p['task']}"
            counts[key] = counts.get(key, 0) + 1

    for key, count in sorted(counts.items()):
        print(f"  {key}: {count}")

    print("\nDone!")

if __name__ == "__main__":
    main()