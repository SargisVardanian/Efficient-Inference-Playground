from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eip.ollama_client import OllamaClient


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemma4:e4b")
    parser.add_argument("--url", default="http://localhost:11434")
    args = parser.parse_args()

    client = OllamaClient(args.url)
    tags = client.tags()
    models = sorted(item["name"] for item in tags.get("models", []))
    print("Installed Ollama models:")
    for model in models:
        print(f"- {model}")

    if args.model not in models:
        print(f"\nMissing requested model: {args.model}", file=sys.stderr)
        return 1

    result = client.generate_streaming(args.model, "Reply with exactly: ok", {"temperature": 0, "num_predict": 8})
    print(f"\nSmoke test output: {result.output.strip()}")
    print(f"Latency: {result.latency_s:.3f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

