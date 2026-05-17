from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass
class OllamaResult:
    output: str
    latency_s: float
    first_token_latency_s: float | None
    generated_tokens: int | None
    prompt_tokens: int | None
    eval_duration_s: float | None
    prompt_eval_duration_s: float | None


class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434", timeout_s: float = 600) -> None:
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=timeout_s)

    def tags(self) -> dict[str, Any]:
        response = self.client.get(f"{self.base_url}/api/tags")
        response.raise_for_status()
        return response.json()

    def generate_streaming(self, model: str, prompt: str, options: dict[str, Any]) -> OllamaResult:
        started = time.perf_counter()
        first_token_at: float | None = None
        chunks: list[str] = []
        final_payload: dict[str, Any] = {}

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": options,
        }
        with self.client.stream("POST", f"{self.base_url}/api/generate", json=payload) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                item = json.loads(line)
                token = item.get("response", "")
                if token and first_token_at is None:
                    first_token_at = time.perf_counter()
                chunks.append(token)
                if item.get("done"):
                    final_payload = item

        ended = time.perf_counter()
        eval_count = final_payload.get("eval_count")
        prompt_eval_count = final_payload.get("prompt_eval_count")
        eval_duration_ns = final_payload.get("eval_duration")
        prompt_eval_duration_ns = final_payload.get("prompt_eval_duration")

        return OllamaResult(
            output="".join(chunks),
            latency_s=ended - started,
            first_token_latency_s=(first_token_at - started) if first_token_at else None,
            generated_tokens=eval_count,
            prompt_tokens=prompt_eval_count,
            eval_duration_s=(eval_duration_ns / 1e9) if eval_duration_ns else None,
            prompt_eval_duration_s=(prompt_eval_duration_ns / 1e9) if prompt_eval_duration_ns else None,
        )

