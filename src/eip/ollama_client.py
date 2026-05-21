from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass
class OllamaResult:
    output: str
    latency_s: float
    generated_tokens: int
    prompt_tokens: int
    eval_duration_s: float | None
    prompt_eval_duration_s: float | None


class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434") -> None:
        self.base_url = base_url.rstrip("/")

    def generate(self, model: str, prompt: str, options: dict | None = None) -> OllamaResult:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": options or {},
        }
        started = time.perf_counter()
        request = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=3600) as response:
                body = response.read()
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Failed to call Ollama at {self.base_url}: {exc}") from exc

        latency_s = time.perf_counter() - started
        data = json.loads(body.decode("utf-8"))
        eval_duration_s = _ns_to_s(data.get("eval_duration"))
        prompt_eval_duration_s = _ns_to_s(data.get("prompt_eval_duration"))
        return OllamaResult(
            output=data.get("response", ""),
            latency_s=latency_s,
            generated_tokens=int(data.get("eval_count") or 0),
            prompt_tokens=int(data.get("prompt_eval_count") or 0),
            eval_duration_s=eval_duration_s,
            prompt_eval_duration_s=prompt_eval_duration_s,
        )


def _ns_to_s(value: int | float | None) -> float | None:
    if value is None:
        return None
    return float(value) / 1_000_000_000.0
