from __future__ import annotations

from difflib import SequenceMatcher


def exact_match(a: str, b: str) -> float:
    return float(a.strip() == b.strip())


def char_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.strip(), b.strip()).ratio()


def safe_tokens_per_second(tokens: int | None, seconds: float | None) -> float | None:
    if not tokens or not seconds or seconds <= 0:
        return None
    return tokens / seconds

