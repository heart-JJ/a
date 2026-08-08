from __future__ import annotations

import hashlib
import json
import math
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Iterable


LATIN_WORD_RE = re.compile(r"[a-zA-Z0-9_]+")
CJK_BLOCK_RE = re.compile(r"[\u3400-\u9fff]+")
SENTENCE_RE = re.compile(r"(?<=[。！？!?；;])\s*|\n+")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def json_loads(value: str | None, default: Any = None) -> Any:
    if value is None or value == "":
        return default
    return json.loads(value)


def content_hash(value: Any) -> str:
    return hashlib.sha256(json_dumps(value).encode("utf-8")).hexdigest()


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def text_tokens(text: str) -> set[str]:
    """Tokenize English and Chinese without third-party segmenters.

    CJK blocks contribute unigrams, bigrams and trigrams. This is intentionally
    deterministic so matching decisions can be replayed later.
    """

    normalized = text.lower().strip()
    tokens = set(LATIN_WORD_RE.findall(normalized))
    for block in CJK_BLOCK_RE.findall(normalized):
        if len(block) <= 3:
            tokens.add(block)
        for width in (1, 2, 3):
            if len(block) >= width:
                tokens.update(block[index : index + width] for index in range(len(block) - width + 1))
    return {token for token in tokens if token.strip()}


def similarity(left: str, right: str) -> float:
    a = text_tokens(left)
    b = text_tokens(right)
    if not a or not b:
        return 0.0
    overlap = len(a & b)
    union = len(a | b)
    jaccard = overlap / union
    containment = overlap / min(len(a), len(b))
    normalized_left = "".join(left.lower().split())
    normalized_right = "".join(right.lower().split())
    substring = 1.0 if normalized_left in normalized_right or normalized_right in normalized_left else 0.0
    return clamp(0.45 * jaccard + 0.45 * containment + 0.10 * substring)


def split_sentences(text: str) -> list[str]:
    return [part.strip() for part in SENTENCE_RE.split(text) if part.strip()]


def keyword_list(text: str, limit: int = 8) -> list[str]:
    stop = {
        "的", "了", "和", "是", "在", "把", "将", "要", "这", "那", "一个", "我们", "你", "我",
        "请", "进行", "以及", "可以", "需要", "the", "a", "an", "and", "or", "to", "of", "is", "in",
    }
    counts: dict[str, int] = {}
    latin = LATIN_WORD_RE.findall(text.lower())
    for token in latin:
        if len(token) > 1 and token not in stop:
            counts[token] = counts.get(token, 0) + 1
    for block in CJK_BLOCK_RE.findall(text):
        width = 2 if len(block) < 8 else 3
        for index in range(max(1, len(block) - width + 1)):
            token = block[index : index + width]
            if token and token not in stop:
                counts[token] = counts.get(token, 0) + 1
    ranked = sorted(counts, key=lambda token: (-counts[token], -len(token), token))
    return ranked[:limit]


def pattern_key(task: str, tags: Iterable[str] | None = None) -> str:
    clean_tags = sorted({tag.strip().lower() for tag in (tags or []) if tag.strip()})
    if clean_tags:
        return "tag:" + clean_tags[0]
    keywords = keyword_list(task, limit=4)
    if keywords:
        return "kw:" + "|".join(sorted(keywords))
    return "text:" + hashlib.sha1(task.strip().lower().encode("utf-8")).hexdigest()[:12]


def bayesian_reliability(successes: int, failures: int, alpha: float = 2.0, beta: float = 2.0) -> float:
    return (successes + alpha) / (successes + failures + alpha + beta)


def mastery_score(uses: int) -> float:
    return 1.0 - math.exp(-max(0, uses) / 10.0)


def slugify(value: str, fallback: str = "skill") -> str:
    latin = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    if latin:
        return latin[:48]
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]
    return f"{fallback}-{digest}"
