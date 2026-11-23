"""Retrieval and answer metrics.

Retrieval is page-oriented because the gold labels are supporting pages:

* Recall@k is the fraction of gold pages that appear in the top k chunks.
* MRR uses the chunk rank of the first chunk whose page is gold.
* nDCG@k walks the top k chunks, keeps the first time each page appears,
  pads that page list to length k, and uses binary page relevance. The ideal
  is min(k, number of gold pages) relevant pages at the top.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from mira.textnorm import normalize_for_index

_PUNCT = re.compile(r"[^\w\s\u0600-\u06ff]+", re.UNICODE)
_EN_ARTICLES = frozenset({"a", "an", "the"})


def dcg(relevances: list[float]) -> float:
    total = 0.0
    for index, rel in enumerate(relevances):
        total += float(rel) / math.log2(index + 2)
    return total


def pages_from_chunks(pages: list[tuple[str, int]], k: int) -> list[tuple[str, int]]:
    seen: set[tuple[str, int]] = set()
    ordered: list[tuple[str, int]] = []
    for page in pages[:k]:
        if page not in seen:
            seen.add(page)
            ordered.append(page)
    return ordered


def recall_at_k(pages: list[tuple[str, int]], gold: set[tuple[str, int]], k: int) -> float:
    if not gold:
        return 0.0
    found = set(pages_from_chunks(pages, k))
    return len(found & gold) / len(gold)


def reciprocal_rank(pages: list[tuple[str, int]], gold: set[tuple[str, int]]) -> float:
    for rank, page in enumerate(pages, start=1):
        if page in gold:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(pages: list[tuple[str, int]], gold: set[tuple[str, int]], k: int) -> float:
    if k <= 0 or not gold:
        return 0.0
    ordered = pages_from_chunks(pages, k)
    rels = [1 if page in gold else 0 for page in ordered]
    if len(rels) < k:
        rels.extend([0] * (k - len(rels)))
    ideal = dcg([1.0] * min(k, len(gold)))
    if ideal == 0.0:
        return 0.0
    return dcg(rels) / ideal


def hit_at_k(pages: list[tuple[str, int]], gold: set[tuple[str, int]], k: int) -> float:
    found = set(pages_from_chunks(pages, k))
    return 1.0 if found & gold else 0.0


def normalize_answer(text: str, language: str) -> str:
    folded = normalize_for_index(text)
    folded = _PUNCT.sub(" ", folded)
    tokens = [tok for tok in folded.split() if tok]
    if language == "en":
        tokens = [tok for tok in tokens if tok not in _EN_ARTICLES]
    return " ".join(tokens)


def token_f1(prediction: str, gold: str, language: str) -> float:
    pred_tokens = normalize_answer(prediction, language).split()
    gold_tokens = normalize_answer(gold, language).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, gold: str, language: str) -> float:
    return 1.0 if normalize_answer(prediction, language) == normalize_answer(gold, language) else 0.0


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def macro_average(rows: list[dict], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return mean(values)


def bootstrap_mean_ci(values: list[float], n_resamples: int = 1000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Percentile bootstrap of the mean. Returns None bounds when the sample is empty."""
    if not values:
        return {"mean": None, "low": None, "high": None, "n": 0}
    import numpy as np

    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.choice(array, size=(n_resamples, array.size), replace=True).mean(axis=1)
    draws.sort()
    low_index = int(alpha / 2 * n_resamples)
    high_index = min(n_resamples - 1, int((1 - alpha / 2) * n_resamples))
    return {
        "mean": float(array.mean()),
        "low": float(draws[low_index]),
        "high": float(draws[high_index]),
        "n": int(array.size),
    }
