"""BM25, dense, reciprocal rank fusion, translation, and reranking."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from mira.types import ScoredChunk

METHODS = (
    "bm25",
    "dense",
    "hybrid",
    "hybrid_rerank",
    "hybrid_translate",
    "hybrid_translate_rerank",
)


def reciprocal_rank_fusion(rankings: Sequence[Sequence[str]], k: int = 60) -> list[tuple[str, float]]:
    """Fuse ranked id lists. Earlier ranks contribute more. Ties break by id."""
    if k < 0:
        raise ValueError("rrf k must be non-negative")
    scores: dict[str, float] = {}
    for ranking in rankings:
        seen: set[str] = set()
        for rank, item_id in enumerate(ranking, start=1):
            if item_id in seen:
                continue
            seen.add(item_id)
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
    ordered = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
    return ordered


def top_ids(pairs: Sequence[tuple[str, float]], k: int) -> list[str]:
    return [item_id for item_id, _score in list(pairs)[:k]]


def _as_scored(index_by_id: dict, pairs: Sequence[tuple[str, float]]) -> list[ScoredChunk]:
    scored: list[ScoredChunk] = []
    for rank, (chunk_id, score) in enumerate(pairs, start=1):
        chunk = index_by_id.get(chunk_id)
        if chunk is None:
            continue
        scored.append(ScoredChunk(chunk=chunk, score=float(score), rank=rank))
    return scored


def search(
    index,
    query: str,
    method: str,
    top_k: int,
    candidate_k: int,
    rrf_k: int,
    rerank_pool: int,
    embed_query: Callable[[str], np.ndarray] | None = None,
    rerank: Callable[[str, list[str]], list[float]] | None = None,
    translate: Callable[[str, str], str] | None = None,
    query_language: str | None = None,
) -> tuple[list[ScoredChunk], str | None]:
    """Return ranked chunks and the English query when translation was used."""
    if method not in METHODS:
        raise ValueError(f"Unknown retrieval method: {method}")
    from mira.textnorm import detect_language

    language = query_language or detect_language(query)
    translated: str | None = None
    if method in {"hybrid_translate", "hybrid_translate_rerank"} and language == "ur" and translate is not None:
        translated = translate(query, "ur-en").strip() or None

    def bm25_ids(text: str) -> list[tuple[str, float]]:
        return index.bm25_search(text, candidate_k)

    def dense_ids(text: str) -> list[tuple[str, float]]:
        if embed_query is None:
            raise RuntimeError("Dense retrieval needs a query embedder")
        vector = np.asarray(embed_query(text), dtype=np.float32)
        return index.dense_search(vector, candidate_k)

    lists: list[list[str]] = []
    score_maps: dict[str, float] = {}

    if method == "bm25":
        pairs = bm25_ids(query)[:top_k]
        return _as_scored(index.by_id, pairs), translated
    if method == "dense":
        pairs = dense_ids(query)[:top_k]
        return _as_scored(index.by_id, pairs), translated

    lists.append(top_ids(bm25_ids(query), candidate_k))
    lists.append(top_ids(dense_ids(query), candidate_k))
    if translated:
        lists.append(top_ids(bm25_ids(translated), candidate_k))
        lists.append(top_ids(dense_ids(translated), candidate_k))

    fused = reciprocal_rank_fusion(lists, k=rrf_k)
    for chunk_id, score in fused:
        score_maps[chunk_id] = score

    use_rerank = method in {"hybrid_rerank", "hybrid_translate_rerank"}
    if use_rerank:
        pool_ids = [chunk_id for chunk_id, _score in fused[:rerank_pool]]
        passages = [index.by_id[chunk_id].text for chunk_id in pool_ids]
        rerank_query = translated or query
        if rerank is None:
            raise RuntimeError("Reranking needs a cross-encoder")
        raw_scores = list(rerank(rerank_query, passages))
        order = sorted(range(len(pool_ids)), key=lambda i: (-float(raw_scores[i]), pool_ids[i]))
        pairs = [(pool_ids[i], float(raw_scores[i])) for i in order[:top_k]]
        return _as_scored(index.by_id, pairs), translated

    pairs = fused[:top_k]
    return _as_scored(index.by_id, pairs), translated
