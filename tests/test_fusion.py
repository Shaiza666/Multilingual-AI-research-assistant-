import numpy as np

from mira.indexing import IndexStore
from mira.retrieve import reciprocal_rank_fusion, search
from mira.types import Chunk


def test_rrf_known_order():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "c", "a"]], k=60)
    assert [item for item, _score in fused] == ["b", "a", "c"]
    scores = dict(fused)
    assert scores["b"] == (1 / 61) + (1 / 62)
    assert scores["a"] == (1 / 61) + (1 / 63)


def test_duplicate_ranks_in_one_list_count_once():
    fused = reciprocal_rank_fusion([["a", "a", "b"]], k=60)
    scores = dict(fused)
    assert scores["a"] == 1 / 61
    # The repeated id still occupies a rank, so b is third in the list.
    assert scores["b"] == 1 / 63


def _store() -> IndexStore:
    chunks = [
        Chunk("A::p1::c0", "A", "A", 1, 0, "zebra zebra zebra", "en", "A.pdf"),
        Chunk("B::p1::c0", "B", "B", 1, 0, "quartz quartz quartz", "en", "B.pdf"),
        Chunk("C::p1::c0", "C", "C", 1, 0, "unrelated words here now", "en", "C.pdf"),
    ]
    embeddings = np.eye(3, dtype=np.float32)
    return IndexStore(directory=__import__("pathlib").Path("."), chunks=chunks, embeddings=embeddings, meta={})


def test_hybrid_merges_keyword_and_dense_hits():
    store = _store()

    def embed_query(_text: str) -> np.ndarray:
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)

    ranked, translated = search(
        store,
        "zebra",
        method="hybrid",
        top_k=2,
        candidate_k=5,
        rrf_k=60,
        rerank_pool=5,
        embed_query=embed_query,
    )
    assert translated is None
    assert [hit.chunk.chunk_id for hit in ranked] == ["A::p1::c0", "B::p1::c0"]


def test_rerank_can_promote_a_lower_fused_hit():
    store = _store()

    def embed_query(_text: str) -> np.ndarray:
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)

    def rerank(_query: str, passages: list[str]) -> list[float]:
        return [1.0 if "unrelated" in text else 0.1 for text in passages]

    ranked, _translated = search(
        store,
        "zebra",
        method="hybrid_rerank",
        top_k=1,
        candidate_k=5,
        rrf_k=60,
        rerank_pool=5,
        embed_query=embed_query,
        rerank=rerank,
    )
    assert ranked[0].chunk.chunk_id == "C::p1::c0"


def test_urdu_translation_is_used_only_for_urdu_questions():
    store = _store()
    calls = {"n": 0}

    def embed_query(_text: str) -> np.ndarray:
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)

    def translate(text: str, direction: str) -> str:
        calls["n"] += 1
        assert direction == "ur-en"
        assert text
        return "zebra"

    ranked, translated = search(
        store,
        "زیبرا کہاں ہے اس متن میں",
        method="hybrid_translate",
        top_k=2,
        candidate_k=5,
        rrf_k=60,
        rerank_pool=5,
        embed_query=embed_query,
        translate=translate,
        query_language="ur",
    )
    assert calls["n"] == 1
    assert translated == "zebra"
    assert "A::p1::c0" in [hit.chunk.chunk_id for hit in ranked]

    calls["n"] = 0
    search(
        store,
        "zebra",
        method="hybrid_translate",
        top_k=2,
        candidate_k=5,
        rrf_k=60,
        rerank_pool=5,
        embed_query=embed_query,
        translate=translate,
        query_language="en",
    )
    assert calls["n"] == 0
