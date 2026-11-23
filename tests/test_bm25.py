from mira.indexing import IndexStore
from mira.types import Chunk


def _chunk(doc_id: str, page: int, index: int, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"{doc_id}::p{page}::c{index}",
        doc_id=doc_id,
        title=doc_id,
        page=page,
        chunk_index=index,
        text=text,
        language="en",
        source=f"{doc_id}.pdf",
    )


def test_bm25_ranks_keyword_overlap_first():
    chunks = [
        _chunk("a", 1, 0, "transformer attention mechanism for documents"),
        _chunk("b", 1, 0, "urdu poetry and gardens"),
        _chunk("c", 1, 0, "attention is all you need in the transformer"),
    ]
    store = IndexStore(directory=__import__("pathlib").Path("."), chunks=chunks, embeddings=None, meta={})
    ranking = store.bm25_search("transformer attention", k=3)
    ids = [chunk_id for chunk_id, _score in ranking]
    assert ids[0] in {"a::p1::c0", "c::p1::c0"}
    assert "b::p1::c0" not in ids


def test_bm25_matches_urdu_after_letter_folding():
    chunks = [
        _chunk("u", 1, 0, "يہ كتاب اردو مشین ترجمہ کے بارے میں ہے"),
        _chunk("e", 1, 0, "english poetry only"),
    ]
    store = IndexStore(directory=__import__("pathlib").Path("."), chunks=chunks, embeddings=None, meta={})
    ranking = store.bm25_search("یہ کتاب مشین", k=2)
    assert ranking[0][0] == "u::p1::c0"
