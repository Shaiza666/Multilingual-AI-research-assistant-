"""Chunk store, BM25 view, and a FAISS inner-product index over normalised embeddings."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mira.errors import MiraError
from mira.textnorm import tokenize
from mira.types import Chunk


class IndexStore:
    def __init__(self, directory: Path, chunks: list[Chunk], embeddings: np.ndarray | None, meta: dict):
        self.directory = Path(directory)
        self.chunks = chunks
        self.by_id = {chunk.chunk_id: chunk for chunk in chunks}
        if len(self.by_id) != len(chunks):
            raise MiraError("Chunk ids are not unique")
        self.embeddings = embeddings
        self.meta = meta
        self._bm25 = None
        self._faiss = None
        self._row = {chunk.chunk_id: i for i, chunk in enumerate(chunks)}

    def save(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        chunks_path = self.directory / "chunks.jsonl"
        with chunks_path.open("w", encoding="utf-8") as handle:
            for chunk in self.chunks:
                handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
        if self.embeddings is not None:
            np.save(self.directory / "embeddings.npy", np.asarray(self.embeddings, dtype=np.float32))
        elif (self.directory / "embeddings.npy").exists():
            (self.directory / "embeddings.npy").unlink()
        (self.directory / "meta.json").write_text(
            json.dumps(self.meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: str | Path) -> IndexStore:
        path = Path(directory)
        chunks_path = path / "chunks.jsonl"
        meta_path = path / "meta.json"
        if not chunks_path.exists() or not meta_path.exists():
            raise MiraError(f"Index not found in {path}. Run ingest first.")
        chunks = [
            Chunk.from_dict(json.loads(line))
            for line in chunks_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        embeddings = None
        emb_path = path / "embeddings.npy"
        if emb_path.exists():
            embeddings = np.load(emb_path)
        return cls(path, chunks, embeddings, meta)

    def bm25(self):
        if self._bm25 is None:
            from rank_bm25 import BM25L

            # BM25L keeps a positive IDF on small collections. Okapi's IDF is
            # zero when a term appears in about half of the chunks, which is
            # common in a short paper index. Documents that contain none of the
            # query terms still score zero.
            tokenized = [tokenize(chunk.text) or ["__empty__"] for chunk in self.chunks]
            self._bm25 = BM25L(tokenized)
        return self._bm25

    def bm25_search(self, query: str, k: int) -> list[tuple[str, float]]:
        if not self.chunks or k <= 0:
            return []
        scores = np.asarray(self.bm25().get_scores(tokenize(query)), dtype=np.float64)
        if scores.size == 0:
            return []
        order = np.argsort(-scores)
        pairs: list[tuple[str, float]] = []
        for index in order[:k]:
            score = float(scores[index])
            if score <= 0:
                break
            pairs.append((self.chunks[int(index)].chunk_id, score))
        return pairs

    def _faiss_index(self):
        if self.embeddings is None:
            raise MiraError("This index has no embeddings. Rebuild it with the embedding model available.")
        if self._faiss is None:
            import faiss

            matrix = np.ascontiguousarray(self.embeddings, dtype=np.float32)
            index = faiss.IndexFlatIP(matrix.shape[1])
            index.add(matrix)
            self._faiss = index
        return self._faiss

    def dense_search(self, vector: np.ndarray, k: int) -> list[tuple[str, float]]:
        if self.embeddings is None:
            raise MiraError("This index has no embeddings. Rebuild it with the embedding model available.")
        if k <= 0:
            return []
        query = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        index = self._faiss_index()
        scores, ids = index.search(query, min(k, len(self.chunks)))
        pairs: list[tuple[str, float]] = []
        for score, row in zip(scores[0], ids[0]):
            if int(row) < 0:
                continue
            pairs.append((self.chunks[int(row)].chunk_id, float(score)))
        return pairs

    def vector_for(self, chunk_id: str) -> np.ndarray | None:
        if self.embeddings is None or chunk_id not in self._row:
            return None
        return self.embeddings[self._row[chunk_id]]


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return (matrix / norms).astype(np.float32)


def build_index(chunks: list[Chunk], directory: str | Path, cfg, embed: bool = True) -> IndexStore:
    """Persist chunks and, when requested, unit-normalised passage embeddings."""
    embeddings = None
    if embed:
        from mira.models import build_embeddings

        embeddings = l2_normalize(build_embeddings(cfg, chunks))
    meta = {
        "embedding_model": cfg.embedding.model_name if embed else None,
        "query_prefix": cfg.embedding.query_prefix,
        "passage_prefix": cfg.embedding.passage_prefix,
        "chunk_words": cfg.chunk.chunk_words,
        "overlap_words": cfg.chunk.overlap_words,
        "chunk_count": len(chunks),
        "documents": sorted({chunk.doc_id for chunk in chunks}),
        "search_backend": "faiss",
    }
    store = IndexStore(Path(directory), chunks, embeddings, meta)
    store.save()
    return store


def write_chunk_index(directory: str | Path, chunks: list[Chunk], meta: dict | None = None) -> IndexStore:
    store = IndexStore(Path(directory), chunks, None, meta or {"embedding_model": None})
    store.save()
    return store
