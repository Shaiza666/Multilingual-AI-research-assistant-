from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    page: int
    chunk_index: int
    text: str
    language: str
    source: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Chunk:
        return cls(
            chunk_id=str(data["chunk_id"]),
            doc_id=str(data["doc_id"]),
            title=str(data.get("title") or data["doc_id"]),
            page=int(data["page"]),
            chunk_index=int(data["chunk_index"]),
            text=str(data["text"]),
            language=str(data.get("language") or "unknown"),
            source=str(data.get("source") or ""),
        )


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float
    rank: int

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk.chunk_id,
            "doc_id": self.chunk.doc_id,
            "title": self.chunk.title,
            "page": self.chunk.page,
            "chunk_index": self.chunk.chunk_index,
            "text": self.chunk.text,
            "language": self.chunk.language,
            "source": self.chunk.source,
            "score": self.score,
            "rank": self.rank,
        }
