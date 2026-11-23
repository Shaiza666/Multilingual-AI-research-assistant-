"""PDF reading and page-aware chunk construction."""

from __future__ import annotations

import re
from pathlib import Path

from mira.chunking import chunk_pages, clean_page_text, remove_repeated_lines
from mira.errors import MiraError
from mira.textnorm import normalize_display
from mira.types import Chunk

_SAFE_ID = re.compile(r"[^A-Za-z0-9.\-]+")


def safe_doc_id(stem: str) -> str:
    cleaned = _SAFE_ID.sub("-", stem).strip("-")
    if not cleaned:
        raise MiraError(f"Cannot derive a document id from '{stem}'")
    return cleaned


def iter_pdfs(paths: list[str | Path]) -> list[Path]:
    found: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            found.extend(sorted(p for p in path.glob("*.pdf") if p.is_file()))
        elif path.is_file() and path.suffix.lower() == ".pdf":
            found.append(path)
        else:
            raise MiraError(f"Not a PDF file or directory: {path}")
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in found:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def extract_pages(path: Path) -> tuple[str, list[str]]:
    import pymupdf

    document = pymupdf.open(path)
    try:
        title = normalize_display(document.metadata.get("title") or "") or path.stem
        raw_pages = [page.get_text("text") or "" for page in document]
    finally:
        document.close()
    return title, raw_pages


def ingest_pdf(
    path: Path,
    chunk_words: int = 160,
    overlap_words: int = 40,
    min_words: int = 8,
    doc_id: str | None = None,
) -> list[Chunk]:
    title, raw_pages = extract_pages(path)
    stripped = remove_repeated_lines(raw_pages)
    pages: list[tuple[int, str]] = []
    for number, text in enumerate(stripped, start=1):
        cleaned = clean_page_text(text)
        if len(cleaned.split()) < min_words:
            continue
        pages.append((number, cleaned))
    if not pages:
        raise MiraError(f"No extractable text in {path.name}")
    return chunk_pages(
        doc_id=doc_id or safe_doc_id(path.stem),
        title=title,
        pages=pages,
        source=path.name,
        chunk_words=chunk_words,
        overlap_words=overlap_words,
        min_words=min_words,
    )


def ingest_paths(paths: list[str | Path], chunk_words: int, overlap_words: int, min_words: int) -> list[Chunk]:
    files = iter_pdfs(paths)
    if not files:
        raise MiraError("No PDF files found")
    chunks: list[Chunk] = []
    seen_docs: set[str] = set()
    for path in files:
        doc_id = safe_doc_id(path.stem)
        if doc_id in seen_docs:
            raise MiraError(f"Duplicate document id '{doc_id}' from {path.name}")
        seen_docs.add(doc_id)
        chunks.extend(
            ingest_pdf(
                path,
                chunk_words=chunk_words,
                overlap_words=overlap_words,
                min_words=min_words,
                doc_id=doc_id,
            )
        )
    return chunks
