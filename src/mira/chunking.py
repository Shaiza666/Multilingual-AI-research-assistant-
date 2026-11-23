"""Page-aware word-window chunking and light PDF text cleaning."""

from __future__ import annotations

import re
from collections import Counter

from mira.errors import MiraError
from mira.textnorm import detect_language, normalize_display
from mira.types import Chunk

_PAGE_NO = re.compile(r"^\d{1,4}$")


def dehyphenate(text: str) -> str:
    """Join line-break hyphenation when the continuation is a lowercase letter.

    Real hyphens such as ``ML-1M`` split across a line are left in place.
    """
    return re.sub(r"-\n(?=[a-z])", "", text)


def remove_repeated_lines(pages: list[str]) -> list[str]:
    """Drop running headers, footers, and bare page numbers."""
    if len(pages) < 3:
        cleaned = []
        for page in pages:
            lines = [ln.strip() for ln in page.splitlines() if ln.strip() and not _PAGE_NO.match(ln.strip())]
            cleaned.append("\n".join(lines))
        return cleaned
    parsed = [[ln.strip() for ln in page.splitlines() if ln.strip()] for page in pages]
    counts: Counter[str] = Counter()
    for lines in parsed:
        counts.update(set(lines))
    threshold = max(3, int(0.6 * len(pages)))
    repeated = {line for line, count in counts.items() if count >= threshold and len(line.split()) <= 12}
    cleaned = []
    for lines in parsed:
        kept = [ln for ln in lines if ln not in repeated and not _PAGE_NO.match(ln)]
        cleaned.append("\n".join(kept))
    return cleaned


def clean_page_text(text: str) -> str:
    text = dehyphenate(text.replace("\r\n", "\n").replace("\r", "\n"))
    text = text.replace("\n", " ")
    text = normalize_display(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_words(text: str, chunk_words: int, overlap_words: int) -> list[str]:
    if chunk_words < 1:
        raise MiraError("chunk_words must be positive")
    if overlap_words < 0 or overlap_words >= chunk_words:
        raise MiraError("overlap_words must be >= 0 and smaller than chunk_words")
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_words:
        return [" ".join(words)]
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(len(words), start + chunk_words)
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        nxt = end - overlap_words
        if nxt <= start:
            nxt = start + 1
        start = nxt
    return chunks


def chunk_pages(
    doc_id: str,
    title: str,
    pages: list[tuple[int, str]],
    source: str,
    chunk_words: int = 160,
    overlap_words: int = 40,
    min_words: int = 8,
) -> list[Chunk]:
    """Chunk already-cleaned page strings. ``pages`` items are (1-based page, text)."""
    chunks: list[Chunk] = []
    for page_no, text in pages:
        pieces = split_words(text, chunk_words, overlap_words)
        kept_index = 0
        for piece in pieces:
            if len(piece.split()) < min_words:
                continue
            chunk_id = f"{doc_id}::p{page_no}::c{kept_index}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    title=title,
                    page=page_no,
                    chunk_index=kept_index,
                    text=piece,
                    language=detect_language(piece),
                    source=source,
                )
            )
            kept_index += 1
    return chunks
