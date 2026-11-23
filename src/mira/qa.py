"""Load and validate the benchmark question files."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from mira.errors import MiraError


@dataclass
class QAItem:
    id: str
    language: str
    question: str
    answer: str
    answer_en: str
    gold_pages: list[tuple[str, int]]
    answerable: bool
    translation_of: str | None
    query_type: str
    evidence_quotes: list[str] = field(default_factory=list)
    split: str = "test"
    corpus: str = "papers"

    def gold_set(self) -> set[tuple[str, int]]:
        return set(self.gold_pages)


def _parse_item(raw: dict, split: str) -> QAItem:
    pages = []
    for entry in raw.get("gold_pages") or []:
        pages.append((str(entry["doc_id"]), int(entry["page"])))
    answerable = bool(raw["answerable"])
    if answerable and not pages:
        raise MiraError(f"{raw.get('id')}: answerable items need gold pages")
    if not answerable and pages:
        raise MiraError(f"{raw.get('id')}: unanswerable items must not have gold pages")
    language = str(raw["language"])
    if language not in {"en", "ur"}:
        raise MiraError(f"{raw.get('id')}: language must be en or ur")
    return QAItem(
        id=str(raw["id"]),
        language=language,
        question=str(raw["question"]).strip(),
        answer=str(raw.get("answer") or "").strip(),
        answer_en=str(raw.get("answer_en") or raw.get("answer") or "").strip(),
        gold_pages=pages,
        answerable=answerable,
        translation_of=raw.get("translation_of"),
        query_type=str(raw.get("query_type") or ("unanswerable" if not answerable else "lexical")),
        evidence_quotes=[str(q) for q in raw.get("evidence_quotes") or []],
        split=split,
        corpus=str(raw.get("corpus") or "papers"),
    )


def load_qa(path: str | Path, split: str) -> tuple[list[QAItem], dict]:
    file_path = Path(path)
    if not file_path.exists():
        raise MiraError(f"Question file not found: {file_path}")
    data = json.loads(file_path.read_text(encoding="utf-8"))
    items = [_parse_item(row, split) for row in data.get("items") or []]
    ids = [item.id for item in items]
    if len(ids) != len(set(ids)):
        raise MiraError(f"Duplicate question ids in {file_path}")
    return items, data


def assert_disjoint(pilot: list[QAItem], test: list[QAItem]) -> None:
    pilot_ids = {item.id for item in pilot}
    overlap = pilot_ids & {item.id for item in test}
    if overlap:
        raise MiraError(f"Pilot and test share ids: {sorted(overlap)}")
    pilot_q = {item.question.strip() for item in pilot}
    shared_q = [item.id for item in test if item.question.strip() in pilot_q]
    if shared_q:
        raise MiraError(f"Pilot and test share question text: {shared_q}")
