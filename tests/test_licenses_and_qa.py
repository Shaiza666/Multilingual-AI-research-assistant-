import json
import re
from pathlib import Path

from mira.licenses import classify_license
from mira.qa import assert_disjoint, load_qa
from mira.textnorm import detect_language


def test_license_classifier():
    assert classify_license("License: CC-BY 4.0, see https://creativecommons.org/licenses/by/4.0/.") == "CC BY 4.0"
    assert classify_license("published under Creative Commons CC-BY 4.0 License") == "CC BY 4.0"
    assert classify_license("https://creativecommons.org/licenses/by-nc/4.0/") is None
    assert classify_license("all rights reserved") is None


def test_question_files_are_disjoint_and_well_formed():
    pilot, _pilot_meta = load_qa("data/qa/pilot.json", "pilot")
    test, meta = load_qa("data/qa/gold.json", "test")
    assert_disjoint(pilot, test)
    assert len(test) == meta["counts"]["test_items"]
    assert len(pilot) == 8
    for item in test + pilot:
        if item.language == "ur":
            assert detect_language(item.question) == "ur"
        if item.answerable:
            assert item.answer
            assert item.evidence_quotes
            assert item.gold_pages
        else:
            assert item.gold_pages == []
            assert item.query_type == "unanswerable"
    links = {item.id for item in test}
    for item in test:
        if item.translation_of:
            assert item.translation_of in links


def test_evidence_quotes_occur_on_the_cited_pages():
    root = Path("data/papers/text")
    if not any(root.glob("*.txt")):
        return
    pages: dict[tuple[str, int], str] = {}
    for path in root.glob("*.txt"):
        raw = path.read_text(encoding="utf-8")
        for part in raw.split("===== PAGE ")[1:]:
            head, _, body = part.partition("=====")
            number = int(head.strip().split()[0])
            pages[(path.stem, number)] = re.sub(r"\s+", " ", body).lower()
    for split, filename in (("test", "data/qa/gold.json"), ("pilot", "data/qa/pilot.json")):
        items, _meta = load_qa(filename, split)
        for item in items:
            if not item.answerable:
                continue
            for quote in item.evidence_quotes:
                folded = re.sub(r"\s+", " ", quote).lower()
                found = any(folded in pages.get((doc_id, page), "") for doc_id, page in item.gold_pages)
                assert found, f"{item.id} quote not on gold pages: {quote}"


def test_unanswerable_topics_are_absent_from_the_extracts():
    root = Path("data/papers/text")
    blob = "\n".join(path.read_text(encoding="utf-8").lower() for path in root.glob("*.txt"))
    if not blob:
        return
    for term in ("bleu", "vaccine", "llama", "photosynthesis", "shakespeare"):
        assert term not in blob
    data = json.loads(Path("data/qa/gold.json").read_text(encoding="utf-8"))
    assert data["counts"]["papers"] == 3
